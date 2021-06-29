
import logging
from typing import Dict, List, Optional
import copy

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

# Import submodules.
from sodner.models.ner import NERTagger
from sodner.models.relation import RelationExtractor
from sodner.models.gat import AGGCN

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("sodner")
class SodNER(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_layer: Seq2SeqEncoder,
                 modules,
                 feature_size: int,
                 max_span_width: int,
                 loss_weights: Dict[str, int],
                 use_dep: bool,
                 lexical_dropout: float = 0.2,
                 lstm_dropout: float = 0.4,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 display_metrics: List[str] = None,
                 span_extractor: SpanExtractor = None) -> None:
        super(SodNER, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer

        self._loss_weights = loss_weights
        self._permanent_loss_weights = copy.deepcopy(self._loss_weights)

        modules = Params(modules)

        self._ner = NERTagger.from_params(vocab=vocab, params=modules.pop("ner"))
        self._relation = RelationExtractor.from_params(vocab=vocab, params=modules.pop("relation"))

        # Make endpoint span extractor.
        self._endpoint_span_extractor = span_extractor

        self._max_span_width = max_span_width

        self._display_metrics = display_metrics

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        if lstm_dropout > 0:
            self._lstm_dropout = torch.nn.Dropout(p=lstm_dropout)
        else:
            self._lstm_dropout = lambda x: x

        self.use_dep = use_dep
        if self.use_dep:
            self._dep_tree = AGGCN.from_params(vocab=vocab, params=modules.pop("gat_tree"))


        initializer(self)

    @overrides
    def forward(self,
                text,
                spans,
                ner_labels,
                relation_labels,
                metadata,
                dep_span_children,
                ):

        # In AllenNLP, AdjacencyFields are passed in as floats. This fixes it.
        relation_labels = relation_labels.long()

        text_embeddings = self._text_field_embedder(text)

        text_embeddings = self._lexical_dropout(text_embeddings)

        # Shape: (batch_size, max_sentence_length)
        text_mask = util.get_text_field_mask(text).float()

        sentence_lengths = 0*text_mask.sum(dim=1).long()
        for i in range(len(metadata)):
            sentence_lengths[i] = metadata[i]["end_ix"] - metadata[i]["start_ix"]

        # Shape: (batch_size, max_sentence_length, encoding_dim)
        contextualized_embeddings = self._lstm_dropout(self._context_layer(text_embeddings, text_mask))
        assert spans.max() < contextualized_embeddings.shape[1]

        if self.use_dep:
            dep_span_children = dep_span_children + 1
            dep_feature_embeddings = self._dep_tree(dep_span_children, contextualized_embeddings, text_mask)
            contextualized_embeddings = torch.cat([contextualized_embeddings, dep_feature_embeddings], dim=-1)

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).float()
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans, text_mask)

        output_ner = {'loss': 0}
        output_relation = {'loss': 0}

        # Prune and compute span representations for relation module
        if self._loss_weights["relation"] > 0 or self._relation.rel_prop > 0:
            output_relation = self._relation.compute_representations(
                spans, span_mask, span_embeddings, sentence_lengths, relation_labels, metadata)

        if self._relation.rel_prop > 0:
            output_relation = self._relation.relation_propagation(output_relation)
            span_embeddings = self.update_span_embeddings(span_embeddings, span_mask,
                output_relation["top_span_embeddings"], output_relation["top_span_mask"],
                output_relation["top_span_indices"])

        # Make predictions and compute losses for each module
        if self._loss_weights['ner'] > 0:
            output_ner = self._ner(
                spans, span_mask, span_embeddings, sentence_lengths, ner_labels, metadata)
            self._ner.decode(output_ner)

        if self._loss_weights['relation'] > 0:
            output_relation = self._relation.predict_labels(relation_labels, output_relation, metadata, output_ner)

        if "loss" not in output_relation:
            output_relation["loss"] = 0

        loss = (self._loss_weights['ner'] * output_ner['loss'] +
                self._loss_weights['relation'] * output_relation['loss'])

        output_dict = dict(
                           relation=output_relation,
                           ner=output_ner,
                           )
        output_dict['loss'] = loss

        return output_dict

    def update_span_embeddings(self, span_embeddings, span_mask, top_span_embeddings, top_span_mask, top_span_indices):

        new_span_embeddings = span_embeddings.clone()
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if top_span_mask[sample_nr, top_span_nr] == 0 or span_mask[sample_nr, span_nr] == 0:
                    break
                new_span_embeddings[sample_nr, span_nr] = top_span_embeddings[sample_nr, top_span_nr]
        return new_span_embeddings

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        res = {}
        res["ner"] = output_dict['ner']['decoded_ner']
        res["relation"] = output_dict['relation']['decoded_relations']
        return res

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._relation.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = (list(metrics_ner.keys()) +
                        list(metrics_relation.keys()))
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(
                           list(metrics_ner.items()) +
                           list(metrics_relation.items())
                           )

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res
