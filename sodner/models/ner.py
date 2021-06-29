import logging
from typing import Any, Dict, List, Optional

import torch
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from sodner.training.ner_metrics import NERMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NERTagger(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 mention_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        self._n_labels = vocab.get_vocab_size('ner_labels')

        null_label = vocab.get_token_index("", "ner_labels")
        assert null_label == 0  # If not, the dummy class won't correspond to the null label.

        self._ner_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(
                mention_feedforward.get_output_dim(),
                self._n_labels - 1)))

        self._ner_metrics = NERMetrics(self._n_labels, null_label)

        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask: torch.IntTensor,
                span_embeddings: torch.IntTensor,
                sentence_lengths: torch.Tensor,
                ner_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                previous_step_output: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:


        # Shape: (Batch size, Number of Spans, Span Embedding Size)
        # span_embeddings
        ner_scores = self._ner_scorer(span_embeddings)
        # Give large negative scores to masked-out elements.
        mask = span_mask.unsqueeze(-1)
        ner_scores = util.replace_masked_values(ner_scores, mask, -1e20)
        dummy_dims = [ner_scores.size(0), ner_scores.size(1), 1]
        dummy_scores = ner_scores.new_zeros(*dummy_dims)
        if previous_step_output is not None and "predicted_span" in previous_step_output and not self.training:
            dummy_scores.masked_fill_(previous_step_output["predicted_span"].bool().unsqueeze(-1), -1e20)
            dummy_scores.masked_fill_((1-previous_step_output["predicted_span"]).bool().unsqueeze(-1), 1e20)

        ner_scores = torch.cat((dummy_scores, ner_scores), -1)

        if previous_step_output is not None and "predicted_seq_span" in previous_step_output and not self.training:
            for row_idx, all_spans in enumerate(spans):
                pred_spans = previous_step_output["predicted_seq_span"][row_idx]
                pred_spans = all_spans.new_tensor(pred_spans)
                for col_idx, span in enumerate(all_spans):
                    if span_mask[row_idx][col_idx] == 0:
                        continue
                    bFind = False
                    for pred_span in pred_spans:
                        if span[0] == pred_span[0] and span[1] == pred_span[1]:
                            bFind = True
                            break
                    if bFind:
                        # if find, use the ner scores, set dummy to a big negative
                        ner_scores[row_idx, col_idx, 0] = -1e20
                    else:
                        # if not find, use the previous step, set dummy to a big positive
                        ner_scores[row_idx, col_idx, 0] = 1e20

        _, predicted_ner = ner_scores.max(2)

        output_dict = {"spans": spans,
                       "span_mask": span_mask,
                       "ner_scores": ner_scores,
                       "predicted_ner": predicted_ner}

        if ner_labels is not None:
            self._ner_metrics(predicted_ner, ner_labels, span_mask)
            ner_scores_flat = ner_scores.view(-1, self._n_labels)
            ner_labels_flat = ner_labels.view(-1)
            mask_flat = span_mask.view(-1).bool()

            loss = self._loss(ner_scores_flat[mask_flat], ner_labels_flat[mask_flat])
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["document"] = [x["sentence"] for x in metadata]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        predicted_ner_batch = output_dict["predicted_ner"].detach().cpu()
        spans_batch = output_dict["spans"].detach().cpu()
        span_mask_batch = output_dict["span_mask"].detach().cpu().bool()

        res_list = []
        res_dict = []
        for spans, span_mask, predicted_NERs in zip(spans_batch, span_mask_batch, predicted_ner_batch):
            entry_list = []
            entry_dict = {}
            for span, ner in zip(spans[span_mask], predicted_NERs[span_mask]):
                ner = ner.item()
                if ner > 0:
                    the_span = (span[0].item(), span[1].item())
                    the_label = self.vocab.get_token_from_index(ner, "ner_labels")
                    entry_list.append((the_span[0], the_span[1], the_label))
                    entry_dict[the_span] = the_label
            res_list.append(entry_list)
            res_dict.append(entry_dict)

        output_dict["decoded_ner"] = res_list
        output_dict["decoded_ner_dict"] = res_dict
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ner_precision, ner_recall, ner_f1 = self._ner_metrics.get_metric(reset)
        return {"ner_precision": ner_precision,
                "ner_recall": ner_recall,
                "ner_f1": ner_f1}
