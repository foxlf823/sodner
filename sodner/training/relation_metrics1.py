from overrides import overrides

from allennlp.training.metrics.metric import Metric

from sodner.training.f1 import compute_f1

def is_clique(entity, relations):
    entity = list(entity)

    for idx, fragment1 in enumerate(entity):
        for idy, fragment2 in enumerate(entity):
            if idx < idy:
                if (fragment1, fragment2) not in relations and (fragment2, fragment1) not in relations:
                    return False

    return True

class RelationMetrics1(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold spans.
    """
    def __init__(self):
        self.reset()

    @overrides
    def __call__(self, predicted_relation_list, metadata_list, predicted_ner_list):
        for predicted_relations, metadata in zip(predicted_relation_list, metadata_list):
            gold_relations = metadata["relation_dict"]
            self._total_gold += len(gold_relations)
            self._total_predicted += len(predicted_relations)
            for (span_1, span_2), label in predicted_relations.items():
                ix = (span_1, span_2)
                if ix in gold_relations and gold_relations[ix] == label:
                    self._total_matched += 1

        # get gold entities from metadata
        gold_entities = []
        for metadata in metadata_list:
            entities = []
            for ner_span, _ in metadata['ner_dict'].items():
                entity = set()
                entity.add(ner_span)
                for (arg1_span, arg2_span), label in metadata['relation_dict'].items():
                    # Overlap
                    if label != 'Combined':
                        continue
                    if ner_span == arg1_span:
                        entity.add(arg2_span)
                    if ner_span == arg2_span:
                        entity.add(arg1_span)
                if entity not in entities:
                    if is_clique(entity, metadata['relation_dict']):
                        entities.append(entity)
            gold_entities.append(entities)

        # get predict entities from predicted_ner_list and predicted_relation_list
        predict_entities = []
        for predicted_ner, predicted_relation in zip(predicted_ner_list, predicted_relation_list):
            entities = []
            for ner_span, _ in predicted_ner.items():
                entity = set()
                entity.add(ner_span)
                for (arg1_span, arg2_span), label in predicted_relation.items():
                    # Overlap
                    if label != 'Combined':
                        continue
                    if ner_span == arg1_span:
                        entity.add(arg2_span)
                    if ner_span == arg2_span:
                        entity.add(arg1_span)
                if entity not in entities:
                    if is_clique(entity, predicted_relation):
                        entities.append(entity)
            predict_entities.append(entities)

        # compute p r f1
        for golds, predicts in zip(gold_entities, predict_entities):
            self._real_ner_gold += len(golds)
            self._real_ner_predicted += len(predicts)
            for predict in predicts:
                if predict in golds:
                    self._real_ner_correct += 1


    @overrides
    def get_metric(self, reset=False):
        precision, recall, f1 = compute_f1(self._total_predicted, self._total_gold, self._total_matched)

        real_ner_precision, real_ner_recall, real_ner_f1 = compute_f1(self._real_ner_predicted, self._real_ner_gold, self._real_ner_correct)
        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return precision, recall, f1, real_ner_precision, real_ner_recall, real_ner_f1

    @overrides
    def reset(self):
        self._total_gold = 0
        self._total_predicted = 0
        self._total_matched = 0

        self._real_ner_gold = 0
        self._real_ner_predicted = 0
        self._real_ner_correct = 0


class CandidateRecall(Metric):
    """
    Computes relation candidate recall.
    """
    def __init__(self):
        self.reset()

    def __call__(self, predicted_relation_list, metadata_list):
        for predicted_relations, metadata in zip(predicted_relation_list, metadata_list):
            gold_spans = set(metadata["relation_dict"].keys())
            candidate_spans = set(predicted_relations.keys())
            self._total_gold += len(gold_spans)
            self._total_matched += len(gold_spans & candidate_spans)

    @overrides
    def get_metric(self, reset=False):
        recall = self._total_matched / self._total_gold if self._total_gold > 0 else 0

        if reset:
            self.reset()

        return recall

    @overrides
    def reset(self):
        self._total_gold = 0
        self._total_matched = 0
