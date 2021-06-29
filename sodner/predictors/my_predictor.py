
from allennlp.predictors.predictor import Predictor
from overrides import overrides
from allennlp.common.util import JsonDict
import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union

from overrides import overrides




class MissingDict(dict):
    """
    If key isn't there, returns default value. Like defaultdict, but it doesn't store the missing
    keys that were queried.
    """
    def __init__(self, missing_val, generator=None) -> None:
        if generator:
            super().__init__(generator)
        else:
            super().__init__()
        self._missing_val = missing_val

    def __missing__(self, key):
        return self._missing_val


def format_label_fields(sentence: List[str],
                        ner: List[List[Union[int,str]]],
                        relations: List[List[Union[int,str]]],
                        sentence_start: int,
                        dep_tree: Dict[str, Any],
                        use_overlap_rel: bool) -> Tuple[Dict[Tuple[int,int],str],
                                                      Dict[Tuple[Tuple[int,int],Tuple[int,int]],str],
                                                      Dict[Tuple[int,int],int], Dict[Tuple[int, int],str],
                                                    Dict[Tuple[int, int],List[Tuple[int, int]]],
                                                    Dict[str,Any]]:
    ss = sentence_start
    # NER
    ner_dict = MissingDict("",
        (
            ((span_start-ss, span_end-ss), named_entity)
            for (span_start, span_end, named_entity) in ner
        )
    )

    # Relations
    relation_dict_values = []
    for (span1_start, span1_end, span2_start, span2_end, relation) in relations:
        if relation == 'Overlap' and not use_overlap_rel:
            continue
        relation_dict_values.append((((span1_start - ss, span1_end - ss), (span2_start - ss, span2_end - ss)), relation))
    relation_dict = MissingDict("", relation_dict_values)



    if 'nodes' in dep_tree:
        dep_children_dict = MissingDict("",
                                    (
                                        ((node_idx, adj_node_idx), "1")
                                        for node_idx, adj_node_idxes in enumerate(dep_tree['nodes']) for adj_node_idx in adj_node_idxes
                                    )
                                    )
    else:
        dep_children_dict = MissingDict("")

    return ner_dict, relation_dict, dep_children_dict

@Predictor.register('my_predictor')
class MyPredictor(Predictor):

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        js = json_dict
        sentence_start = 0
        doc_key = js["doc_key"]
        dataset = js["dataset"] if "dataset" in js else None

        n_sentences = len(js["sentences"])
        js["sentence_groups"] = [[word for sentence in
                                  js["sentences"][max(0, i):min(n_sentences, i + 1)] for word in
                                  sentence] for i in range(n_sentences)]
        js["sentence_start_index"] = [
            sum(len(js["sentences"][i - j - 1]) for j in range(min(0, i))) if i > 0 else 0 for i in
            range(n_sentences)]
        js["sentence_end_index"] = [js["sentence_start_index"][i] + len(js["sentences"][i]) for i in range(n_sentences)]
        for sentence_group_nr in range(len(js["sentence_groups"])):
            if len(js["sentence_groups"][sentence_group_nr]) > 300:
                js["sentence_groups"][sentence_group_nr] = js["sentences"][sentence_group_nr]
                js["sentence_start_index"][sentence_group_nr] = 0
                js["sentence_end_index"][sentence_group_nr] = len(js["sentences"][sentence_group_nr])
                if len(js["sentence_groups"][sentence_group_nr]) > 300:
                    import ipdb;

        for field in ["ner", "relations", 'dep']:
            if field not in js:
                js[field] = [[] for _ in range(n_sentences)]

        zipped = zip(js["sentences"], js["ner"], js["relations"], js["sentence_groups"],
                     js["sentence_start_index"], js["sentence_end_index"], js['dep'])

        outputs = []
        # Loop over the sentences.
        for sentence_num, (sentence, ner, relations, groups, start_ix, end_ix, dep) in enumerate(
                zipped):

            ner_dict, relation_dict, dep_children_dict \
                = format_label_fields(sentence, ner, relations, sentence_start, dep, False)
            sentence_start += len(sentence)
            instance = self._dataset_reader.text_to_instance(
                sentence, ner_dict, relation_dict,
                doc_key, dataset, sentence_num, groups, start_ix, end_ix,
                dep_children_dict)
            outputs_one_instance = self.predict_instance(instance)
            outputs.append(outputs_one_instance)

        return outputs
