

import os
import json
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

# The directory that contains "train.txt", "dev.txt" and "test.txt" outputted by Dai's preprocessing code
daiACL2020_preprocess_dir = 'xxx'

# Temporary directory
temp_output_dir = './temp'

# Output directory
output_dir = './temp'

stanfordcorenlp_dir = './stanford-corenlp-full-2018-10-05'

def do_stat(file_path, dataset_type, output=False, output_path=''):

    sentences = []
    sentence = {}
    ct_sentence = 0
    ct_total_entity = 0
    ct_discontinuous_entity = 0

    step = 1
    with open(file_path) as fp:
        for line in fp:
            line = line.strip()
            if step == 1:
                # words
                sentence["tokens"] = line.split(' ')
                sentence["text"] = line
                sentence["doc"] = dataset_type+str(ct_sentence)
                sentence["start"] = -1
                sentence["end"] = -1
            elif step == 2:
                # entity
                sentence["entities"] = []
                if len(line) != 0:
                    dai_entities = line.split('|')
                    for dai_en in dai_entities:
                        offset_type = dai_en.split(' ')
                        entity = {}
                        entity["type"] = offset_type[1]
                        entity["span"] = []
                        offsets = offset_type[0].split(',')
                        assert len(offsets)%2 == 0
                        for i in range(0, len(offsets), 2):
                            entity["span"].append(offsets[i]+","+offsets[i+1])
                        sentence["entities"].append(entity)

                        ct_total_entity += 1
                        if len(entity["span"]) > 1:
                            ct_discontinuous_entity += 1

            else:
                # white line
                sentences.append(sentence)
                sentence = {}
                ct_sentence += 1

            step = (step+1) % 3

    print("sentence number: {}".format(ct_sentence))
    print("mention number: {}".format(ct_total_entity))
    print("discontinuous {}".format(ct_discontinuous_entity))

    if output:
        with open(output_path, 'w') as fp1:
            for sentence in sentences:
                json_sent = json.dumps(sentence)
                fp1.write(json_sent + "\n")

def read_file(file_path):
    filtered_instances = []
    fp = open(file_path, 'r', encoding='utf-8')
    for line in fp:
        line = line.strip()
        if len(line) == 0:
            continue

        instance = json.loads(line)
        filtered_instances.append(instance)

    fp.close()
    return filtered_instances

def is_span_overlapped(start, end, other_start, other_end):
    if other_start < start and other_end >= start:
        return True
    elif other_start <= end and other_end > end:
        return True
    elif other_start >= start and other_end <= end:
        return True
    else:
        return False

def getDepTree1(tokens, edges_list):
    nodes = [[] for token in tokens]
    nodes_dict = {"nodes":nodes}

    for edge in edges_list:
        if edge['dep'] == 'ROOT':
            continue
        if edge['dependent']-1 not in nodes[edge['governor']-1]:
            nodes[edge['governor']-1].append(edge['dependent']-1)
        if edge['governor']-1 not in nodes[edge['dependent']-1]:
            nodes[edge['dependent']-1].append(edge['governor']-1)

    return nodes_dict

def transfer_into_dygie(instances, output_file):

    fp = open(output_file, 'w')
    for idx, instance in enumerate(tqdm(instances)):
        doc = {}
        doc['doc_key'] = instance['doc']+"_"+str(instance['start'])+"_"+str(instance['end'])
        doc['sentences'] = []
        doc['sentences'].append(instance['tokens'])
        doc['ner'] = []
        ner_for_this_sentence = []
        doc['relations'] = []
        relation_for_this_sentence = []
        for entity_idx, entity in enumerate(instance['entities']):
            for span in entity['span']:
                start = int(span.split(',')[0])
                end = int(span.split(',')[1])
                entity_output = [start, end, entity['type']]
                ner_for_this_sentence.append(entity_output)

                # detect whether a span is overlapped with another
                for other_idx, other in enumerate(instance['entities']):
                    if other_idx == entity_idx:
                        continue
                    for other_span in other['span']:
                        other_start = int(other_span.split(',')[0])
                        other_end = int(other_span.split(',')[1])
                        if start == other_start and end == other_end:
                            continue
                        if is_span_overlapped(start, end, other_start, other_end):
                            relation_for_this_sentence.append([start, end, other_start, other_end, "Overlap"])


            n_spans = len(entity['span'])
            candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans) if i!=j] # undirected relations
            for i, j in candidate_indices:
                arg1_start = int(entity['span'][i].split(',')[0])
                arg1_end = int(entity['span'][i].split(',')[1])
                arg2_start = int(entity['span'][j].split(',')[0])
                arg2_end = int(entity['span'][j].split(',')[1])
                relation_for_this_sentence.append([arg1_start, arg1_end, arg2_start, arg2_end, "Combined"])

        doc['ner'].append(ner_for_this_sentence)
        doc['relations'].append(relation_for_this_sentence)
        doc['dep'] = []

        try:
            nlp_res_raw = nlp.annotate(' '.join(instance['tokens']), properties={'annotators': 'tokenize,ssplit,pos,parse'})
            nlp_res = json.loads(nlp_res_raw)
        except Exception as e:
            doc['dep'].append({})
            fp.write(json.dumps(doc) + "\n")
            continue

        if len(nlp_res['sentences']) >= 2:
            doc['dep'].append({})
            fp.write(json.dumps(doc) + "\n")
            continue

        dep_nodes = getDepTree1(nlp_res['sentences'][0]['tokens'], nlp_res['sentences'][0]['enhancedPlusPlusDependencies'])
        doc['dep'].append(dep_nodes)
        fp.write(json.dumps(doc)+"\n")

    fp.close()

if __name__ == "__main__":
    if not os.path.exists(temp_output_dir):
        os.makedirs(temp_output_dir)

    do_stat(os.path.join(daiACL2020_preprocess_dir, 'train.txt'), 'train', True, os.path.join(temp_output_dir, 'train.txt'))
    do_stat(os.path.join(daiACL2020_preprocess_dir, 'dev.txt'), 'dev', True, os.path.join(temp_output_dir, 'dev.txt'))
    do_stat(os.path.join(daiACL2020_preprocess_dir, 'test.txt'), 'test', True, os.path.join(temp_output_dir, 'test.txt'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with StanfordCoreNLP(stanfordcorenlp_dir, memory='8g', timeout=60000) as nlp:

        # for cadec
        train_instances = read_file(os.path.join(temp_output_dir, 'train.txt'))
        dev_instances = read_file(os.path.join(temp_output_dir, 'dev.txt'))
        test_instances = read_file(os.path.join(temp_output_dir, 'test.txt'))

        transfer_into_dygie(train_instances, os.path.join(output_dir, 'train.json'))
        transfer_into_dygie(dev_instances, os.path.join(output_dir, 'dev.json'))
        transfer_into_dygie(test_instances, os.path.join(output_dir, 'test.json'))