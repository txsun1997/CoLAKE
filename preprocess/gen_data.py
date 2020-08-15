import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
import random
from transformers import RobertaTokenizer
import os
from tqdm import tqdm
from time import *
from multiprocessing import Pool
import json

input_folder = "../pretrain_data/ann"

file_list = []
for path, _, filenames in os.walk(input_folder):
    for filename in filenames:
        file_list.append(os.path.join(path, filename))
print('# of files', len(file_list))


def load_data():
    wiki5m_alias2qid, wiki5m_qid2alias = {}, {}
    with open("../wikidata5m_alias/wikidata5m_entity.txt", 'r',
              encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            v = line.strip().split("\t")
            if len(v) < 2:
                continue
            qid = v[0]
            for alias in v[1:]:
                wiki5m_qid2alias[qid] = alias
                wiki5m_alias2qid[alias] = qid

    d_ent = wiki5m_alias2qid
    print('wikidata5m_entity.txt (Wikidata5M) loaded!')

    wiki5m_pid2alias = {}
    with open("../wikidata5m_alias/wikidata5m_relation.txt", 'r',
              encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            v = line.strip().split("\t")
            if len(v) < 2:
                continue
            wiki5m_pid2alias[v[0]] = v[1]
    print('wikidata5m_relation.txt (Wikidata5M) loaded!')

    # This is to remove FewRel test set from our training data. If your need is not just reproducing the experiments,
    # you can discard this part. The `ernie_data` is obtained from https://github.com/thunlp/ERNIE
    fewrel_triples = set()
    with open('../ernie_data/fewrel/test.json', 'r', encoding='utf-8') as fin:
        fewrel_data = json.load(fin)
        for ins in fewrel_data:
            r = ins['label']
            h, t = ins['ents'][0][0], ins['ents'][1][0]
            fewrel_triples.add((h, r, t))
    print('# triples in FewRel test set: {}'.format(len(fewrel_triples)))
    print(list(fewrel_triples)[0])
    head_cluster, tail_cluster = {}, {}
    num_del = total = 0
    with open("../wikidata5m_triplet.txt", 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            v = line.strip().split("\t")
            if len(v) != 3:
                continue
            h, r, t = v
            if (h, r, t) not in fewrel_triples:
                if h in head_cluster:
                    head_cluster[h].append((r, t))
                else:
                    head_cluster[h] = [(r, t)]
                if t in tail_cluster:
                    tail_cluster[t].append((r, h))
                else:
                    tail_cluster[t] = [(r, h)]
            else:
                num_del += 1
            total += 1
    print('wikidata5m_triplet.txt (Wikidata5M) loaded!')
    print('deleted {} triples from Wikidata5M.'.format(num_del))

    return d_ent, head_cluster


d_ent, head_cluster = load_data()

# args
max_neighbors = 15
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


def run_proc(index, n, file_list, min_seq_len=80, max_seq_len=200, n_samples_per_file=5000):
    output_folder = '../pretrain_data/data/' + str(index)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    j = index
    large_j = index
    drop_samples = 0
    is_large = False
    n_normal_data, n_large_data = 0, 0
    target_filename = os.path.join(output_folder, str(j))
    large_target_filename = os.path.join(output_folder, 'large_' + str(large_j))
    fout_normal = open(target_filename, 'a+', encoding='utf-8')
    fout_large = open(large_target_filename, 'a+', encoding='utf-8')
    for i in range(len(file_list)):
        if i % n == index:
            # num_words, num_ents = 0, 0
            start_time = time()
            input_name = file_list[i]
            print('[processing] # {}: {}'.format(i, input_name))
            fin = open(input_name, 'r', encoding='utf-8')

            for doc in fin:
                doc = doc.strip()
                segs = doc.split("[_end_]")
                content = segs[0]
                sentences = sent_tokenize(content)
                map_segs = segs[1:]
                maps = {}  # mention -> QID
                for x in map_segs:
                    v = x.split("[_map_]")
                    if len(v) != 2:
                        continue
                    if v[1] in d_ent:  # if a wikipedia title is the alias of an entity in wikidata
                        maps[v[0]] = d_ent[v[1]]
                    elif v[1].lower() in d_ent:
                        maps[v[0]] = d_ent[v[1].lower()]
                blocks, word_lst = [], []
                s = ''
                for sent in sentences:
                    s = '{} {}'.format(s, sent)
                    # s = s + ' ' + sent
                    word_lst = tokenizer.encode(s)
                    if len(word_lst) >= min_seq_len:
                        blocks.append(s)
                        s = ''
                if len(s) > 0:
                    blocks.append(s)
                for block in blocks: 
                    anchor_segs = [x.strip() for x in block.split("sepsepsep")]
                    tokens, entities = [0], []  # [<s>]
                    node2label = {0: 0}  # node:0 -> <s>:0
                    # edges = []  # mention - entity links
                    idx = 1  # idx of word nodes in G
                    pos = 1  # position of current node
                    soft_position = [0]
                    entity_position = []

                    for x in anchor_segs:
                        if len(x) < 1:
                            continue
                        if x in maps and maps[x] not in entities:
                            entities.append(maps[x])
                            entity_position.append(pos)
                            pos += 1
                        else:
                            words = tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True)
                            words = words[:max_seq_len]
                            for word in words:
                                node2label[idx] = word
                                tokens.append(word)
                                soft_position.append(pos)
                                idx += 1
                                pos += 1
                    if len(entities) == 0:
                        continue
                    node2label[idx] = 2  # node:idx -> </s>:2
                    tokens.append(2)  # </s>: 2
                    soft_position.append(pos)
                    idx += 1
                    assert len(tokens) == idx

                    G = nx.complete_graph(idx)
                    for entity, pos in zip(entities, entity_position):
                        if entity not in G.nodes:
                            G.add_node(entity)
                            node2label[entity] = entity
                            soft_position.append(pos)
                    G = nx.complete_graph(G)
                    n_word_nodes = idx
                    token_types = [0] * n_word_nodes + [1] * len(entities)
                    relation_to_add = []
                    for entity, pos in zip(entities, entity_position):
                        if entity in head_cluster and random.uniform(0, 1) > 0.5:
                            triple_lst = head_cluster[entity]
                            random.shuffle(triple_lst)
                            head_neighbors = 0
                            for (r, t) in triple_lst:
                                if head_neighbors >= max_neighbors:
                                    break
                                if t not in G.nodes:
                                    G.add_node(t)
                                    node2label[t] = t
                                    soft_position.append(pos + 2)
                                    token_types.append(1)
                                relation_to_add.append((idx, r, entity, pos + 1, t))
                                head_neighbors += 1
                                idx += 1
                    for idx, r, entity, pos, t in relation_to_add:
                        G.add_node(idx)
                        node2label[idx] = r
                        G.add_edge(entity, idx)
                        soft_position.append(pos)
                        token_types.append(2)
                        G.add_edge(idx, t)
                    # check dimension
                    if len(G.nodes) != len(soft_position):
                        print('[warning] number of nodes does not match length of position ids')
                        continue
                    if len(G.nodes) != len(token_types):
                        print('[warning] number of nodes does not match length of token_types')
                        continue

                    if len(G.nodes) > 256 and len(G.nodes) <= 512:
                        is_large = True
                    elif len(G.nodes) <= 256:
                        is_large = False
                    elif len(G.nodes) > 512:
                        drop_samples += 1
                        # print('\texceed max nodes limitation. drop instance.')
                        continue

                    adj = np.array(nx.adjacency_matrix(G).todense())
                    adj = adj + np.eye(adj.shape[0], dtype=int)
                    if not is_large:
                        normal_ins = {'n_word_nodes': n_word_nodes, 'nodes': [node2label[k] for k in G.nodes],
                                      'soft_position': soft_position, 'adj': adj.tolist(),
                                      'token_type_ids': token_types}
                        fout_normal.write(json.dumps(normal_ins) + '\n')
                        n_normal_data += 1

                    else:
                        large_ins = {'n_word_nodes': n_word_nodes, 'nodes': [node2label[k] for k in G.nodes],
                                     'soft_position': soft_position, 'adj': adj.tolist(), 'token_type_ids': token_types}
                        fout_large.write(json.dumps(large_ins) + '\n')
                        n_large_data += 1

                    if n_normal_data >= n_samples_per_file:
                        n_normal_data = 0
                        fout_normal.close()
                        j += n
                        target_filename = os.path.join(output_folder, str(j))
                        fout_normal = open(target_filename, 'a+', encoding='utf-8')
                    if n_large_data >= n_samples_per_file:
                        n_large_data = 0
                        fout_large.close()
                        large_j += n
                        large_target_filename = os.path.join(output_folder, 'large_' + str(large_j))
                        fout_large = open(large_target_filename, 'a+', encoding='utf-8')

            # avg_pro = num_words * 1.0 / num_ents
            # print('words : entities/relations = {}'.format(avg_pro))
            fin.close()
            end_time = time()
            print('[TIME] {}s'.format(end_time - start_time))
    print('drop {} samples due to max nodes limitation.\n[finished].'.format(drop_samples))
    fout_normal.close()
    fout_large.close()
    print(target_filename)
    print(large_target_filename)


import sys

n = int(sys.argv[1])
p = Pool(n)
for i in range(n):
    p.apply_async(run_proc, args=(i, n, file_list))
p.close()
p.join()
