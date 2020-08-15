import os
import sys
import json
import pickle
import torch
import random
import numpy as np
from torch.utils.data import Dataset, Sampler
from pretrain.utils import create_mlm_labels

WORD_PADDING_INDEX = 1
ENTITY_PADDING_INDEX = 1
RELATION_PADDING_INDEX = 1


class GraphOTFDataSet(Dataset):
    def __init__(self, indexed_train_fps, n_workers, rank, word_mask_index, word_vocab_size, k_negative_samples,
                 ent_vocab, rel_vocab, ent_freq):
        # index and mask

        self.input_ids, self.n_word_nodes, self.n_entity_nodes, self.position_ids, self.attention_mask, self.masked_lm_labels, \
        self.ent_masked_lm_labels, self.rel_masked_lm_labels, self.token_type_ids = [], [], [], [], [], [], [], [], []
        self.word_mask_index = word_mask_index
        self.word_vocab_size = word_vocab_size
        self.rank = rank
        self.ent_vocab = ent_vocab
        self.rel_vocab = rel_vocab
        self.ent_freq = ent_freq

        self.k_negative_samples = k_negative_samples
        self.nega_samp_weight = self._nagative_sampling_weight()

        file_per_process = len(indexed_train_fps) // n_workers
        if file_per_process * n_workers != len(indexed_train_fps):
            if rank == 0:
                print('Drop {} files.'.format(len(indexed_train_fps) - file_per_process * n_workers))
                print('# files per process: {}'.format(file_per_process))
        self.fps = indexed_train_fps[rank * file_per_process:(rank + 1) * file_per_process]

        self.current_file_idx = 0
        self.data = self.read_file(self.current_file_idx)
        self.num_samples_per_file = len(self.data)
        self.total_num_samples = self.num_samples_per_file * len(self.fps)

    def read_file(self, idx):
        data = []
        with open(self.fps[idx], 'r', encoding='utf-8') as fin:
            for x in fin:
                instance = json.loads(x)
                # n_word_nodes = instance['n_word_nodes']
                words, entities, relations = self._split_nodes(instance['nodes'], instance['token_type_ids'])
                anchor_entities = self._find_anchor_entities(instance['adj'], len(words), entities)
                # entities = self._replace_anchor_entities(anchor_entities, entities)
                entities = [self.ent_vocab[ent] for ent in entities]
                anchor_entities = [self.ent_vocab[ent] for ent in anchor_entities]
                relations = [self.rel_vocab[rel] for rel in relations]
                words, word_mlm_labels = create_mlm_labels(words, self.word_mask_index, self.word_vocab_size)
                entities, entity_mlm_labels = create_mlm_labels(entities, self.ent_vocab['<mask>'], len(self.ent_vocab),
                                                                anchor_nodes=anchor_entities)
                relations, relation_mlm_labels = create_mlm_labels(relations, self.rel_vocab['<mask>'],
                                                                   len(self.rel_vocab))
                assert len(instance['nodes']) == len(words + entities + relations)
                assert len(instance['nodes']) == len(instance['soft_position'])
                assert len(instance['nodes']) == len(instance['adj'])
                assert len(instance['nodes']) == len(instance['token_type_ids'])
                data.append({
                    'input_ids': words + entities + relations,
                    'n_word_nodes': len(words),
                    'n_entity_nodes': len(entities),
                    'position_ids': instance['soft_position'],
                    'attention_mask': instance['adj'],
                    'token_type_ids': instance['token_type_ids'],
                    'masked_lm_labels': word_mlm_labels,
                    'ent_masked_lm_labels': entity_mlm_labels,
                    'rel_masked_lm_labels': relation_mlm_labels
                })
        # data is a list of dict
        return data

    def __getitem__(self, item):
        file_idx = item // self.num_samples_per_file
        if file_idx != self.current_file_idx:
            self.data = self.read_file(file_idx)
            self.current_file_idx = file_idx
        sample = self.data[item - file_idx * self.num_samples_per_file]
        # label = {
        #     'masked_lm_labels': sample['masked_lm_labels'],
        #     'ent_masked_lm_labels': sample['ent_masked_lm_labels'],
        #     'rel_masked_lm_labels': sample['rel_masked_lm_labels']
        # }
        return sample

    def __len__(self):
        return self.total_num_samples

    def _split_nodes(self, nodes, types):
        assert len(nodes) == len(types)
        words, entities, relations = [], [], []
        for node, type in zip(nodes, types):
            if type == 0:
                words.append(node)
            elif type == 1:
                entities.append(node)
            elif type == 2:
                relations.append(node)
            else:
                raise ValueError('unknown token type id.')
        return words, entities, relations

    def _find_anchor_entities(self, adj, n_word_nodes, entities):
        anchor_entities = []
        for i in adj[:n_word_nodes]:
            ents = i[n_word_nodes:n_word_nodes + len(entities)]
            for j, mask in enumerate(ents):
                if mask == 1 and entities[j] not in anchor_entities:
                    anchor_entities.append(entities[j])
        # if have relations
        # if len(adj) > n_word_nodes + len(entities):
        #     for idx, attn_mask in enumerate(adj[n_word_nodes:n_word_nodes + len(entities)]):
        #         if entities[idx] in anchor_entities:
        #             i = idx + n_word_nodes
        #             for j in range(n_word_nodes + len(entities), len(adj)):
        #                 adj[i][j] = 0

        return anchor_entities

    def _replace_anchor_entities(self, anchor_entities, entities):
        replaced_ents = []
        for entity in entities:
            if entity in anchor_entities:
                x = random.uniform(0, 1)
                if x < 0.3:
                    replaced_ents.append('<unk>')
                else:
                    replaced_ents.append(entity)
            else:
                replaced_ents.append(entity)
        return replaced_ents

    def _nagative_sampling_weight(self, pwr=0.75):
        ef = []
        for i, ent in enumerate(self.ent_vocab.keys()):
            assert self.ent_vocab[ent] == i
            ef.append(self.ent_freq[ent])
        # freq = np.array([self.ent_freq[ent] for ent in self.ent_vocab.keys()])
        ef = np.array(ef)
        ef = ef / ef.sum()
        ef = np.power(ef, pwr)
        ef = ef / ef.sum()
        return torch.FloatTensor(ef)

    def collate_fn(self, batch):
        # batch: [[x1:dict, y1:dict], [x2:dict, y2:dict], ...]
        input_keys = ['input_ids', 'n_word_nodes', 'n_entity_nodes', 'position_ids', 'attention_mask', 'ent_index',
                      'masked_lm_labels', 'ent_masked_lm_labels', 'rel_masked_lm_labels', 'token_type_ids']
        target_keys = ['masked_lm_labels', 'ent_masked_lm_labels', 'rel_masked_lm_labels',
                       'word_seq_len', 'ent_seq_len', 'rel_seq_len']
        max_word_nodes, max_entity_nodes, max_relation_nodes = 0, 0, 0
        batch_word, batch_entity, batch_relation = [], [], []
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in target_keys}

        ent_convert_dict = {}
        ent_index = []
        # convert list of dict into dict of list
        for sample in batch:
            for n, v in sample.items():
                if n in input_keys:
                    batch_x[n].append(v)
                if n in target_keys:
                    batch_y[n].append(v)
            # for n, v in x.items():
            #     batch_x[n].append(v)
            # for n, v in y.items():
            #     batch_y[n].append(v)
            batch_x['ent_index'].append([])
            n_word_nodes = sample['n_word_nodes']
            n_entity_nodes = sample['n_entity_nodes']
            words = sample['input_ids'][0:n_word_nodes]
            entities = sample['input_ids'][n_word_nodes:n_word_nodes + n_entity_nodes]
            relations = sample['input_ids'][n_word_nodes + n_entity_nodes:]
            batch_word.append(words)
            batch_entity.append(entities)
            batch_relation.append(relations)

            batch_y['word_seq_len'].append(n_word_nodes)
            batch_y['ent_seq_len'].append(n_entity_nodes)
            batch_y['rel_seq_len'].append(len(relations))

            max_word_nodes = len(words) if len(words) > max_word_nodes else max_word_nodes
            max_entity_nodes = len(entities) if len(entities) > max_entity_nodes else max_entity_nodes
            max_relation_nodes = len(relations) if len(relations) > max_relation_nodes else max_relation_nodes

            for golden_ent in sample['ent_masked_lm_labels']:
                if golden_ent >= 0 and golden_ent not in ent_convert_dict:
                    ent_convert_dict[golden_ent] = len(ent_convert_dict)
                    ent_index.append(golden_ent)

        # # check convert
        # for i in range(len(ent_index)):
        #     assert ent_convert_dict[ent_index[i]] == i

        if len(ent_index) > 0:
            # negative sampling
            k_negas = self.k_negative_samples * len(ent_index)
            nega_samples = torch.multinomial(self.nega_samp_weight, num_samples=k_negas, replacement=True)
            for nega_ent in nega_samples:
                ent = int(nega_ent)
                if ent not in ent_convert_dict:  # 保证无重复
                    ent_convert_dict[ent] = len(ent_convert_dict)
                    ent_index.append(ent)
        else:
            ent_index = [ENTITY_PADDING_INDEX]

        # pad
        seq_len = max_word_nodes + max(max_entity_nodes, 1) + max(max_relation_nodes, 1)
        for i in range(len(batch_word)):
            word_pad = max_word_nodes - len(batch_word[i])
            entity_pad = max_entity_nodes - len(batch_entity[i]) if max_entity_nodes > 0 else 1
            relation_pad = max_relation_nodes - len(batch_relation[i]) if max_relation_nodes > 0 else 1
            batch_x['input_ids'][i] = batch_word[i] + [WORD_PADDING_INDEX] * word_pad + \
                                      batch_entity[i] + [ENTITY_PADDING_INDEX] * entity_pad + \
                                      batch_relation[i] + [RELATION_PADDING_INDEX] * relation_pad

            n_words = batch_x['n_word_nodes'][i]
            n_entities = batch_x['n_entity_nodes'][i]
            batch_x['position_ids'][i] = batch_x['position_ids'][i][:n_words] + [0] * word_pad + \
                                         batch_x['position_ids'][i][n_words:n_words + n_entities] + [0] * entity_pad + \
                                         batch_x['position_ids'][i][n_words + n_entities:] + [0] * relation_pad

            batch_x['token_type_ids'][i] = batch_x['token_type_ids'][i][:n_words] + [0] * word_pad + \
                                           batch_x['token_type_ids'][i][n_words:n_words + n_entities] + [
                                               0] * entity_pad + \
                                           batch_x['token_type_ids'][i][n_words + n_entities:] + [0] * relation_pad

            adj = torch.tensor(batch_x['attention_mask'][i], dtype=torch.int)
            adj = torch.cat((adj[:n_words, :],
                             torch.ones(word_pad, adj.shape[1], dtype=torch.int),
                             adj[n_words:n_words + n_entities, :],
                             torch.ones(entity_pad, adj.shape[1], dtype=torch.int),
                             adj[n_words + n_entities:, :],
                             torch.ones(relation_pad, adj.shape[1], dtype=torch.int)), dim=0)
            assert adj.shape[0] == seq_len
            adj = torch.cat((adj[:, :n_words],
                             torch.zeros(seq_len, word_pad, dtype=torch.int),
                             adj[:, n_words:n_words + n_entities],
                             torch.zeros(seq_len, entity_pad, dtype=torch.int),
                             adj[:, n_words + n_entities:],
                             torch.zeros(seq_len, relation_pad, dtype=torch.int)), dim=1)

            batch_x['attention_mask'][i] = adj
            batch_x['masked_lm_labels'][i] = batch_x['masked_lm_labels'][i] + [-1] * word_pad
            batch_y['masked_lm_labels'][i] = batch_y['masked_lm_labels'][i] + [-1] * word_pad

            batch_x['ent_masked_lm_labels'][i] = [ent_convert_dict[lb] if lb in ent_convert_dict else -1 for lb in
                                                  batch_x['ent_masked_lm_labels'][i]] + [-1] * entity_pad
            batch_y['ent_masked_lm_labels'][i] = batch_x['ent_masked_lm_labels'][i]

            batch_x['rel_masked_lm_labels'][i] = batch_x['rel_masked_lm_labels'][i] + [-1] * relation_pad
            batch_y['rel_masked_lm_labels'][i] = batch_x['rel_masked_lm_labels'][i]

            batch_x['n_word_nodes'][i] = max(max_word_nodes, 1)
            batch_x['n_entity_nodes'][i] = max(max_entity_nodes, 1)
            batch_x['ent_index'][i] = ent_index

        for k, v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            else:
                batch_x[k] = torch.tensor(v)
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(v)
        return (batch_x, batch_y)


class GraphDataSet(Dataset):
    def __init__(self, data_dir, word_mask_index, word_vocab_size, k_negative_samples,
                 ent_vocab, rel_vocab, ent_freq):
        # index and mask
        self.input_ids, self.n_word_nodes, self.n_entity_nodes, self.position_ids, self.attention_mask, self.masked_lm_labels, \
        self.ent_masked_lm_labels, self.rel_masked_lm_labels, self.token_type_ids = [], [], [], [], [], [], [], [], []
        self.word_mask_index = word_mask_index
        self.word_vocab_size = word_vocab_size
        self.ent_vocab = ent_vocab
        self.rel_vocab = rel_vocab
        self.ent_freq = ent_freq

        self.k_negative_samples = k_negative_samples
        self.nega_samp_weight = self._nagative_sampling_weight()

        self.data = self.read_file(data_dir)

    def read_file(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as fin:
            for x in fin:
                instance = json.loads(x)
                # n_word_nodes = instance['n_word_nodes']
                words, entities, relations = self._split_nodes(instance['nodes'], instance['token_type_ids'])
                anchor_entities = self._find_anchor_entities(instance['adj'], len(words), entities)
                # entities = self._replace_anchor_entities(anchor_entities, entities)
                entities = [self.ent_vocab[ent] for ent in entities]
                anchor_entities = [self.ent_vocab[ent] for ent in anchor_entities]
                relations = [self.rel_vocab[rel] for rel in relations]
                words, word_mlm_labels = create_mlm_labels(words, self.word_mask_index, self.word_vocab_size)
                entities, entity_mlm_labels = create_mlm_labels(entities, self.ent_vocab['<mask>'], len(self.ent_vocab),
                                                                anchor_nodes=anchor_entities)
                relations, relation_mlm_labels = create_mlm_labels(relations, self.rel_vocab['<mask>'],
                                                                   len(self.rel_vocab))
                assert len(instance['nodes']) == len(words + entities + relations)
                assert len(instance['nodes']) == len(instance['soft_position'])
                assert len(instance['nodes']) == len(instance['adj'])
                assert len(instance['nodes']) == len(instance['token_type_ids'])
                data.append({
                    'input_ids': words + entities + relations,
                    'n_word_nodes': len(words),
                    'n_entity_nodes': len(entities),
                    'position_ids': instance['soft_position'],
                    'attention_mask': instance['adj'],
                    'token_type_ids': instance['token_type_ids'],
                    'masked_lm_labels': word_mlm_labels,
                    'ent_masked_lm_labels': entity_mlm_labels,
                    'rel_masked_lm_labels': relation_mlm_labels
                })
        # data is a list of dict
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def _split_nodes(self, nodes, types):
        assert len(nodes) == len(types)
        words, entities, relations = [], [], []
        for node, type in zip(nodes, types):
            if type == 0:
                words.append(node)
            elif type == 1:
                entities.append(node)
            elif type == 2:
                relations.append(node)
            else:
                raise ValueError('unknown token type id.')
        return words, entities, relations

    def _nagative_sampling_weight(self, pwr=0.75):
        ef = []
        for i, ent in enumerate(self.ent_vocab.keys()):
            assert self.ent_vocab[ent] == i
            ef.append(self.ent_freq[ent])
        # freq = np.array([self.ent_freq[ent] for ent in self.ent_vocab.keys()])
        ef = np.array(ef)
        ef = ef / ef.sum()
        ef = np.power(ef, pwr)
        ef = ef / ef.sum()
        return torch.FloatTensor(ef)

    def _find_anchor_entities(self, adj, n_word_nodes, entities):
        anchor_entities = []
        for i in adj[:n_word_nodes]:
            ents = i[n_word_nodes:n_word_nodes + len(entities)]
            for j, mask in enumerate(ents):
                if mask == 1 and entities[j] not in anchor_entities:
                    anchor_entities.append(entities[j])

        return anchor_entities

    def collate_fn(self, batch):
        # batch: [[x1:dict, y1:dict], [x2:dict, y2:dict], ...]
        input_keys = ['input_ids', 'n_word_nodes', 'n_entity_nodes', 'position_ids', 'attention_mask', 'ent_index',
                      'masked_lm_labels', 'ent_masked_lm_labels', 'rel_masked_lm_labels', 'token_type_ids']
        target_keys = ['masked_lm_labels', 'ent_masked_lm_labels', 'rel_masked_lm_labels',
                       'word_seq_len', 'ent_seq_len', 'rel_seq_len']
        max_word_nodes, max_entity_nodes, max_relation_nodes = 0, 0, 0
        batch_word, batch_entity, batch_relation = [], [], []
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in target_keys}

        ent_convert_dict = {}
        ent_index = []
        # convert list of dict into dict of list
        for sample in batch:
            for n, v in sample.items():
                if n in input_keys:
                    batch_x[n].append(v)
                if n in target_keys:
                    batch_y[n].append(v)
            # for n, v in x.items():
            #     batch_x[n].append(v)
            # for n, v in y.items():
            #     batch_y[n].append(v)
            batch_x['ent_index'].append([])
            n_word_nodes = sample['n_word_nodes']
            n_entity_nodes = sample['n_entity_nodes']
            words = sample['input_ids'][0:n_word_nodes]
            entities = sample['input_ids'][n_word_nodes:n_word_nodes + n_entity_nodes]
            relations = sample['input_ids'][n_word_nodes + n_entity_nodes:]
            batch_word.append(words)
            batch_entity.append(entities)
            batch_relation.append(relations)

            batch_y['word_seq_len'].append(n_word_nodes)
            batch_y['ent_seq_len'].append(n_entity_nodes)
            batch_y['rel_seq_len'].append(len(relations))

            max_word_nodes = len(words) if len(words) > max_word_nodes else max_word_nodes
            max_entity_nodes = len(entities) if len(entities) > max_entity_nodes else max_entity_nodes
            max_relation_nodes = len(relations) if len(relations) > max_relation_nodes else max_relation_nodes

            for golden_ent in sample['ent_masked_lm_labels']:
                if golden_ent >= 0 and golden_ent not in ent_convert_dict:
                    ent_convert_dict[golden_ent] = len(ent_convert_dict)
                    ent_index.append(golden_ent)

        # # check convert
        # for i in range(len(ent_index)):
        #     assert ent_convert_dict[ent_index[i]] == i

        if len(ent_index) > 0:
            # negative sampling
            k_negas = self.k_negative_samples * len(ent_index)
            nega_samples = torch.multinomial(self.nega_samp_weight, num_samples=k_negas, replacement=True)
            for nega_ent in nega_samples:
                ent = int(nega_ent)
                if ent not in ent_convert_dict:  # 保证无重复
                    ent_convert_dict[ent] = len(ent_convert_dict)
                    ent_index.append(ent)
        else:
            ent_index = [ENTITY_PADDING_INDEX]

        # pad
        seq_len = max_word_nodes + max(max_entity_nodes, 1) + max(max_relation_nodes, 1)
        for i in range(len(batch_word)):
            word_pad = max_word_nodes - len(batch_word[i])
            entity_pad = max_entity_nodes - len(batch_entity[i]) if max_entity_nodes > 0 else 1
            relation_pad = max_relation_nodes - len(batch_relation[i]) if max_relation_nodes > 0 else 1
            batch_x['input_ids'][i] = batch_word[i] + [WORD_PADDING_INDEX] * word_pad + \
                                      batch_entity[i] + [ENTITY_PADDING_INDEX] * entity_pad + \
                                      batch_relation[i] + [RELATION_PADDING_INDEX] * relation_pad

            n_words = batch_x['n_word_nodes'][i]
            n_entities = batch_x['n_entity_nodes'][i]
            batch_x['position_ids'][i] = batch_x['position_ids'][i][:n_words] + [0] * word_pad + \
                                         batch_x['position_ids'][i][n_words:n_words + n_entities] + [0] * entity_pad + \
                                         batch_x['position_ids'][i][n_words + n_entities:] + [0] * relation_pad

            batch_x['token_type_ids'][i] = batch_x['token_type_ids'][i][:n_words] + [0] * word_pad + \
                                           batch_x['token_type_ids'][i][n_words:n_words + n_entities] + [
                                               0] * entity_pad + \
                                           batch_x['token_type_ids'][i][n_words + n_entities:] + [0] * relation_pad

            adj = torch.tensor(batch_x['attention_mask'][i], dtype=torch.int)
            adj = torch.cat((adj[:n_words, :],
                             torch.ones(word_pad, adj.shape[1], dtype=torch.int),
                             adj[n_words:n_words + n_entities, :],
                             torch.ones(entity_pad, adj.shape[1], dtype=torch.int),
                             adj[n_words + n_entities:, :],
                             torch.ones(relation_pad, adj.shape[1], dtype=torch.int)), dim=0)
            assert adj.shape[0] == seq_len
            adj = torch.cat((adj[:, :n_words],
                             torch.zeros(seq_len, word_pad, dtype=torch.int),
                             adj[:, n_words:n_words + n_entities],
                             torch.zeros(seq_len, entity_pad, dtype=torch.int),
                             adj[:, n_words + n_entities:],
                             torch.zeros(seq_len, relation_pad, dtype=torch.int)), dim=1)

            batch_x['attention_mask'][i] = adj
            batch_x['masked_lm_labels'][i] = batch_x['masked_lm_labels'][i] + [-1] * word_pad
            batch_y['masked_lm_labels'][i] = batch_y['masked_lm_labels'][i] + [-1] * word_pad

            batch_x['ent_masked_lm_labels'][i] = [ent_convert_dict[lb] if lb in ent_convert_dict else -1 for lb in
                                                  batch_x['ent_masked_lm_labels'][i]] + [-1] * entity_pad
            batch_y['ent_masked_lm_labels'][i] = batch_x['ent_masked_lm_labels'][i]

            batch_x['rel_masked_lm_labels'][i] = batch_x['rel_masked_lm_labels'][i] + [-1] * relation_pad
            batch_y['rel_masked_lm_labels'][i] = batch_x['rel_masked_lm_labels'][i]

            batch_x['n_word_nodes'][i] = max(max_word_nodes, 1)
            batch_x['n_entity_nodes'][i] = max(max_entity_nodes, 1)
            batch_x['ent_index'][i] = ent_index

        for k, v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            else:
                batch_x[k] = torch.tensor(v)
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(v)
        return (batch_x, batch_y)


class FewRelDevDataSet(Dataset):
    def __init__(self, path, label_vocab, ent_vocab):
        self.label_vocab = label_vocab
        self.ent_vocab = ent_vocab
        self.data = []
        with open(path, 'r', encoding='utf-8') as fin:
            raw_data = json.load(fin)
        for ins in raw_data:
            nodes_index = []
            for node in ins['nodes']:
                if isinstance(node, str):
                    nodes_index.append(ent_vocab[node])
                else:
                    nodes_index.append(node)

            self.data.append({
                'input_ids': nodes_index,
                # 'n_word_nodes': n_words,
                # 'n_entity_nodes': 2,
                'position_ids': ins['soft_position'],
                'attention_mask': ins['adj'],
                # 'token_type_ids': [0]*n_words + [1]*2 + [2],
                'target': label_vocab[ins['label']]
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        input_keys = ['input_ids', 'n_word_nodes', 'n_entity_nodes', 'position_ids', 'attention_mask', 'ent_index',
                      'masked_lm_labels', 'ent_masked_lm_labels', 'rel_masked_lm_labels', 'token_type_ids']
        target_keys = ['masked_lm_labels', 'ent_masked_lm_labels', 'rel_masked_lm_labels',
                       'word_seq_len', 'ent_seq_len', 'rel_seq_len']
        max_nodes = 0
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in target_keys}

        ent_index = [ENTITY_PADDING_INDEX, 3]
        for sample in batch:
            max_nodes = len(sample['input_ids']) if len(sample['input_ids']) > max_nodes else max_nodes

        for sample in batch:
            word_pad = max_nodes - len(sample['input_ids'])
            n_words = len(sample['input_ids']) - 3
            batch_y['word_seq_len'].append(n_words)
            batch_y['ent_seq_len'].append(2)
            batch_y['rel_seq_len'].append(1)

            batch_x['input_ids'].append(sample['input_ids'][:-3] + [WORD_PADDING_INDEX] * word_pad + sample['input_ids'][-3:])
            batch_x['n_word_nodes'].append(max_nodes - 3)
            batch_x['n_entity_nodes'].append(2)
            batch_x['position_ids'].append(sample['position_ids'][:-3] + [0] * word_pad + sample['position_ids'][-3:])
            adj = torch.tensor(sample['attention_mask'], dtype=torch.int)
            adj = torch.cat((adj[:-3, :],
                             torch.ones(word_pad, adj.shape[1], dtype=torch.int),
                             adj[-3:, :]), dim=0)

            adj = torch.cat((adj[:, :-3],
                             torch.zeros(max_nodes, word_pad, dtype=torch.int),
                             adj[:, -3:]), dim=1)

            batch_x['attention_mask'].append(adj)
            batch_x['token_type_ids'].append([0] * (max_nodes - 3) + [1, 1, 2])
            batch_x['ent_index'].append(ent_index)
            batch_x['masked_lm_labels'].append([-1] * (max_nodes - 3))
            batch_x['ent_masked_lm_labels'].append([-1, -1])
            batch_x['rel_masked_lm_labels'].append([sample['target']])

            batch_y['masked_lm_labels'].append([-1] * (max_nodes - 3))
            batch_y['ent_masked_lm_labels'].append([-1, -1])
            batch_y['rel_masked_lm_labels'].append([sample['target']])

        for k, v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            else:
                batch_x[k] = torch.tensor(v)
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(v)

        return (batch_x, batch_y)
