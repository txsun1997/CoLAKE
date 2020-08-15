import os
import torch
import json
from torch.utils.data import Dataset
from fastNLP import seq_len_to_mask
from transformers import RobertaTokenizer

WORD_PADDING_INDEX = 1
ENTITY_PADDING_INDEX = 1


class REGraphDataSet(Dataset):
    def __init__(self, path, set_type, label_vocab, ent_vocab):
        self.set_type = set_type + '.json'
        self.label_vocab = label_vocab
        self.data = []
        with open(os.path.join(path, self.set_type), 'r', encoding='utf-8') as fin:
            raw_data = json.load(fin)
        for ins in raw_data:
            nodes_index = []
            n_words = 0
            for node in ins['nodes']:
                if isinstance(node, str):
                    nodes_index.append(ent_vocab[node])
                else:
                    nodes_index.append(node)
                    n_words += 1

            self.data.append({
                'input_ids': nodes_index,
                'n_word_nodes': n_words,
                'n_entity_nodes': len(nodes_index) - n_words,
                'position_ids': ins['soft_position'],
                'token_type_ids': ins['token_type_ids'],
                'target': label_vocab[ins['label']]
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        input_keys = ['input_ids', 'n_word_nodes', 'n_entity_nodes', 'position_ids', 'attention_mask', 'target',
                      'token_type_ids']
        target_keys = ['target']
        max_words = max_ents = 0
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in target_keys}
        for sample in batch:
            max_words = sample['n_word_nodes'] if sample['n_word_nodes'] > max_words else max_words
            max_ents = sample['n_entity_nodes'] if sample['n_entity_nodes'] > max_ents else max_ents

        for sample in batch:
            word_pad = max_words - sample['n_word_nodes']
            ent_pad = max_ents - sample['n_entity_nodes']
            n_words = sample['n_word_nodes']
            batch_x['input_ids'].append(sample['input_ids'][:n_words] + [WORD_PADDING_INDEX] * word_pad + \
                                        sample['input_ids'][n_words:] + [ENTITY_PADDING_INDEX] * ent_pad)

            batch_x['position_ids'].append(sample['position_ids'][:n_words] + [0] * word_pad + \
                                           sample['position_ids'][n_words:] + [0] * ent_pad)

            batch_x['token_type_ids'].append(sample['token_type_ids'][:n_words] + [0] * word_pad + \
                                             sample['token_type_ids'][n_words:] + [0] * ent_pad)

            batch_x['n_word_nodes'].append(max_words)
            batch_x['n_entity_nodes'].append(max_ents)

            adj = torch.ones(len(sample['input_ids']), len(sample['input_ids']), dtype=torch.int)
            adj = torch.cat((adj[:n_words, :],
                             torch.ones(word_pad, adj.shape[1], dtype=torch.int),
                             adj[n_words:, :],
                             torch.ones(ent_pad, adj.shape[1], dtype=torch.int)), dim=0)

            adj = torch.cat((adj[:, :n_words],
                             torch.zeros(max_words + max_ents, word_pad, dtype=torch.int),
                             adj[:, n_words:],
                             torch.zeros(max_words + max_ents, ent_pad, dtype=torch.int)), dim=1)

            batch_x['attention_mask'].append(adj)
            batch_x['target'].append(sample['target'])
            batch_y['target'].append(sample['target'])

        for k, v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            else:
                batch_x[k] = torch.tensor(v)
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(v)

        return (batch_x, batch_y)


class TypingGraphDataSet(Dataset):
    def __init__(self, path, set_type, label_vocab, ent_vocab):
        self.set_type = set_type + '.json'
        self.label_vocab = label_vocab
        self.data = []
        with open(os.path.join(path, self.set_type), 'r', encoding='utf-8') as fin:
            raw_data = json.load(fin)
        for ins in raw_data:
            nodes_index = []
            n_words = 0
            for node in ins['nodes']:
                if isinstance(node, str):
                    nodes_index.append(ent_vocab[node])
                else:
                    nodes_index.append(node)
                    n_words += 1

            label_index = [self.label_vocab[x] for x in ins['labels']]
            label_vec = [0] * len(self.label_vocab)
            for golden_id in label_index:
                label_vec[golden_id] = 1

            self.data.append({
                'input_ids': nodes_index,
                'n_word_nodes': n_words,
                'n_entity_nodes': len(nodes_index) - n_words,
                'position_ids': ins['soft_position'],
                'token_type_ids': ins['token_type_ids'],
                'target': label_vec
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        input_keys = ['input_ids', 'n_word_nodes', 'n_entity_nodes', 'position_ids', 'attention_mask', 'target',
                      'token_type_ids']
        target_keys = ['target']
        max_words = max_ents = 0
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in target_keys}
        for sample in batch:
            max_words = sample['n_word_nodes'] if sample['n_word_nodes'] > max_words else max_words
            max_ents = sample['n_entity_nodes'] if sample['n_entity_nodes'] > max_ents else max_ents

        for sample in batch:
            word_pad = max_words - sample['n_word_nodes']
            ent_pad = max_ents - sample['n_entity_nodes']
            n_words = sample['n_word_nodes']
            batch_x['input_ids'].append(sample['input_ids'][:n_words] + [WORD_PADDING_INDEX] * word_pad + \
                                        sample['input_ids'][n_words:] + [ENTITY_PADDING_INDEX] * ent_pad)

            batch_x['position_ids'].append(sample['position_ids'][:n_words] + [0] * word_pad + \
                                           sample['position_ids'][n_words:] + [0] * ent_pad)

            batch_x['token_type_ids'].append(sample['token_type_ids'][:n_words] + [0] * word_pad + \
                                             sample['token_type_ids'][n_words:] + [0] * ent_pad)

            batch_x['n_word_nodes'].append(max_words)
            batch_x['n_entity_nodes'].append(max_ents)

            adj = torch.ones(len(sample['input_ids']), len(sample['input_ids']), dtype=torch.int)
            adj = torch.cat((adj[:n_words, :],
                             torch.ones(word_pad, adj.shape[1], dtype=torch.int),
                             adj[n_words:, :],
                             torch.ones(ent_pad, adj.shape[1], dtype=torch.int)), dim=0)

            adj = torch.cat((adj[:, :n_words],
                             torch.zeros(max_words + max_ents, word_pad, dtype=torch.int),
                             adj[:, n_words:],
                             torch.zeros(max_words + max_ents, ent_pad, dtype=torch.int)), dim=1)

            batch_x['attention_mask'].append(adj)
            batch_x['target'].append(sample['target'])
            batch_y['target'].append(sample['target'])

        for k, v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            elif k == 'target':
                batch_x[k] = torch.FloatTensor(v)
            else:
                batch_x[k] = torch.tensor(v)
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(v)

        return (batch_x, batch_y)