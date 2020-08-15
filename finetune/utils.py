import os
import json


def build_label_vocab(data_dir, task_type='re'):
    label_vocab = {}
    with open(os.path.join(data_dir, 'train.json'), 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    if task_type == 're':
        for ins in data:
            label = ins['label']
            if label not in label_vocab:
                label_vocab[label] = len(label_vocab)
    elif task_type == 'typing':
        for ins in data:
            labels = ins['labels']
            for label in labels:
                if label not in label_vocab:
                    label_vocab[label] = len(label_vocab)
    else:
        raise RuntimeError('wrong task_type')
    print('# of labels: {}'.format(len(label_vocab)))
    return label_vocab


def build_temp_ent_vocab(path):
    ent_vocab = {'<unk>': 0, '<pad>': 1, '<mask>': 2}
    files = ['train.json', 'dev.json', 'test.json']
    for file in files:
        with open(os.path.join(path, file), 'r', encoding='utf-8') as fin:
            data = json.load(fin)
        for ins in data:
            for node in ins['nodes']:
                if isinstance(node, str) and node.startswith('Q'):
                    if node not in ent_vocab:
                        ent_vocab[node] = len(ent_vocab)
    print('# of entities occurred in train/dev/test files: {}'.format(len(ent_vocab)))
    return ent_vocab