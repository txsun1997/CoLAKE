import os
import json


def load_file(filename):
    data = []
    with open(filename, "r", encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def parse_template(template, subject_label, object_label):
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return [template]

def load_vocab(vocab_filename):
    with open(vocab_filename, "r", encoding='utf-8') as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab