import os
import torch
import pickle
import numpy as np
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base')
path = '../wikidata5m_alias'
if not os.path.exists('../wikidata5m_alias_emb'):
    os.makedirs('../wikidata5m_alias_emb')

with open('../read_ent_vocab.bin', 'rb') as fin:
    ent_vocab = pickle.load(fin)
with open('../read_rel_vocab.bin', 'rb') as fin:
    rel_vocab = pickle.load(fin)
print(len(ent_vocab))
print(len(rel_vocab))

aliases = {}
with open(os.path.join(path, 'wikidata5m_entity.txt'), 'r', encoding='utf-8') as fin:
    for line in fin:
        segs = line.strip().split('\t')
        entity = segs[0]
        alias = segs[1:]
        aliases[entity] = alias
print(len(aliases))

miss = 0
entity_embeddings = []
for k, v in ent_vocab.items():
    if k in aliases:
        alias = aliases[k][0]
        tokens = tokenizer.encode(' '+alias, add_special_tokens=False)
        embedding = roberta.embeddings.word_embeddings(torch.tensor(tokens).view(1,-1)).squeeze(0).mean(dim=0)
    else:
        miss += 1
        embedding = torch.randn(768) / 10
    entity_embeddings.append(embedding)

assert len(ent_vocab) == len(entity_embeddings)
entity_embeddings = torch.stack(entity_embeddings, dim=0)
print(miss * 1.0 / len(ent_vocab))
print(entity_embeddings.shape)

np.save('../wikidata5m_alias_emb/entities.npy', entity_embeddings.detach().numpy())
del entity_embeddings

rel_aliases = {}
with open(os.path.join(path, 'wikidata5m_relation.txt'), 'r', encoding='utf-8') as fin:
    for line in fin:
        segs = line.strip().split('\t')
        relation = segs[0]
        alias = segs[1:]
        rel_aliases[relation] = alias

miss = 0
relation_embeddings = []
for k, v in rel_vocab.items():
    if k in rel_aliases:
        alias = rel_aliases[k][0]
        tokens = tokenizer.encode(' '+alias, add_special_tokens=False)
        embedding = roberta.embeddings.word_embeddings(torch.tensor(tokens).view(1,-1)).squeeze(0).mean(dim=0)
    else:
        miss += 1
        embedding = torch.randn(768) / 10
    relation_embeddings.append(embedding)

assert len(rel_vocab) == len(relation_embeddings)
relation_embeddings = torch.stack(relation_embeddings, dim=0)
print(relation_embeddings.shape)
print(miss * 1.0 / len(ent_vocab))
np.save('../wikidata5m_alias_emb/relations.npy', relation_embeddings.detach().numpy())