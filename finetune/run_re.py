import os
import sys

import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from transformers import RobertaConfig, RobertaTokenizer

import fitlog
from fastNLP import cache_results
from fastNLP import FitlogCallback, WarmupCallback, GradientClipCallback
from fastNLP import RandomSampler, TorchLoaderIter, LossInForward, Trainer, Tester

sys.path.append('../')
from finetune.dataloader import REGraphDataSet
from finetune.model import CoLAKEForRE
from finetune.metrics import MacroMetric
from finetune.utils import build_label_vocab, build_temp_ent_vocab
from pretrain.utils import load_ent_rel_vocabs


@cache_results(_cache_fp='fewrel_CoLAKE.bin', _refresh=False)
def load_fewrel_graph_data(data_dir):
    datasets = ['train', 'dev', 'test']
    label_vocab = build_label_vocab(data_dir)
    ent_vocab = build_temp_ent_vocab(data_dir)
    result = []
    for set_type in datasets:
        print('processing {} set...'.format(set_type))
        dataset = REGraphDataSet(data_dir, set_type=set_type, label_vocab=label_vocab, ent_vocab=ent_vocab)
        result.append(dataset)
    result.append(ent_vocab)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/fewrel',
                        help="data directory path")
    parser.add_argument('--log_dir', type=str, default='./logs/',
                        help="fitlog directory path")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--beta', type=float, default=0.999, help="beta_2 of adam")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay")
    parser.add_argument('--warm_up', type=float, default=0.1, help="warmup proportion or steps")
    parser.add_argument('--epoch', type=int, default=3, help="number of epochs")
    parser.add_argument('--grad_accumulation', type=int, default=1, help="gradient accumulation")
    parser.add_argument('--gpu', type=str, default='all', help="run script on which devices")
    parser.add_argument('--debug', action='store_true', help="do not log")
    parser.add_argument('--model_path', type=str, default="../model/",
                        help="the path of directory containing model and entity embeddings.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        fitlog.debug()

    fitlog.set_log_dir(args.log_dir)
    fitlog.commit(__file__)
    fitlog.add_hyper_in_file(__file__)
    fitlog.add_hyper(args)
    if args.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_set, dev_set, test_set, temp_ent_vocab = load_fewrel_graph_data(data_dir=args.data_dir)

    print('data directory: {}'.format(args.data_dir))
    print('# of train samples: {}'.format(len(train_set)))
    print('# of dev samples: {}'.format(len(dev_set)))
    print('# of test samples: {}'.format(len(test_set)))

    ent_vocab, rel_vocab = load_ent_rel_vocabs(path='../')

    # load entity embeddings
    ent_index = []
    for k, v in temp_ent_vocab.items():
        ent_index.append(ent_vocab[k])
    ent_index = torch.tensor(ent_index)
    ent_emb = np.load(os.path.join(args.model_path, 'entities.npy'))
    ent_embedding = nn.Embedding.from_pretrained(torch.from_numpy(ent_emb))
    ent_emb = ent_embedding(ent_index.view(1, -1)).squeeze().detach()

    # load CoLAKE parameters
    config = RobertaConfig.from_pretrained('roberta-base', type_vocab_size=3)
    model = CoLAKEForRE(config,
                        num_types=len(train_set.label_vocab),
                        ent_emb=ent_emb)
    states_dict = torch.load(os.path.join(args.model_path, 'model.bin'))
    model.load_state_dict(states_dict, strict=False)
    print('parameters below are randomly initializecd:')
    for name, param in model.named_parameters():
        if name not in states_dict:
            print(name)

    # tie relation classification head
    rel_index = []
    for k, v in train_set.label_vocab.items():
        rel_index.append(rel_vocab[k])
    rel_index = torch.LongTensor(rel_index)
    rel_embeddings = nn.Embedding.from_pretrained(states_dict['rel_embeddings.weight'])
    rel_index = rel_index.cuda()
    rel_cls_weight = rel_embeddings(rel_index.view(1, -1)).squeeze()
    model.tie_rel_weights(rel_cls_weight)

    model.rel_head.dense.weight.data = states_dict['rel_lm_head.dense.weight']
    model.rel_head.dense.bias.data = states_dict['rel_lm_head.dense.bias']
    model.rel_head.layer_norm.weight.data = states_dict['rel_lm_head.layer_norm.weight']
    model.rel_head.layer_norm.bias.data = states_dict['rel_lm_head.layer_norm.bias']

    model.resize_token_embeddings(len(RobertaTokenizer.from_pretrained('roberta-base')) + 4)
    print('parameters of CoLAKE has been loaded.')

    # fine-tune
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'embedding']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, args.beta), eps=1e-6)

    metrics = [MacroMetric(pred='pred', target='target')]

    test_data_iter = TorchLoaderIter(dataset=test_set, batch_size=args.batch_size, sampler=RandomSampler(),
                                     num_workers=4,
                                     collate_fn=test_set.collate_fn)
    devices = list(range(torch.cuda.device_count()))
    tester = Tester(data=test_data_iter, model=model, metrics=metrics, device=devices)
    # tester.test()

    fitlog_callback = FitlogCallback(tester=tester, log_loss_every=100, verbose=1)
    gradient_clip_callback = GradientClipCallback(clip_value=1, clip_type='norm')
    warmup_callback = WarmupCallback(warmup=args.warm_up, schedule='linear')

    bsz = args.batch_size // args.grad_accumulation

    train_data_iter = TorchLoaderIter(dataset=train_set,
                                      batch_size=bsz,
                                      sampler=RandomSampler(),
                                      num_workers=4,
                                      collate_fn=train_set.collate_fn)
    dev_data_iter = TorchLoaderIter(dataset=dev_set,
                                    batch_size=bsz,
                                    sampler=RandomSampler(),
                                    num_workers=4,
                                    collate_fn=dev_set.collate_fn)

    trainer = Trainer(train_data=train_data_iter,
                      dev_data=dev_data_iter,
                      model=model,
                      optimizer=optimizer,
                      loss=LossInForward(),
                      batch_size=bsz,
                      update_every=args.grad_accumulation,
                      n_epochs=args.epoch,
                      metrics=metrics,
                      callbacks=[fitlog_callback, gradient_clip_callback, warmup_callback],
                      device=devices,
                      use_tqdm=True)

    trainer.train(load_best_model=False)


if __name__ == '__main__':
    main()
