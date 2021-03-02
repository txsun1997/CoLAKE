import os
import sys
import numpy as np

sys.path.append('../')

import fitlog
import argparse
import torch
from torch import optim
import torch.distributed as dist
from transformers import RobertaTokenizer
from fastNLP import RandomSampler, TorchLoaderIter, LossInForward, Tester, DistTrainer, get_local_rank
from fastNLP import FitlogCallback, WarmupCallback, GradientClipCallback, logger, init_logger_dist
from pretrain.model import CoLAKE
from pretrain.utils import OTFDistributedSampler, SaveModelCallback
from pretrain.utils import load_ent_rel_vocabs, get_ent_freq, MyFitlogCallback
from pretrain.dataset import GraphOTFDataSet, GraphDataSet, FewRelDevDataSet
from pretrain.metrics import WordMLMAccuracy, EntityMLMAccuracy, RelationMLMAccuracy
from pretrain.large_emb import EmbUpdateCallback
from transformers import PYTORCH_PRETRAINED_BERT_CACHE, RobertaConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test', help="experiment name")
    parser.add_argument('--data_dir', type=str,
                        default='../pretrain_data/data',
                        help="data directory path")
    parser.add_argument('--test_data', type=str, default=None,
                        help="fewrel test data directory path")
    parser.add_argument('--save_dir', type=str, default='../ckpts/',
                        help="model directory path")
    parser.add_argument('--log_dir', type=str, default='./pretrain_logs',
                        help="fitlog directory path")
    parser.add_argument('--rel_emb', type=str,
                        default='../wikidata5m_alias_emb/relations.npy')
    parser.add_argument('--kg_path', type=str, default='../wikidata5m')
    parser.add_argument('--emb_name', type=str, default='entity_emb')
    parser.add_argument('--data_prop', type=float, default=0.3, help="using what proportion of wiki to train")
    parser.add_argument('--n_negs', type=int, default=100, help="number of negative samples")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--beta', type=float, default=0.999, help="beta_2 in Adam")
    parser.add_argument('--warm_up', type=float, default=0.1, help="warmup proportion or steps")
    parser.add_argument('--epoch', type=int, default=1, help="number of epochs")
    parser.add_argument('--grad_accumulation', type=int, default=4, help="gradient accumulation")
    parser.add_argument('--local_rank', type=int, default=0, help="local rank")
    parser.add_argument('--fp16', action='store_true', help="whether to use fp16")
    parser.add_argument('--save_model', action='store_true', help="whether save model")
    parser.add_argument('--do_test', action='store_true', help="test trained model")
    parser.add_argument('--debug', action='store_true', help="do not log")
    parser.add_argument('--model_name', type=str, default=None, help="test or further train")
    parser.add_argument('--ent_dim', type=int, default=200, help="dimension of entity embeddings")
    parser.add_argument('--rel_dim', type=int, default=200, help="dimension of relation embeddings")
    parser.add_argument('--ip_config', type=str, default='emb_ip.cfg')
    parser.add_argument('--ent_lr', type=float, default=1e-4, help="entity embedding learning rate")
    return parser.parse_args()


def train():
    args = parse_args()
    if args.debug:
        fitlog.debug()
        args.save_model = False
    # ================= define =================
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    word_mask_index = tokenizer.mask_token_id
    word_vocab_size = len(tokenizer)

    if get_local_rank() == 0: 
        fitlog.set_log_dir(args.log_dir)
        fitlog.commit(__file__, fit_msg=args.name)
        fitlog.add_hyper_in_file(__file__)
        fitlog.add_hyper(args)

    # ================= load data =================
    dist.init_process_group('nccl')
    init_logger_dist()

    n_proc = dist.get_world_size()
    bsz = args.batch_size // args.grad_accumulation // n_proc
    args.local_rank = get_local_rank()
    args.save_dir = os.path.join(args.save_dir, args.name) if args.save_model else None
    if args.save_dir is not None and os.path.exists(args.save_dir):
        raise RuntimeError('save_dir has already existed.')
    logger.info('save directory: {}'.format('None' if args.save_dir is None else args.save_dir))
    devices = list(range(torch.cuda.device_count()))
    NUM_WORKERS = 4

    ent_vocab, rel_vocab = load_ent_rel_vocabs()
    logger.info('# entities: {}'.format(len(ent_vocab)))
    logger.info('# relations: {}'.format(len(rel_vocab)))
    ent_freq = get_ent_freq()
    assert len(ent_vocab) == len(ent_freq), '{} {}'.format(len(ent_vocab), len(ent_freq))

    #####
    root = args.data_dir
    dirs = os.listdir(root)
    drop_files = []
    for dir in dirs:
        path = os.path.join(root, dir)
        max_idx = 0
        for file_name in os.listdir(path):
            if 'large' in file_name:
                continue
            max_idx = int(file_name) if int(file_name) > max_idx else max_idx
        drop_files.append(os.path.join(path, str(max_idx)))
    #####

    file_list = []
    for path, _, filenames in os.walk(args.data_dir):
        for filename in filenames:
            file = os.path.join(path, filename)
            if 'large' in file or file in drop_files:
                continue
            file_list.append(file)
    logger.info('used {} files in {}.'.format(len(file_list), args.data_dir))
    if args.data_prop > 1:
        used_files = file_list[:int(args.data_prop)]
    else:
        used_files = file_list[:round(args.data_prop * len(file_list))]

    data = GraphOTFDataSet(used_files, n_proc, args.local_rank, word_mask_index, word_vocab_size,
                           args.n_negs, ent_vocab, rel_vocab, ent_freq)
    dev_data = GraphDataSet(used_files[0], word_mask_index, word_vocab_size, args.n_negs, ent_vocab,
                            rel_vocab, ent_freq)

    sampler = OTFDistributedSampler(used_files, n_proc, get_local_rank())
    train_data_iter = TorchLoaderIter(dataset=data, batch_size=bsz, sampler=sampler, num_workers=NUM_WORKERS,
                                      collate_fn=data.collate_fn)
    dev_data_iter = TorchLoaderIter(dataset=dev_data, batch_size=bsz, sampler=RandomSampler(),
                                    num_workers=NUM_WORKERS,
                                    collate_fn=dev_data.collate_fn)
    if args.test_data is not None:
        test_data = FewRelDevDataSet(path=args.test_data, label_vocab=rel_vocab, ent_vocab=ent_vocab)
        test_data_iter = TorchLoaderIter(dataset=test_data, batch_size=32, sampler=RandomSampler(),
                                         num_workers=NUM_WORKERS,
                                         collate_fn=test_data.collate_fn)

    if args.local_rank == 0:
        print('full wiki files: {}'.format(len(file_list)))
        print('used wiki files: {}'.format(len(used_files)))
        print('# of trained samples: {}'.format(len(data) * n_proc))
        print('# of trained entities: {}'.format(len(ent_vocab)))
        print('# of trained relations: {}'.format(len(rel_vocab)))

    # ================= prepare model =================
    logger.info('model init')
    if args.rel_emb is not None:  # load pretrained relation embeddings
        rel_emb = np.load(args.rel_emb)
        # add_embs = np.random.randn(3, rel_emb.shape[1])  # add <pad>, <mask>, <unk>
        # rel_emb = np.r_[add_embs, rel_emb]
        rel_emb = torch.from_numpy(rel_emb).float()
        assert rel_emb.shape[0] == len(rel_vocab), '{} {}'.format(rel_emb.shape[0], len(rel_vocab))
        # assert rel_emb.shape[1] == args.rel_dim
        logger.info('loaded pretrained relation embeddings. dim: {}'.format(rel_emb.shape[1]))
    else:
        rel_emb = None
    if args.model_name is not None:
        logger.info('further pre-train.')
        config = RobertaConfig.from_pretrained('roberta-base', type_vocab_size=3)
        model = CoLAKE(config=config,
                       num_ent=len(ent_vocab),
                       num_rel=len(rel_vocab),
                       ent_dim=args.ent_dim,
                       rel_dim=args.rel_dim,
                       ent_lr=args.ent_lr,
                       ip_config=args.ip_config,
                       rel_emb=None,
                       emb_name=args.emb_name)
        states_dict = torch.load(args.model_name)
        model.load_state_dict(states_dict, strict=True)
    else:
        model = CoLAKE.from_pretrained('roberta-base',
                                       num_ent=len(ent_vocab),
                                       num_rel=len(rel_vocab),
                                       ent_lr=args.ent_lr,
                                       ip_config=args.ip_config,
                                       rel_emb=rel_emb,
                                       emb_name=args.emb_name,
                                       cache_dir=PYTORCH_PRETRAINED_BERT_CACHE + '/dist_{}'.format(args.local_rank))
        model.extend_type_embedding(token_type=3)
    # if args.local_rank == 0:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad is True:
    #             print('{}: {}'.format(name, param.shape))

    # ================= train model =================
    # lr=1e-4 for peak value, lr=5e-5 for initial value
    logger.info('trainer init')
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.bias', 'layer_norm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    word_acc = WordMLMAccuracy(pred='word_pred', target='masked_lm_labels', seq_len='word_seq_len')
    ent_acc = EntityMLMAccuracy(pred='entity_pred', target='ent_masked_lm_labels', seq_len='ent_seq_len')
    rel_acc = RelationMLMAccuracy(pred='relation_pred', target='rel_masked_lm_labels', seq_len='rel_seq_len')
    metrics = [word_acc, ent_acc, rel_acc]

    if args.test_data is not None:
        test_metric = [rel_acc]
        tester = Tester(data=test_data_iter, model=model, metrics=test_metric, device=list(range(torch.cuda.device_count())))
        # tester.test()
    else:
        tester = None

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, args.beta), eps=1e-6)
    # warmup_callback = WarmupCallback(warmup=args.warm_up, schedule='linear')
    fitlog_callback = MyFitlogCallback(tester=tester, log_loss_every=100, verbose=1)
    gradient_clip_callback = GradientClipCallback(clip_value=1, clip_type='norm')
    emb_callback = EmbUpdateCallback(model.ent_embeddings)
    all_callbacks = [gradient_clip_callback, emb_callback]
    if args.save_dir is None:
        master_callbacks = [fitlog_callback]
    else:
        save_callback = SaveModelCallback(args.save_dir, model.ent_embeddings, only_params=True)
        master_callbacks = [fitlog_callback, save_callback]

    if args.do_test:
        states_dict = torch.load(os.path.join(args.save_dir, args.model_name)).state_dict()
        model.load_state_dict(states_dict)
        data_iter = TorchLoaderIter(dataset=data, batch_size=args.batch_size, sampler=RandomSampler(),
                                    num_workers=NUM_WORKERS,
                                    collate_fn=data.collate_fn)
        tester = Tester(data=data_iter, model=model, metrics=metrics, device=devices)
        tester.test()
    else:
        trainer = DistTrainer(train_data=train_data_iter,
                              dev_data=dev_data_iter,
                              model=model,
                              optimizer=optimizer,
                              loss=LossInForward(),
                              batch_size_per_gpu=bsz,
                              update_every=args.grad_accumulation,
                              n_epochs=args.epoch,
                              metrics=metrics,
                              callbacks_master=master_callbacks,
                              callbacks_all=all_callbacks,
                              validate_every=5000,
                              use_tqdm=True,
                              fp16='O1' if args.fp16 else '')
        trainer.train(load_best_model=False)


if __name__ == '__main__':
    train()
