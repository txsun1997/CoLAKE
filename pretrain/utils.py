import os
import random
import pickle
import torch
import collections
import numpy as np
from fastNLP import Callback, cache_results
from torch.utils.data import Sampler
from itertools import chain
import pandas as pd
from copy import deepcopy
from fastNLP import Tester, DataSet
import fitlog


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


class OTFDistributedSampler(Sampler):
    # On-The-Fly sampler
    def __init__(self, indexed_train_fps, n_workers, rank, shuffle=True):
        super(OTFDistributedSampler, self).__init__(0)
        self.epoch = 0
        self.shuffle = shuffle

        file_per_process = len(indexed_train_fps) // n_workers
        if file_per_process * n_workers != len(indexed_train_fps):
            if rank == 0:
                print('[Sampler] Drop {} files.'.format(len(indexed_train_fps) - file_per_process * n_workers))
                print('[Sampler] # files per process: {}'.format(file_per_process))
        self.fps = indexed_train_fps[rank * file_per_process:(rank + 1) * file_per_process]
        self.file_per_process = file_per_process

        data = []
        with open(self.fps[0], 'r', encoding='utf-8') as fin:
            import json
            for x in fin:
                data.append(json.loads(x))
            self.num_samples_per_file = len(data)
        assert self.num_samples_per_file == 5000
        self.total_num_samples = self.num_samples_per_file * len(self.fps)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = []
            for i in range(self.file_per_process):
                indexes = list(np.arange(self.num_samples_per_file) + i * self.num_samples_per_file) # indices within one file
                np.random.shuffle(indexes)
                indices.append(indexes)
            np.random.shuffle(indices)
            indices = list(chain(*indices))
        else:
            indices = []
            for i in range(self.file_per_process):
                indexes = list(np.arange(self.num_samples_per_file) + i * self.num_samples_per_file)
                indices.extend(indexes)

        return iter(indices)

    def __len__(self):
        return self.total_num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class SaveModelCallback(Callback):
    def __init__(self, save_path, ent_emb, only_params=True):
        super(SaveModelCallback, self).__init__()
        self.save_path = save_path
        self.only_params = only_params
        self.ent_emb = ent_emb

    def on_epoch_end(self):
        if self.is_master:
            path = os.path.join(self.save_path, 'epoch_' + str(self.epoch))
            os.makedirs(path, exist_ok=True)
            model_path = os.path.join(path, 'model.bin')
            model_to_save = self.trainer.ddp_model.module
            if self.only_params:
                model_to_save = model_to_save.state_dict()
            torch.save(model_to_save, model_path)
            self.ent_emb.save(path)
            self.trainer.logger.info('Saved checkpoint to {}.'.format(path))


def create_mlm_labels(tokens, mask_index, vocab_size, masked_lm_prob=0.15, max_predictions_per_seq=15, anchor_nodes=None):
    rng = random.Random(2020)
    cand_indexes = []
    if mask_index == 50264:  # indicates word nodes
        special_tokens = [0, 1, 2, 3]  # 0: <s>, 1: <pad>, 2: </s>, 3: <unk>
    else:
        special_tokens = [0, 1]  # 0: <unk> 1: <pad>
    for (i, token) in enumerate(tokens):
        if token in special_tokens:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))
    masked_labels = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_labels) >= num_to_predict:
            if anchor_nodes is None:
                break
            elif tokens[index] not in anchor_nodes:
                continue
            else: # tokens[index] is anchor node
                if index in covered_indexes:
                    continue
                covered_indexes.add(index)
                if rng.random() < 0.8:
                    masked_token = tokens[index]  # 以80%概率是本身
                else:
                    if rng.random() < 0.5:
                        masked_token = mask_index
                    else:
                        masked_token = rng.randint(0, vocab_size - 1)
        else:
            if index in covered_indexes:
                continue
            covered_indexes.add(index)
            if rng.random() < 0.8:
                masked_token = mask_index  # [MASK]
            else:
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                else:
                    masked_token = rng.randint(0, vocab_size - 1)
        output_tokens[index] = masked_token
        masked_labels.append(MaskedLmInstance(index=index, label=tokens[index]))
    masked_labels = sorted(masked_labels, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_labels:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    masked_labels = np.ones(len(tokens), dtype=int) * -1
    masked_labels[masked_lm_positions] = masked_lm_labels
    masked_labels = list(masked_labels)
    return output_tokens, masked_labels


class MyFitlogCallback(Callback):
    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=0, log_exception=False):
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        assert isinstance(log_loss_every, int) and log_loss_every >= 0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

        self.verbose = verbose
        self._log_loss_every = log_loss_every
        self._avg_loss = 0

    def on_train_begin(self):
        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.batch_size),
                                metrics=self.trainer.metrics,
                                verbose=0,
                                use_tqdm=self.trainer.test_use_tqdm)
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        if len(self.testers) > 0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    if self.verbose != 0:
                        self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                        self.pbar.write(tester._format_eval_results(eval_result))
                    fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                    if better_result:
                        fitlog.add_best_metric(eval_result, name=key)
                except Exception as e:
                    self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

    def on_train_end(self):
        fitlog.finish()

    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')


@cache_results(_cache_fp='ent_freq.bin', _refresh=False)
def get_ent_freq(path):
    with open(path, 'rb') as fin:
        ent_freq = pickle.load(fin)
    print('# of entities: {}'.format(len(ent_freq)))
    return ent_freq


@cache_results(_cache_fp='ent_rel_vocab.bin', _refresh=False)
def load_ent_rel_vocabs(path):
    with open(os.path.join(path, 'read_ent_vocab.bin'), 'rb') as fin:
        ent_vocab = pickle.load(fin)
    print('# of entities: {}'.format(len(ent_vocab)))

    with open(os.path.join(path, 'read_rel_vocab.bin'), 'rb') as fin:
        rel_vocab = pickle.load(fin)
    print('# of relations: {}'.format(len(rel_vocab)))

    return ent_vocab, rel_vocab





