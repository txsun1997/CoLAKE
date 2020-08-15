import torch
from fastNLP import AccuracyMetric
from fastNLP.core.utils import _get_func_signature


class MLMAccuracyMetric(AccuracyMetric):

    def __init__(self, pred=None, target=None, seq_len=None):
        super(MLMAccuracyMetric, self).__init__(pred, target, seq_len)

    def evaluate(self, pred, target, seq_len=None):

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if seq_len is not None and not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        masks = target != -1 # ignore_index = -1

        if pred.dim() == target.dim():
            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        target = target.to(pred)

        self.acc_count += torch.sum(torch.eq(pred, target).masked_fill(masks.eq(0), 0)).item()
        self.total += torch.sum(masks).item()


class WordMLMAccuracy(MLMAccuracyMetric):
    def __init__(self, pred=None, target=None, seq_len=None):
        super(WordMLMAccuracy, self).__init__(pred, target, seq_len)


class EntityMLMAccuracy(MLMAccuracyMetric):
    def __init__(self, pred=None, target=None, seq_len=None):
        super(EntityMLMAccuracy, self).__init__(pred, target, seq_len)


class RelationMLMAccuracy(MLMAccuracyMetric):
    def __init__(self, pred=None, target=None, seq_len=None):
        super(RelationMLMAccuracy, self).__init__(pred, target, seq_len)
