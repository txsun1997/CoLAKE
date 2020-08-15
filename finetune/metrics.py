import torch
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from sklearn.metrics import f1_score, precision_recall_fscore_support


class MacroMetric(MetricBase):
    def __init__(self, pred=None, target=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=None)
        self._target = []
        self._pred = []

    def evaluate(self, pred, target, seq_len=None):
        '''
        :param pred: batch_size
        :param target: batch_size
        :param seq_len: not uesed when doing text classification
        :return:
        '''

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if pred.dim() != target.dim():
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        pred = pred.detach().cpu().numpy().tolist()
        target = target.to('cpu').numpy().tolist()
        self._pred.extend(pred)
        self._target.extend(target)

    def get_metric(self, reset=True):
        precision, recall, f_score, _ = precision_recall_fscore_support(self._target, self._pred, average='macro')
        evaluate_result = {
            'f_score': f_score,
            'precision': precision,
            'recall': recall,
        }
        if reset:
            self._pred = []
            self._target = []

        return evaluate_result


class MicroMetric(MetricBase):
    def __init__(self, pred=None, target=None, no_relation_idx=0):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=None)
        self.no_relation = no_relation_idx
        self.num_predict = 0
        self.num_golden = 0
        self.true_positive = 0

    def evaluate(self, pred, target, seq_len=None):
        '''
        :param pred: batch_size
        :param target: batch_size
        :param seq_len: not uesed when doing text classification
        :return:
        '''

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if pred.dim() != target.dim():
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        preds = pred.detach().cpu().numpy().tolist()
        targets = target.to('cpu').numpy().tolist()
        for pred, target in zip(preds, targets):
            if pred == target and pred != self.no_relation:
                self.true_positive += 1
            if target != self.no_relation:
                self.num_golden += 1
            if pred != self.no_relation:
                self.num_predict += 1

    def get_metric(self, reset=True):
        if self.num_predict > 0:
            micro_precision = self.true_positive / self.num_predict
        else:
            micro_precision = 0.
        micro_recall = self.true_positive / self.num_golden
        micro_fscore = self._calculate_f1(micro_precision, micro_recall)
        evaluate_result = {
            'f_score': micro_fscore,
            'precision': micro_precision,
            'recall': micro_recall
        }

        if reset:
            self.num_predict = 0
            self.num_golden = 0
            self.true_positive = 0

        return evaluate_result

    def _calculate_f1(self, p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)


class TypingMetric(MetricBase):
    def __init__(self, pred=None, target=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=None)
        self.acc_count = 0
        self.total = 0
        self._target = []
        self._pred = []

    def evaluate(self, pred, target, seq_len=None):
        '''
        :param pred: batch_size x num_labels
        :param target: batch_size x num_labels
        :param seq_len: not uesed when doing text classification
        :return:
        '''

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if pred.dim() != target.dim():
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        pred = pred.detach().cpu().numpy()
        target = target.to('cpu').numpy()
        cnt = 0
        y1, y2 = [], []
        for x1, x2 in zip(pred, target):
            yy1 = []
            yy2 = []
            for i in range(len(x1)):
                if x1[i] > 0:
                    yy1.append(i)
                if x2[i] > 0:
                    yy2.append(i)
            y1.append(yy1)
            y2.append(yy2)
            cnt += set(yy1) == set(yy2)

        self.acc_count += cnt
        self.total += len(pred)
        self._pred.extend(y1)
        self._target.extend(y2)

    def get_metric(self, reset=True):
        # for calculating macro F1
        num_predict, num_golden = 0, 0
        p = 0.
        r = 0.
        # for calculating micro F1
        num_predicted_labels = 0.
        num_golden_labels = 0.
        num_correct_labels = 0.

        for true_labels, predicted_labels in zip(self._target, self._pred):
            overlap = len(set(predicted_labels).intersection(set(true_labels)))
            # calculating macro F1
            if len(predicted_labels) > 0:
                p += overlap / float(len(predicted_labels))
                num_predict += 1
            if len(true_labels) > 0:
                r += overlap / float(len(true_labels))
                num_golden += 1
            # calculating micro F1
            num_predicted_labels += len(predicted_labels)
            num_golden_labels += len(true_labels)
            num_correct_labels += overlap

        if num_predict > 0:
            macro_precision = p / num_predict
        else:
            macro_precision = 0.
        macro_recall = r / num_golden
        macro = self._calculate_f1(macro_precision, macro_recall)

        if num_predicted_labels > 0:
            micro_precision = num_correct_labels / num_predicted_labels
        else:
            micro_precision = 0.
        micro_recall = num_correct_labels / num_golden_labels
        micro = self._calculate_f1(micro_precision, micro_recall)

        evaluate_result = {'micro_f': micro,
                           'micro_p': micro_precision,
                           'micro_r': micro_recall,
                           'acc': round(float(self.acc_count) / (self.total + 1e-12), 6),
                           # 'macro_p': macro_precision,
                           # 'macro_r': macro_recall,
                           # 'macro_f': macro,
                           }
        if reset:
            self.acc_count = 0
            self.total = 0
            self._pred = []
            self._target = []

        return evaluate_result

    def _calculate_f1(self, p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)