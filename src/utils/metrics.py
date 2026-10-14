import numpy as np
import torch
from sklearn import metrics


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


class Metrics:
    def __init__(self, loss=None, predicts=None, targets=None):
        self._loss = float(loss) if loss is not None else 0.0
        self._targets = list(to_numpy(targets)) if targets is not None else []
        self._predicts = list(to_numpy(predicts)) if predicts is not None else []

    def update(self, other):
        if other is not None:
            self._predicts.extend(to_numpy(other._predicts))
            self._targets.extend(to_numpy(other._targets))
            self._loss += other._loss

    def _calculate(self, metric, **kwargs):
        if len(self._targets) == 0 or len(self._predicts) == 0:
            return 0.0
        return metric(self._targets, self._predicts, **kwargs)

    @property
    def loss(self):
        if len(self._targets) > 0:
            return self._loss / len(self._targets)
        return 0.0

    @property
    def accuracy(self):
        return self._calculate(metrics.accuracy_score) * 100

    @property
    def macro_precision(self):
        return self._calculate(metrics.precision_score, average="macro", zero_division=0) * 100

    @property
    def macro_recall(self):
        return self._calculate(metrics.recall_score, average="macro", zero_division=0) * 100

    @property
    def macro_f1(self):
        return self._calculate(metrics.f1_score, average="macro", zero_division=0) * 100
