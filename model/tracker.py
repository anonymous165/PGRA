import numpy as np
import torch
import copy


class LossTracker:

    def __init__(self):
        self.step = 0
        self.losses = None

    def update(self, losses):
        if not isinstance(losses, tuple) and not isinstance(losses, list):
            losses = (losses,)
        losses = np.array([x.item() for x in losses])
        if self.losses is None:
            self.losses = losses
        else:
            self.losses = ((self.losses * self.step) + losses) / (self.step + 1)
        self.step += 1

    @property
    def value(self):
        return sum(self.losses)

    def __str__(self):
        return "{:.5f} ".format(sum(self.losses)) + ' '.join("{:.5f}".format(loss) for loss in self.losses)


class MultiClsTracker:

    def __init__(self, metrics, n_classes):
        self.metric_scores = {metric: np.zeros(n_classes) for metric in metrics}
        self.n_samples = np.zeros(n_classes) + 1e-8
        self.num_classes = n_classes
        self.metric_summary = {}
        self._is_updated = False

    def __len__(self):
        return int(self.n_samples.sum())

    def update(self, pred_scores, classes):
        for metric, values in self.metric_scores.items():
            metric_score = get_metric(metric)(pred_scores)
            for _r_type in range(self.num_classes):
                values[_r_type] += ((classes == _r_type).float() * metric_score).sum().item()
        for _r_type in range(self.num_classes):
            self.n_samples[_r_type] += (classes == _r_type).float().sum().item()
        self._is_updated = True

    def summarize(self):
        metric_summary = {}
        for metric, values in self.metric_scores.items():
            metric_summary[metric + '_macro'] = (values / self.n_samples).mean()
            metric_summary[metric + '_micro'] = values.sum() / self.n_samples.sum()
            metric_summary[metric] = (metric_summary[metric + '_micro'] + metric_summary[metric + '_macro']) / 2
        self.metric_summary = metric_summary
        self._is_updated = False

    def __getitem__(self, item):
        if self._is_updated:
            self.summarize()
        return self.metric_summary[item]

    def value(self):
        return copy.deepcopy(self.metric_summary)

    @property
    def macro(self):
        return {metric: values / self.n_samples for metric, values in self.metric_scores.items()}

    def __str__(self):
        if self._is_updated:
            self.summarize()
        return ', '.join([metric + ": {:.5f}".format(score) for metric, score in self.metric_summary.items()])

    def __format__(self, format_spec):
        return str(self)


def get_metric(metric):
    mapper = {
        'mrr': mrr
    }
    return mapper[metric.lower()]


def mrr(scores):
    scores += torch.rand_like(scores) * 1e-8  # Alleviate same scores
    rank = (scores - scores[:, :1] > 0).sum(-1) + 1
    return 1. / rank.float()
