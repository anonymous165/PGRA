import torch
import torch.nn as nn


def l1_loss(x):
    return torch.abs(x).sum(-1)


def l2_loss(x):
    return (x**2).sum(-1)


def get_loss(loss):
    _loss_map = {
        'l2': l2_loss,
        'l1': l1_loss,
    }
    assert loss in _loss_map
    return _loss_map[loss]


class Regularizer(nn.Module):

    def __init__(self, loss='l2', l_reg=1.):
        super().__init__()
        self.loss = None
        self.l_reg = l_reg
        self.loss_func = get_loss(loss)

    def forward(self, x, weights=None):
        if self.training and self.l_reg != 0:
            if weights is not None:
                loss = (weights * self.loss_func(x)).sum()
            else:
                loss = self.loss_func(x).mean()
            loss = loss * self.l_reg
            if self.loss is None:
                self.loss = loss
            else:
                self.loss = self.loss + loss