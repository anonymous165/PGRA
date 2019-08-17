import torch
import torch.nn as nn


class NeighborRegularizer(nn.Module):

    def __init__(self, p=2, l_reg=1.):
        super().__init__()
        self.loss = None
        self.p = p
        self.l_reg = l_reg

    def forward(self, self_vector, nb_vector, weights=None):
        if self.training and self.l_reg != 0:
            losses = (torch.abs(self_vector-nb_vector)**2).sum(-1)
            if weights is not None:
                loss = (weights * losses).sum()
            else:
                loss = losses.mean()
            loss = loss * self.l_reg
            if self.loss is None:
                self.loss = loss
            else:
                self.loss = self.loss + loss
