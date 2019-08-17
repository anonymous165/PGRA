import math

from torch import nn as nn
from torch.nn import functional as F


class TransH(nn.Module):

    def __init__(self, n_relation, emb_size, tied_emb=None):
        super(TransH, self).__init__()
        if tied_emb is not None:
            self.rela_emb = tied_emb
        else:
            self.rela_emb = nn.Embedding(n_relation, emb_size)

    def reset_parameters(self):
        nn.init.constant_(self.rela_emb.weight, 1)

    def forward(self, node_emb, relation):
        r_emb = self.rela_emb(relation)
        while len(r_emb.size()) != len(node_emb.size()):
            r_emb = r_emb.unsqueeze(1)
        return node_emb - (node_emb * r_emb).sum(-1, keepdim=True) * r_emb

    def constraint(self):
        self.rela_emb.weight.data = F.normalize(self.rela_emb.weight.data)


class DistMult(nn.Module):

    def __init__(self, n_relation, emb_size, tied_emb=None):
        super(DistMult, self).__init__()
        if tied_emb is not None:
            self.rela_emb = tied_emb
        else:
            self.rela_emb = nn.Embedding(n_relation, emb_size)
        self.emb_size = emb_size

    def reset_parameters(self):
        nn.init.constant_(self.rela_emb.weight, 1)

    def forward(self, node_emb, relation):
        r_emb = self.rela_emb(relation)
        while len(r_emb.size()) != len(node_emb.size()):
            r_emb = r_emb.unsqueeze(1)
        return node_emb * r_emb * math.sqrt(self.emb_size)

    def constraint(self):
        self.rela_emb.weight.data = F.normalize(self.rela_emb.weight.data)