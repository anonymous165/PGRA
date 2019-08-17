import torch
from torch.nn import functional as F


class CosineSim(torch.nn.Module):

    def forward(self, a, b):
        # return (a*b).sum(-1)
        # return torch.abs((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1))

        # return ((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1) + 1) / 2
        # return torch.abs(a * b).sum(dim=-1)
        return F.relu((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1))
