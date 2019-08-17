import torch
from torch.nn import functional as F


class CosineSim(torch.nn.Module):

    def forward(self, a, b):
        return torch.abs((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1))
