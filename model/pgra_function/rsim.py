import torch
from torch.nn import functional as F


class AbsCosineSim(torch.nn.Module):

    def forward(self, a, b):
        return torch.abs((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1))


_map_rsim = {
    'abs_cos': AbsCosineSim
}


def get_rsim(rsim_name):
    return _map_rsim[rsim_name.lower()]
