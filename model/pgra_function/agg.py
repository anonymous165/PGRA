import torch.nn as nn


class NeighborAGG(nn.Module):

    def forward(self, self_vector, nb_vector):
        return nb_vector


_map_agg = {
    'neighbor': NeighborAGG
}


def get_agg(agg_name):
    return _map_agg[agg_name.lower()]
