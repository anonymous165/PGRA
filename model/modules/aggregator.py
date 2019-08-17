import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.attention import RelationFeatureAttention
from model.pgra_function.agg import get_agg


class Aggregator(nn.Module):

    def __init__(self, agg_name='neighbor', **kwargs):
        super(Aggregator, self).__init__()
        self.attention = RelationFeatureAttention(**kwargs)
        self.att_score = None
        self.AGG = get_agg(agg_name)

    def forward(self, self_vector, neighbor_vectors, **kwargs):
        att = self.attention(self_vector=self_vector, neighbor_vectors=neighbor_vectors, **kwargs)
        self.att_score = att
        att = att.unsqueeze(-1)
        aggregated_neighbor_vector = (att * neighbor_vectors).sum(dim=-2)
        self_vector = self.AGG(self_vector, aggregated_neighbor_vector)
        return self_vector
