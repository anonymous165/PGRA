import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregator(nn.Module):

    def __init__(self, **kwargs):
        super(Aggregator, self).__init__()
        print('unused_kwargs:', kwargs)
        self.att = None

    def forward(self, neighbor_vectors, **kwargs):
        att = self.get_attention(neighbor_vectors=neighbor_vectors, **kwargs)
        self.att = att
        att = att.unsqueeze(-1)
        output = (att * neighbor_vectors).sum(dim=-2)
        return output

    def get_attention(self, *args, **kwargs):
        raise NotImplementedError


class FeatAggregator(Aggregator):

    def __init__(self, emb_size, use_weight=False, **kwargs):
        super(FeatAggregator, self).__init__(**kwargs)
        if use_weight:
            self.att_weight = nn.Linear(emb_size, emb_size, bias=False)
        else:
            self.att_weight = lambda x: x
        self.att_h = nn.Linear(emb_size, 1)
        self.att_t = nn.Linear(emb_size, 1)
        self.att_act = nn.LeakyReLU(0.2)

    def reset_parameters(self):
        if isinstance(self.att_weight, nn.Linear):
            nn.init.xavier_uniform_(self.att_weight.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att_h.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att_t.weight, gain=1.414)

    def get_attention(self, self_vector, neighbor_vectors, softmax=True, **kwargs):
        self_vector = self.att_weight(self_vector).unsqueeze(-2)
        neighbor_vectors = self.att_weight(neighbor_vectors)
        att = (self.att_h(self_vector) + self.att_t(neighbor_vectors)).squeeze(-1)
        att = self.att_act(att)
        if softmax:
            att = F.softmax(att, dim=-1)
        return att


class RelaAggregator(Aggregator):

    def __init__(self, rsim, **kwargs):
        super(RelaAggregator, self).__init__(**kwargs)
        self.rsim = rsim

    def get_attention(self, relation_vectors, neighbor_relation_vectors, softmax=True, **kwargs):
        neighbor_relation_vectors = neighbor_relation_vectors.detach()
        while len(relation_vectors.size()) != len(neighbor_relation_vectors.size()):
            relation_vectors = relation_vectors.unsqueeze(-2)
        att = self.rsim(relation_vectors, neighbor_relation_vectors)
        if softmax:
            att = F.softmax(att, dim=-1)
        return att


class RelaFeatAggregator(FeatAggregator, RelaAggregator):

    def __init__(self, *args, **kwargs):
        super(RelaFeatAggregator, self).__init__(*args, **kwargs)

    def get_attention(self, **kwargs):
        att_f = FeatAggregator.get_attention(self, **kwargs, softmax=False)
        att_r = RelaAggregator.get_attention(self, **kwargs, softmax=False)
        pre_att = att_r * att_f
        att = F.softmax(pre_att, dim=-1)
        return att
