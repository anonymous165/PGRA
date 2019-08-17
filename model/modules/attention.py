import torch.nn as nn
import torch.nn.functional as F


class FeatureAttention(nn.Module):

    def __init__(self, emb_size, use_weight=False, **kwargs):
        super(FeatureAttention, self).__init__()
        if use_weight:
            self.att_weight = nn.Linear(emb_size, emb_size, bias=False)
        else:
            self.att_weight = lambda x: x
        self.att_h = nn.Linear(emb_size, 1, bias=False)
        self.att_t = nn.Linear(emb_size, 1, bias=False)
        self.att_act = nn.LeakyReLU(0.2)

    def reset_parameters(self):
        if isinstance(self.att_weight, nn.Linear):
            nn.init.xavier_uniform_(self.att_weight.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att_h.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att_t.weight, gain=1.414)

    def forward(self, self_vector, neighbor_vectors, softmax=True, **kwargs):
        self_vector = self.att_weight(self_vector).unsqueeze(-2)
        neighbor_vectors = self.att_weight(neighbor_vectors)
        att = (self.att_h(self_vector) + self.att_t(neighbor_vectors)).squeeze(-1)
        att = self.att_act(att)
        if softmax:
            att = F.softmax(att, dim=-1)
        return att


class RelationAttention(nn.Module):

    def __init__(self, rsim, **kwargs):
        super(RelationAttention, self).__init__()
        self.rsim = rsim

    def forward(self, relation_vectors, neighbor_relation_vectors, softmax=True, **kwargs):
        neighbor_relation_vectors = neighbor_relation_vectors.detach()
        while len(relation_vectors.size()) != len(neighbor_relation_vectors.size()):
            relation_vectors = relation_vectors.unsqueeze(-2)
        att = self.rsim(relation_vectors, neighbor_relation_vectors)
        if softmax:
            att = F.softmax(att, dim=-1)
        return att


class RelationFeatureAttention(RelationAttention, FeatureAttention):

    def __init__(self, *args, **kwargs):
        super(RelationFeatureAttention, self).__init__(*args, **kwargs)

    def forward(self, **kwargs):
        att_f = FeatureAttention.forward(self, **kwargs, softmax=False)
        att_r = RelationAttention.forward(self, **kwargs, softmax=False)
        pre_att = att_r * att_f
        att = F.softmax(pre_att, dim=-1)
        return att
