import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.projection import TransH, DistMult
from model.modules.rsim import CosineSim
from model.modules.weight_init import weight_init, manual_init
from model.modules.regularizer import Regularizer
import model.modules.aggregator as agg

import numpy as np


def normalize(tensor):
    node_emb_norm = torch.norm(tensor, dim=-1, keepdim=True)
    return torch.where(node_emb_norm > 1, F.normalize(tensor, dim=-1), tensor)


class PGRA(nn.Module):

    def __init__(self, n_node, n_relation, emb_size, n_hop, n_neighbor, project, norm=False, nb_loss='l2', l_reg=0.):
        super().__init__()
        self.node_emb = nn.Embedding(n_node, emb_size)
        self.rela_emb = nn.Embedding(n_relation, emb_size)
        self.emb_size = emb_size
        self.n_hop = n_hop
        self.n_neighbor = n_neighbor
        self.norm = norm

        rsim = CosineSim()

        project = project.lower()
        if project == 'transh':
            self._projection = TransH(n_relation, emb_size, self.rela_emb)
        elif project == 'distmult':
            self._projection = DistMult(n_relation, emb_size, self.rela_emb)
        else:
            raise Exception()
        self.projection = self._projection

        self.aggregators = nn.ModuleList([
            agg.RelaFeatAggregator(emb_size=emb_size, rsim=rsim) for _ in range(n_hop)
        ])

        adj_node = np.zeros((n_node, n_neighbor))
        adj_rela = np.zeros((n_node, n_neighbor))
        self.register_neighbors(adj_node, adj_rela)

        nb_loss = nb_loss.lower()
        if nb_loss in ('l1', 'l2'):
            self.reg = Regularizer(loss=nb_loss, l_reg=l_reg)
            self.score_reg = lambda x, y: x-y
        else:
            raise Exception()

    def register_neighbors(self, adj_node, adj_rela, cuda=False):
        adj_node = torch.LongTensor(adj_node)
        adj_rela = torch.LongTensor(adj_rela)
        if cuda:
            adj_node = adj_node.cuda()
            adj_rela = adj_rela.cuda()
        self.register_buffer('adj_node', adj_node)
        self.register_buffer('adj_rela', adj_rela)

    def get_projected_emb(self, index, target_rela=None):
        emb = self.node_emb(index)
        emb = self.projection(emb, target_rela)
        return emb

    def reset(self):
        self.apply(weight_init)
        self.apply(manual_init)
        self.reset_parameters()

    def reset_parameters(self):
        self.constraint()

    def get_neighbors(self, nodes):
        batch_size = nodes.size(0)
        entities = [nodes.unsqueeze(-1)]
        relations = []
        for i in range(self.n_hop):
            neighbor_entities = F.embedding(entities[i], self.adj_node).view(batch_size, -1)
            neighbor_relations = F.embedding(entities[i], self.adj_rela).view(batch_size, -1)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, target_rela, entity, relation):
        if relation is not None:
            target_rela_emb = self.rela_emb(target_rela)
        else:
            target_rela_emb = None
        batch_size = entity[0].size(0)
        entity_vectors = [self.get_projected_emb(_entities, target_rela) for _entities in entity]
        relation_vectors = [self.rela_emb(_relations) for _relations in relation]

        shape = [batch_size, -1, self.n_neighbor, self.emb_size]
        for i in range(self.n_hop):
            entity_vectors_next_iter = []
            for hop in range(self.n_hop - i):
                neighbor_vectors = entity_vectors[hop + 1].view(*shape)
                vector = self.aggregators[i](self_vector=entity_vectors[hop],
                                             neighbor_vectors=neighbor_vectors,
                                             neighbor_relation_vectors=relation_vectors[hop].view(*shape),
                                             relation_vectors=target_rela_emb)
                entity_vectors_next_iter.append(vector)
                if i == (self.n_hop-1):
                    if self.training:
                        att = self.aggregators[i].att.detach()
                        att = att / (batch_size * att.size(1))
                        self.reg(self.score_reg(vector.unsqueeze(-2), entity_vectors[hop+1].view(batch_size, -1, self.n_neighbor, self.emb_size)), weights=att)
            entity_vectors = entity_vectors_next_iter
        res = entity_vectors[0].view(batch_size, self.emb_size)
        return res

    def forward(self, node, relation):
        entities, nb_relation = self.get_neighbors(node)
        node_emb = self.aggregate(relation, entities, nb_relation)
        if self.norm:
            node_emb = F.normalize(node_emb, dim=-1)
        return node_emb

    def constraint(self):
        if hasattr(self.projection, 'constraint'):
            self.projection.constraint()

