import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pgra_function.proj import TransH, DistMult
from model.pgra_function.rsim import get_rsim
from model.weight_init import weight_init, manual_init
from model.modules.regularizer import NeighborRegularizer
from model.modules.aggregator import Aggregator

import numpy as np


def normalize(tensor):
    node_emb_norm = torch.norm(tensor, dim=-1, keepdim=True)
    return torch.where(node_emb_norm > 1, F.normalize(tensor, dim=-1), tensor)


class PGRA(nn.Module):

    def __init__(self, n_node, n_relation, emb_size, n_hop, n_neighbor, proj_name, norm=False, rsim_name='abs_cos', agg_name='neighbor', p_reg=2, l_reg=0.):
        super().__init__()
        self.node_emb = nn.Embedding(n_node, emb_size)
        self.rela_emb = nn.Embedding(n_relation, emb_size)
        self.emb_size = emb_size
        self.n_hop = n_hop
        self.n_neighbor = n_neighbor
        self.norm = norm

        RSIM = get_rsim(rsim_name)

        proj_name = proj_name.lower()
        if proj_name == 'transh':
            self.projection = TransH(n_relation, emb_size, self.rela_emb)
        elif proj_name == 'distmult':
            self.projection = DistMult(n_relation, emb_size, self.rela_emb)
        else:
            raise Exception()

        self.aggregators = nn.ModuleList([
            Aggregator(agg_name=agg_name, emb_size=emb_size, rsim=RSIM) for _ in range(n_hop)
        ])

        adj_node = np.zeros((n_node, n_neighbor))
        adj_rela = np.zeros((n_node, n_neighbor))
        self.register_neighbors(adj_node, adj_rela)

        self.neighbor_regularizer = NeighborRegularizer(p=p_reg, l_reg=l_reg)

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
                        att = self.aggregators[i].att_score.detach()
                        att = att / (batch_size * att.size(1))
                        self.neighbor_regularizer(vector.unsqueeze(-2), entity_vectors[hop+1].view(batch_size, -1, self.n_neighbor, self.emb_size), weights=att)
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

