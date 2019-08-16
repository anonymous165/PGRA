import random
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np
import config


class LinkGenerator:

    _non_type = '$NonType$'

    # mode = 0: train, 1: val, 2: test, 3: full
    def __init__(self, edge_label, nodes_of_types=None, mode=0, n_neg=5, node_type_to_id=None):
        _convert_mode = {0: 'train', 1: 'train_val', 2: 'all'}
        edges_mode = _convert_mode[mode]
        if mode == 1:
            self.edges = edge_label.val_samples
            self.edges_type = edge_label.val_labels
        elif mode == 2:
            self.edges = edge_label.test_samples
            self.edges_type = edge_label.test_labels
        else:
            if len(edge_label.train_samples) == 0:
                print('using whole network')
                self.edges = edge_label.samples
                self.edges_type = edge_label.labels
                edges_mode = 'all'
            else:
                self.edges = edge_label.train_samples
                self.edges_type = edge_label.train_labels
        self.edges_of_types = edge_label.edges_of_types(edges_mode)
        self.edges_of_types = {edge_type: {str(n1) + ',' + str(n2) for (n1, n2, _) in edges} for edge_type, edges in self.edges_of_types.items()}

        if nodes_of_types is not None:
            nodes_of_types = {node_type: list(set(nodes).intersection(edge_label.train_nodes)) for node_type, nodes in nodes_of_types.items()}
            self.type_of_nodes = {}
            for node_type, nodes in nodes_of_types.items():
                for node in nodes:
                    self.type_of_nodes[node] = node_type
            self.node_type_to_id = node_type_to_id
            if self.node_type_to_id is None:
                self.node_type_to_id = {node_type: i for i, node_type in enumerate(sorted(nodes_of_types))}
        else:
            self.type_of_nodes = None
            self.node_type_to_id = None

        self.nodes_of_types = nodes_of_types

        for nodes in self.nodes_of_types.values():
            random.shuffle(nodes)
        self.iter_nodes = {node_type: 0 for node_type in self.nodes_of_types}
        self.n_neg = n_neg

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        p_edge = self.edges[index]
        p_edge_type = self.edges_type[index]
        n_nodes = [[], []]
        p_type_edges = self.edges_of_types[p_edge_type]
        for i in (0, 1):
            node_type = self.type_of_nodes[p_edge[i]]
            while len(n_nodes[i]) != self.n_neg:
                node = self._get_random_node(node_type)
                n_edge = str(p_edge[0]) + ',' + str(node) if i == 1 else str(node) + ',' + str(p_edge[1])
                if n_edge in p_type_edges:
                    continue
                n_nodes[i].append(node)
        ret = {'r': p_edge_type,
               'h': p_edge[0],
               't': p_edge[1],
               }
        if self.n_neg > 0:
            ret['h_neg'] = n_nodes[0]
            ret['t_neg'] = n_nodes[1]
        if self.type_of_nodes is not None:
            ret['h_type'] = self.node_type_to_id[self.type_of_nodes[p_edge[0]]]
            ret['t_type'] = self.node_type_to_id[self.type_of_nodes[p_edge[1]]]
        return ret

    def _get_random_node(self, node_type=None):
        if node_type is None:
            node_type = LinkGenerator._non_type
        _iter = self.iter_nodes[node_type]
        if _iter == len(self.nodes_of_types[node_type]):
            random.shuffle(self.nodes_of_types[node_type])
            _iter = 0
            self.iter_nodes[node_type] = 0
        node = self.nodes_of_types[node_type][_iter]
        self.iter_nodes[node_type] += 1
        return node

    @staticmethod
    def collate_fn(batch):
        batch = default_collate(batch)
        for key, tensor in batch.items():
            if isinstance(tensor, list) or isinstance(tensor, tuple):
                batch[key] = torch.stack([x.long() for x in tensor]).transpose(0, 1)
        return batch


def init_seed_fn(seed=config.seed, use_worker_id=False):
    def init_fn(worker_id):
        worker_seed = seed + worker_id if use_worker_id else seed
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    return init_fn
