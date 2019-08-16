import networkx as nx
import numpy as np
import functools
from collections import OrderedDict
import copy


class Graph(object):

    _default_type = ''

    def __init__(self):
        self.G = None
        self.node_size = 0
        self.label_size = 0
        self.edge_type_size = 0
        self.node_to_id = {}
        self.label_to_id = {}
        self.name = 'unnamed'
        self._edge_type_to_id = {}
        self.directed_edge_types = set()

    @property
    def edge_type_to_id(self):
        return dict(self._edge_type_to_id[''])

    @property
    def nodes_of_types(self):
        return {node_type: list(mapper.values()) for node_type, mapper in self.node_to_id.items()}

    @property
    def type_of_nodes(self):
        type_of_nodes = {}
        for node_type, nodes in self.nodes_of_types.items():
            for node in nodes:
                type_of_nodes[node] = node_type
        return type_of_nodes

    def convert_to_id(self, old_id, id_type=None, add=False, mapper=None, counter=None):
        assert mapper is not None
        assert counter is not None
        mapper = getattr(self, mapper)
        if id_type is None:
            id_type = self._default_type
        if add and id_type not in mapper:
            mapper[id_type] = {}
        if add and old_id not in mapper[id_type]:
            i = getattr(self, counter)
            mapper[id_type][old_id] = i
            setattr(self, counter, i+1)
        return mapper[id_type][old_id]

    node2id = functools.partialmethod(convert_to_id, mapper='node_to_id', counter='node_size')
    label2id = functools.partialmethod(convert_to_id, mapper='label_to_id', counter='label_size')
    et2id = functools.partialmethod(convert_to_id, mapper='_edge_type_to_id', counter='edge_type_size')

    def read_adjlist(self, filename, types=None):

        if types is None:
            types = (self._default_type, self._default_type)
        if self.G is None:
            self.G = nx.DiGraph()
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            data = l.split()
            src, dst = data[0], data[1]
            for dst in data[1:]:
                self.add_edge(src, dst, 1., directed=True, types=types)
        fin.close()
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0

    @staticmethod
    def name_edge_label(types):
        return ''.join(types)

    def add_edge(self, src, dst, weight=1., directed=False, types=None, edge_type=None, **edge_attr):
        if types is None:
            types = (self._default_type, self._default_type)
        src = self.node2id(src, id_type=types[0], add=True)
        dst = self.node2id(dst, id_type=types[1], add=True)
        if src not in self.G.nodes:
            self.G.add_node(src, node_type=types[0])
        if dst not in self.G.nodes:
            self.G.add_node(dst, node_type=types[1])

        _edge_type = self.name_edge_label(types) if edge_type is None else edge_type
        _edge_id = self.et2id(_edge_type, add=True)
        _edge_attr = copy.copy(edge_attr)
        _edge_attr.update({'label': _edge_id, 'edge_type': _edge_type, 'weight': weight})
        self.G.add_edge(src, dst, key=_edge_id, **_edge_attr)
        if not directed:
            _edge_type = self.name_edge_label(reversed(types)) if edge_type is None else edge_type
            _edge_attr = copy.copy(edge_attr)
            _edge_attr.update({'label': _edge_id, 'edge_type': _edge_type, 'weight': weight})
            self.G.add_edge(dst, src, key=_edge_id, **_edge_attr)
            if edge_type is None:
                self._edge_type_to_id[''][_edge_type] = _edge_id
        else:
            self.directed_edge_types.add(_edge_id)

    def read_edgelist(self, filename, directed=False, types=None):
        if types is None:
            types = (self._default_type, self._default_type)
        if self.G is None:
            self.G = nx.MultiDiGraph()
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            data = l.split()
            src, dst = data[0], data[1]
            w = 1.
            if len(data) == 3:
                w = float(data[2])
            self.add_edge(src, dst, w, directed, types)
        fin.close()

    def read_node_label(self, filename, node_type=None):
        fin = open(filename, 'r')
        if node_type is None:
            node_type = self._default_type
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            v = self.node2id(vec[0], id_type=node_type)
            self.G.nodes[v]['label'] = list(map(lambda x: self.label2id(x, id_type=node_type, add=True), vec[1:]))
        fin.close()

    def read_node_features(self, filename, node_type=None):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            v = self.node2id(vec[0], id_type=node_type)
            self.G.nodes[v]['feature'] = np.array(
                [float(x) for x in vec[1:]])
        fin.close()

    def get_nodes_features(self):
        return np.vstack([self.G.nodes[i]['feature'] for i in range(self.node_size)])

    def sort(self):
        n_node = 0
        remap = {}
        for node_type in sorted(self.node_to_id):
            mapper = self.node_to_id[node_type]
            new_map = {}
            for old_id in sorted(mapper):
                remap[mapper[old_id]] = n_node
                new_map[old_id] = n_node
                n_node += 1
            self.node_to_id[node_type] = new_map
        self.G = nx.relabel_nodes(self.G, remap)
        return remap

    @property
    def noftypes(self):
        return OrderedDict((node_type, len(self.node_to_id[node_type])) for node_type in sorted(self.node_to_id.keys()))
