from model.pgra import PGRA
from model.link_predictor import LinkPredictor
from data_utils.data_gen import LinkGenerator, init_seed_fn
from torch.utils.data import DataLoader
from time import perf_counter
import torch
import numpy as np
import config
from model.modules.regularizer import Regularizer
import tempfile
from collections import Counter
from tqdm import tqdm
import os

from model.tracker import LossTracker, MultiClsTracker


class Trainer:
    _default_optim = 'Adam'
    _default_optim_params = {'lr': 1e-4}

    def __init__(self, graph, edge_label, emb_size=128, degree=2, n_neighbor=20, batch_size=100, n_neg=5, num_workers=4,
                 score='inner', self_loop=True, test_batch_size=None, **kwargs):
        super().__init__()
        self.dataset_name = graph.name
        self.graph = graph
        self.edge_label = edge_label
        self.n_edge_types = len(edge_label.edge_types)
        self.emb_size = emb_size
        self._cuda = False
        subG = self.graph.G.edge_subgraph(self.edge_label.train_edges)
        self.batch_size = batch_size
        self.self_loop = self_loop
        if self_loop:
            subG = subG.copy()
            for node in subG.nodes:
                subG.add_edge(node, node, key=self.n_edge_types, label=self.n_edge_types)
        self.subG = subG
        self.n_neighbor = n_neighbor
        adj_node, adj_rela = self.create_node_neighbors()
        self.model = PGRA(graph.node_size, self.n_edge_types + (1 if self_loop else 0), emb_size, n_hop=degree,
                          n_neighbor=n_neighbor, **kwargs)
        self.model.register_neighbors(adj_node, adj_rela)
        self.model.reset()

        self.lp_model = LinkPredictor(score=score)

        link_gen_train = LinkGenerator(edge_label, graph.nodes_of_types, n_neg=n_neg)
        nt2id = dict(link_gen_train.node_type_to_id)
        link_gen_val = LinkGenerator(edge_label, graph.nodes_of_types, n_neg=10, mode=1, node_type_to_id=nt2id)
        link_gen_test = LinkGenerator(edge_label, graph.nodes_of_types, n_neg=10, mode=2, node_type_to_id=nt2id)

        self.data_loader_val = None
        self.data_loader_test = None
        lg_collate = LinkGenerator.collate_fn
        self.data_loader_train = DataLoader(link_gen_train, batch_size=batch_size, num_workers=num_workers,
                                            shuffle=True, collate_fn=lg_collate, worker_init_fn=init_seed_fn())
        self.data_loader_train_iter = iter(self.data_loader_train)
        if test_batch_size is None:
            test_batch_size = batch_size // 4
        if len(link_gen_val) > 0:
            self.data_loader_val = DataLoader(link_gen_val, batch_size=test_batch_size, num_workers=num_workers,
                                              shuffle=True, collate_fn=lg_collate, worker_init_fn=init_seed_fn())
        if len(link_gen_test) > 0:
            self.data_loader_test = DataLoader(link_gen_test, batch_size=test_batch_size, num_workers=num_workers,
                                               collate_fn=lg_collate, worker_init_fn=init_seed_fn())
        self.used_keys = ['r', 'h', 't', 'h_neg', 't_neg']

        self.optim = self._default_optim
        self.optim_params = self._default_optim_params
        self.optimizer = None
        self.total_epoch = 0
        self.total_steps = 0
        self.pre_name = 'PGRA'
        self.temp_dir = tempfile.mkstemp()[1]
        self.best_iter = None
        self.best_scores = None
        self.test_scores = None

    def create_node_neighbors(self):
        print('create node neighbors')
        adj_entity = np.zeros([self.graph.node_size, self.n_neighbor], dtype=np.int64)
        adj_relation = np.zeros([self.graph.node_size, self.n_neighbor], dtype=np.int64)
        for i in self.subG.nodes:
            nbs = list(self.subG.edges(i, data='label'))

            edge_type_ct = Counter()
            for x in nbs:
                edge_type_ct[x[2]] += 1

            sampled = np.random.choice(len(nbs), size=self.n_neighbor, replace=True)
            adj_entity[i] = [nbs[x][1] for x in sampled]
            adj_relation[i] = [nbs[x][2] for x in sampled]
        return adj_entity, adj_relation

    def run(self, lr=0.0001, weight_decay=0, max_steps=50000, cuda=True, patience=10, metric='mrr', optim='Adam',
            save=True):
        if cuda:
            self.cuda()
        self.create_optimizer(optim, lr=lr, weight_decay=weight_decay)
        self.train(max_steps=max_steps, patience=patience, metric=metric, save=save)

    def train(self, max_steps=50000, patience=10, metric='mrr', save=True, eval_step=None):
        if self.optimizer is None:
            self.create_optimizer()
        min_loss = float('inf')
        t0 = perf_counter()
        if eval_step is None:
            eval_step = len(self.data_loader_train)
            if eval_step > 500:
                eval_step = 500
        patience *= eval_step
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2, verbose=True, cooldown=20,
                                                               threshold=1e-3)

        best_step = 0
        self.best_scores = None
        print('Training')
        for step in range(0, max_steps, eval_step):
            t = perf_counter()
            losses = self.train_one_epoch(steps=eval_step)
            # scores = self._eval(self.data_loader_train, ('mrr',))
            # print('Train:', scores)
            is_best = False
            if self.data_loader_val is not None:
                scores = self.eval(val=True)
                print("Epoch:", '%04d' % (self.total_epoch + 1), "Step:", step, "train_loss=", losses,
                      "Val:", scores, "time=", "{:.5f}".format(perf_counter() - t))
                if self.best_scores is None or scores[metric] > self.best_scores[metric]:
                    self.best_scores, best_step = scores, step
                    self.best_iter = best_step
                    is_best = True
            else:
                loss = losses.value
                if min_loss > loss:
                    min_loss, best_step = loss, step
                    self.best_iter = step
                    is_best = True
            if is_best and save:
                self._save(self.temp_dir)
            if step - best_step > patience:
                print("Early stopping...")
                break
            reduce_lr.step(losses.value)
            if reduce_lr.cooldown_counter == reduce_lr.cooldown:
                reduce_lr.best = reduce_lr.mode_worse

        train_time = perf_counter() - t0
        print("Train time: {:.4f}s".format(train_time, ))
        if self.total_steps > 0 and save:
            print("Optimization Finished!")
            print('Load model')
            self._load(self.temp_dir)
            self.save()
            if self.data_loader_test is not None:
                if self.data_loader_val is not None:
                    val_scores = self.eval(val=True)
                    print("Val:", val_scores)
                self.test_scores = self.eval(val=False)
                print("Test:", self.test_scores)

    def cuda(self):
        self._cuda = True
        self.model.cuda()

    def create_optimizer(self, optim=None, **kwargs):
        if optim is not None:
            self.optim = optim
        self.optim_params.update(kwargs)
        self.optimizer = getattr(torch.optim, self.optim)(
            list(self.model.parameters()), **self.optim_params)

    def save(self):
        self._save(self.get_model_path())

    def _save(self, path):
        state = {
            'model': self.model.state_dict(),
            'best_iter': self.best_iter
        }
        torch.save(state, path)

    def load(self):
        self._load(self.get_model_path(build_path=False))

    def _load(self, path):
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model'], strict=False)
        self.best_iter = state['best_iter']
        if self._cuda:
            self.cuda()

    def get_model_path(self, build_path=True):
        folder = config.model_dir / self.dataset_name
        if build_path:
            folder.mkdir(parents=True, exist_ok=True)
        path_model = folder / self._join(self.pre_name, self.emb_size, self.edge_label.code)
        return path_model

    @staticmethod
    def _join(*args, sep='_'):
        return sep.join([str(arg) for arg in args])

    def get_train_data(self):
        try:
            data = next(self.data_loader_train_iter)
        except StopIteration:
            self.data_loader_train_iter = iter(self.data_loader_train)
            self.total_epoch += 1
            data = next(self.data_loader_train_iter)
        return data

    def train_one_epoch(self, steps=None):
        if steps is None:
            steps = len(self.data_loader_train)
        pbar = tqdm(range(steps), disable=os.environ.get("DISABLE_TQDM", False))
        losses = LossTracker()
        self.model.train()
        self.lp_model.train()
        t = 0
        for step in pbar:
            _t = perf_counter()
            all_data = self.get_train_data()
            if self._cuda:
                all_data = {k: v.cuda() for k, v in all_data.items()}
            data = [all_data[x] for x in self.used_keys]

            nodes_vec = [self.get_embedding(x, data[0]) for i, x in enumerate(data[1:])]
            train_losses = self.lp_model(data[0], *nodes_vec)
            if not isinstance(train_losses, tuple) and not isinstance(train_losses, list):
                train_losses = [train_losses]
            reg_losses = [m.loss for m in self.model.modules() if isinstance(m, Regularizer) and m.loss is not None]
            train_losses = train_losses + reg_losses
            losses.update(train_losses)
            sum(train_losses).backward()
            self.optimizer.step()
            for m in self.model.modules():
                if isinstance(m, Regularizer):
                    m.loss = None
            self.model.constraint()
            self.total_steps += 1

            t += perf_counter() - _t
            pbar.set_description(
                'epoch {} loss:{} time:{:.2f}'.format(self.total_epoch, losses, t * len(pbar) / (step + 1)))
        return losses

    def eval(self, val=True, metrics=('mrr',)):
        data_loader = self.data_loader_val if val else self.data_loader_test
        return self._eval(data_loader, metrics, n=60000 if val else -1)

    def _eval(self, data_loader, metrics, n=-1):
        self.model.eval()
        self.lp_model.eval()
        lp_tracker = MultiClsTracker(metrics, self.n_edge_types)
        with torch.no_grad():
            for all_data in data_loader:
                data = [all_data[x] for x in self.used_keys]
                if self._cuda:
                    data = [x.cuda() for x in data]
                r, p1, p2, n1, n2 = data
                pos, neg = (p1, p2), (n1, n2)
                for i in (0, 1):
                    feat1 = self.get_embedding(torch.cat((pos[i].unsqueeze(-1), neg[i]), dim=-1), r)
                    feat2 = self.get_embedding(pos[1 - i], r)
                    feat_args = (feat1, feat2) if i == 0 else (feat2, feat1)
                    ratings = self.lp_model.predict(r, *feat_args)
                    lp_tracker.update(ratings, r)
                if 0 < n < len(lp_tracker):
                    break
        print(lp_tracker.macro)
        lp_tracker.summarize()
        return lp_tracker

    def get_embedding(self, node_idx, target_idx):
        old_size = node_idx.size()
        if target_idx is not None:
            target_idx = target_idx.unsqueeze(1) if len(target_idx.size()) != len(node_idx.size()) else target_idx
            target_idx = target_idx.expand_as(node_idx).contiguous().view(-1)
        node_idx = node_idx.contiguous().view(-1)
        return self.model(node_idx, target_idx).view(*old_size, -1)
