from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

import config
import random
from data_utils.label import EdgeLabel
from data_utils.hin_reader import read_hin
from trainer import Trainer


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('-d', '--dataset', required=True,
                        help='Input graph file')
    parser.add_argument('-m', '--mode', default='TransH1',
                        help='Mode (variant) of PGRA',
                        choices=['DistMult', 'TransH1', 'TransH2'])
    parser.add_argument('-l', '--l-reg', default=1e-2, type=float,
                        help='lambda of neighbor regularizer')
    parser.add_argument('--emb-size', default=128, type=int)
    parser.add_argument('--n-nb', default=20, type=int,
                        help='number of sampled neighbors')
    parser.add_argument('--degree', default=2, type=int)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--n-neg', default=5, type=int,
                        help='nagative samples')
    parser.add_argument('--max-steps', default=50000, type=int)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--train-ratio', default=0.6, type=float)
    parser.add_argument('--gpu-device', default=0, type=int)
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of parallel processes for batch sampler.')
    parser.add_argument('--seed', default=config.seed, type=int)

    args = parser.parse_args()
    return args


def main(args):
    config.seed = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu_device >= 0:
        import torch
        torch.cuda.set_device(args.gpu_device)
    print("Reading...")
    g = read_hin(args.dataset)
    el = EdgeLabel(g)
    el.split(n_val=0.1, n_test=0.9-args.train_ratio, seed=args.seed)

    score = None
    nb_loss = None
    project = None
    if args.mode == 'DistMult':
        score = 'inner'
        nb_loss = 'l2'
        project = 'distmult'
    elif args.mode == 'TransH1':
        score = 'l1'
        nb_loss = 'l1'
        project = 'transh'
    elif args.mode == 'TransH2':
        score = 'l2'
        nb_loss = 'l2'
        project = 'transh'

    t = Trainer(g, el, batch_size=args.batch_size, n_neighbor=args.n_nb, self_loop=True, score=score, project=project, nb_loss=nb_loss, l_reg=args.l_reg, degree=args.degree)
    t.run(lr=1e-4, patience=args.patience, max_steps=args.max_steps, cuda=True if args.gpu_device >= 0 else False)


if __name__ == "__main__":
    main(parse_args())
