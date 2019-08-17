from data_utils import graph
import config


def read_hin(ds='DBLP', verbose=1):

    def get_types(f_name):
        return f_name.split('_')

    suffix = 'dat'
    g = graph.Graph()
    g.name = ds
    for f in sorted((config.hin_dir / ds).glob('*.' + suffix)):
        f_types = get_types(f.stem)
        if 2 <= len(f_types) <= 3:
            if verbose:
                print('read:', f.name)
        else:
            print('skip:', f.name)
        g.read_edgelist(f, types=f_types[:2], edge_type=f_types[2] if len(f_types) == 3 else None)
    g.sort()
    return g
