from data_utils import graph
import config

map_type = {
    'author': 'a',
    'paper': 'p',
    'conference': 'c',
    'conf': 'c',
    'term': 't',
    'ref': 'r',
    'reference': 'r',

    'business': 'b',
    'location': 'l',
    'user': 'u',
    'category': 'c',

    'movie': 'm',
    'group': 'g',
    'director': 'd',
    'actor': 'a',
    'type': 't',

    'book': 'b',
    'publisher': 'p',
    'year': 'y',
}


def read_hin(ds='DBLP', verbose=1):

    def get_types(f_name):
        return f_name.split('_')

    suffix = 'dat'
    g = graph.Graph()
    g.name = ds
    for f in sorted((config.hin_dir / ds).glob('*.' + suffix)):
        if verbose:
            print(f.stem)
        f_types = get_types(f.stem)
        if len(f_types) != 2:
            continue
        if f_types[0] in map_type and f_types[1] in map_type:
            g.read_edgelist(f, types=list(map(lambda x: map_type[x], f_types)))
            if verbose:
                print(f_types)
    g.sort()
    return g
