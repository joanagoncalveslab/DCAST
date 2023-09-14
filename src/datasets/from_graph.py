import os
import sys
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower()=='diversepsuedolabeling':
        project_path = '/'.join(path2this[:i+1])
sys.path.insert(0,project_path)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import collections, json
from src import config
#import networkx as nx
import argparse
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser(description='Generate Sequence Features')
parser.add_argument('--graph_loc', '-gl', metavar='the-loc-of-graph', dest='graph_loc', type=str, help='Choose graph file', default='graphs/STRING/9606.protein.links.full.v11.0.txt')
parser.add_argument('--emb_loc', '-el', metavar='the-seqvec-emb-loc', dest='emb_loc', type=str, help='Choose seqvec embeddings', default='graphs/STRING/9606.protein.sequences.v11.0.fa.npz')
parser.add_argument('--ppi_cutoff', '-pc', metavar='the-ppi-cutoff', dest='ppi_cutoff', type=float, help='Choose the cutoff for ppi', default=0.9)
parser.add_argument('--pca_size', '-ps', metavar='the-pca-size', dest='pca_size', type=int, help='Choose the components for PCA', default=64)

args = parser.parse_args()
print(f'Running args:{args}')


def get_PPI_data(in_file='graphs/STRING/9606.protein.links.full.v11.0.txt', cutoff=0.899, source='e'):
    data_loc = config.DATA_DIR / in_file
    df = pd.read_csv(data_loc, sep=" ")
    if 'e' in source and 'c' in source:
        cutoff_df = df[(df['experiments'] > cutoff * 1000) | (df['database'] > cutoff * 1000)]
    elif 'e' in source:
        cutoff_df = df[df['experiments'] > cutoff * 1000]
    elif 'c' in source:
        cutoff_df = df[df['database'] > cutoff * 1000]

    unique_genes = np.unique(np.concatenate([cutoff_df['protein1'].values, cutoff_df['protein2'].values]))

    return cutoff_df, unique_genes

def load_embeddings(emb_loc="embeddings.npz"):
    emb_loc = config.DATA_DIR / emb_loc
    if str(emb_loc).split('.')[-1] == "npz":
        data = np.load(emb_loc) #type: Dict[str, np.ndarray]
        data = dict(data)
    if str(emb_loc).split('.')[-1] == "pkl":
        data = config.load_pickle(emb_loc) #type: #Dict[str, np.ndarray]
    elif str(emb_loc).split('.')[-1] == "npy":
        embs = np.load("embeddings.npy") # shape=(n_proteins,)
        with open("embeddings.json") as fp:
            labels = json.load(fp)
        data = dict(zip(labels, embs))
    data = collections.OrderedDict(sorted(data.items()))
    return data

def pca_embeddings(emb_dict, n_components=None, random_state=124):
    key, values = list(emb_dict.keys()), np.array(list(emb_dict.values()))
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_values = pca.fit_transform(values)
    reduced_dict = collections.OrderedDict(zip(key,reduced_values))

    return reduced_dict


def create_samples(unique_genes):
    unique_genes = np.sort(unique_genes)
    unique_gene_size = len(unique_genes)
    res = []
    for i in range(unique_gene_size-1):
        for j in range(i+1, unique_gene_size):
            res.append([unique_genes[i], unique_genes[j]])
    return np.array(res)

def featurize_and_label_samples(samples, reduced_emb, df):
    res = []
    for val in samples:
        res.append([val[0], val[1]] + list(np.absolute(reduced_emb[val[0]] - reduced_emb[val[1]])))
    cols = [f'emb{i}' for i in range(len(res[0]) - 2)]
    res_df = pd.DataFrame(res, columns=['protein1', 'protein2'] + cols)
    res_df['class'] = 0
    pos_labels = [(min(prots[0], prots[1]), max(prots[0], prots[1])) for prots in df[['protein1', 'protein2']].values]
    res_df = res_df.set_index(['protein1', 'protein2'])
    res_df.loc[pos_labels,'class'] = 1
    res_df = res_df.reset_index()
    return res_df

def get_ppi_samples():
    df, unique_genes = get_PPI_data(args.graph_loc, cutoff=args.ppi_cutoff)
    samples = create_samples(unique_genes)
    end_data_loc = config.DATA_DIR / f'{args.graph_loc}_{args.pca_size}_{args.ppi_cutoff}_features.pkl'
    pca_loc = config.DATA_DIR / f'{args.emb_loc}_{args.pca_size}.pkl'
    if os.path.exists(end_data_loc):
        return pd.read_pickle(end_data_loc)

    if os.path.exists(pca_loc):
        reduced_emb = config.load_pickle(pca_loc)
    else:
        embs_dict = load_embeddings(args.emb_loc)
        reduced_emb = pca_embeddings(embs_dict, args.pca_size, random_state=124)
        config.save_pickle(pca_loc, reduced_emb)

    resulting_df = featurize_and_label_samples(samples, reduced_emb, df)
    resulting_df.to_pickle(end_data_loc)
    return resulting_df

    print()

def main():
    out = get_ppi_samples()
    #print(out)

'''Homo sapiens - 9606.protein.links.detailed.v11.0
Saccharomyces cerevisiae - 4932.protein.links.detailed.v11.0
Mus musculus - 10090.protein.physical.links.v11.0
Rattus norvegicus - 10116.protein.physical.links.v11.0
Sus scrofa - 9823.protein.physical.links.v11.0'''
if __name__ == '__main__':
    main()
