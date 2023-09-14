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

parser = argparse.ArgumentParser(description='Create Datasets for Networks')
parser.add_argument('--network', '-n', metavar='the-network', dest='network', type=str, help='Choose network', default='cora')
parser.add_argument('--class', '-c', metavar='the-class', dest='class_name', type=str, help='Choose class', default='max')
args = parser.parse_args()

def get_edges_features_out_loc(name='cora', chosen_class='none'):
    if name in ['cora', 'citeseer']:
        edge_loc = config.DATA_DIR / 'graphs' / 'linqs' / name / f'{name}.cites'
        feature_loc = config.DATA_DIR / 'graphs' / 'linqs' / name / f'{name}.content'
        out_folder = config.DATA_DIR / 'graphs' / 'linqs' / name
        edges = pd.read_csv(edge_loc, sep='\t', header=None)
    elif name in ['cornell', 'texas', 'washington', 'wisconsin']:
        edge_loc = config.DATA_DIR / 'graphs' / 'linqs' / 'webkb' / f'{name}.cites'
        feature_loc = config.DATA_DIR / 'graphs' / 'linqs' / 'webkb' / f'{name}.content'
        out_folder = config.DATA_DIR / 'graphs' / 'linqs' / 'webkb'
        edges = pd.read_csv(edge_loc, sep=' ', header=None)
    features = pd.read_csv(feature_loc, sep='\t', header=None)
    return edges.astype(str).values, features.values, out_folder

def create_samples(unique_entities):
    unique_entities = np.sort(unique_entities)
    unique_entities_size = len(unique_entities)
    res = []
    for i in range(unique_entities_size-1):
        for j in range(i+1, unique_entities_size):
            res.append([unique_entities[i], unique_entities[j]])
    return np.array(res)

def featurize_and_label_samples(samples, embs, edges):
    res = []
    edge_dict = {}
    for key in embs.keys():
        edge_dict[key] = []
    for edge0, edge1 in edges.tolist():
        if edge0 in edge_dict.keys():
            edge_dict[edge0].append(edge1)
        if edge1 in edge_dict.keys():
            edge_dict[edge1].append(edge0)
    for val0, val1 in samples:
        if val1 in edge_dict[val0]:#False:#[val0,val1] in edges.tolist() or [val1,val0] in edges.tolist():
            res.append([val0, val1] + list(np.bitwise_xor(embs[val0],embs[val1])*-1+np.bitwise_and(embs[val0],embs[val1]))+[1])
        else:
            res.append([val0, val1] + list(np.bitwise_xor(embs[val0],embs[val1])*-1+np.bitwise_and(embs[val0],embs[val1]))+[0])
    cols = [f'emb{i}' for i in range(len(res[0]) - 3)]
    res_df = pd.DataFrame(res, columns=['ent1', 'ent2'] + cols+['class'])
    return res_df

def process_citation(name='cora', class_name='max'):
    edges, features, out_folder = get_edges_features_out_loc(name)
    classes = features[:,-1]
    if class_name.lower()=='max':
        unique_classes, class_sizes = np.unique(classes, return_counts=True)
        chosen_class = unique_classes[np.argmax(class_sizes)]
        entities = features[classes==chosen_class,0].astype(str)
        feature_vals = features[classes==chosen_class,1:-1].astype(int)
    elif class_name.lower()=='none':
        chosen_class='all'
        entities = features[:,0].astype(str)
        feature_vals = features[:,1:-1].astype(int)
    else:
        chosen_class = class_name
        entities = features[classes==chosen_class,0].astype(str)
        feature_vals = features[classes==chosen_class,1:-1].astype(int)
    emb_dict = dict(zip(entities, feature_vals))

    samples = create_samples(entities)
    division=10
    sample_size = len(samples)
    chunk_size=int(len(samples)/division)
    res_df=None
    for div in range(division):
        res_df=None 
        print(f'{div} is started from {chunk_size*div} to {chunk_size*(div+1)}')
        if div==division-1:
            res_df = featurize_and_label_samples(samples[chunk_size*div:,:], emb_dict, edges)
            end_data_loc = out_folder / f'{name}_{chosen_class}_{div}_features.pkl.gz'
            res_df.to_pickle(end_data_loc)
            
        else:
            res_df = featurize_and_label_samples(samples[chunk_size*div:chunk_size*(div+1),:], emb_dict, edges)
            end_data_loc = out_folder / f'{name}_{chosen_class}_{div}_features.pkl.gz'
            res_df.to_pickle(end_data_loc)
    #res_df = featurize_and_label_samples(samples, emb_dict, edges)
    #end_data_loc = out_folder / f'{name}_{chosen_class}_features.pkl.gz'
    #res_df.to_pickle(end_data_loc)

process_citation(args.network, args.class_name)
