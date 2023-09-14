import os
import sys
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower() == 'diversepsuedolabeling':
        project_path = '/'.join(path2this[:i + 1])
sys.path.insert(0, project_path)
from src import load_dataset as ld
from src import bias_techniques as bt
from src import config
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import ks_2samp, mannwhitneyu
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['axes.linewidth'] = 0.1

import math
import warnings
warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser(description='RFE arguments')
parser.add_argument('--bias', '-b', metavar='the-bias', dest='bias', type=str, help='Choose bias', default='hierarchyy9')
parser.add_argument('--bias_size', '-bs', metavar='the-bias-size', dest='bias_size', type=int, help='Choose the bias size per class', default=30)
parser.add_argument('--task', '-t', metavar='the-task', dest='task', type=str, help='Choose the task', default='multi_dist')
parser.add_argument('--dataset_group', '-dg', metavar='the-dataset-group', dest='ds_group', type=int, help='Choose the dataset group', default=0)

args = parser.parse_args()
print(f'Running args:{args}')
if 'umap' in args.task:
    import umap

R_STATE = 123


def factors(n, typ='factor'):
    if typ=='factor':
        return list(set(x for tup in ([i, n//i] for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup))
    if typ=='pair':
        return [[i, n//i] for i in range(1, int(n**0.5)+1) if n % i == 0]
    if typ=='best_pair':
        all_pairs = [[i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0]
        min_dist = math.inf
        best_pair = None
        for pair in all_pairs:
            dist = np.abs(pair[1] - pair[0])
            if dist<min_dist:
                best_pair = pair
                min_dist=dist
        return [min(best_pair), max(best_pair)]


def bias_vis_multi(model_dict, vis_method='umap'):
    X, y = ld.load_dataset(model_dict['params']['dataset']['name'])
    for fold in range(model_dict['params']['dataset']['runs']):
        model_dict['models'][f'{fold}'] = {}
        p_data = config.split_dataset(X, y, model_dict['params']['dataset']['train'],
                                      model_dict['params']['dataset']['test'],
                                      model_dict['params']['dataset']['unlabeled'], r_seed=R_STATE + fold)
        X_train, y_train = p_data['x_train'], p_data['y_train']
        X_tr_umap = umap.UMAP(n_neighbors=15, min_dist=0.25).fit_transform(X_train)
        bias_size = len(model_dict['params']['biases'])
        best_facts = factors(bias_size, typ='best_pair')
        plt.clf()
        plt.rcParams["figure.figsize"] = ((bias_size / best_facts[0]) * 6, (bias_size / best_facts[1]) * 6)
        fig, ax = plt.subplots(*best_facts)
        st = fig.suptitle(f'Fold {fold}')
        total_rows = best_facts[0]
        bias_strs = []
        class_size = len(np.unique(y_train))
        for i, bias in enumerate(model_dict['params']['biases']):
            bias_params = {key: val for key, val in bias.items()
                           if ('name' not in key) and ('y' != key)}
            if 'y' in bias and bias['y']:
                selected_ids = bt.get_bias(bias['name'], X_train, y=y_train,
                                           **bias_params).astype(int)
            else:
                selected_ids = bt.get_bias(bias['name'], X=X_train, **bias_params).astype(int)
            for class_tmp in np.unique(y_train):
                selected_class = selected_ids[y_train[selected_ids] == class_tmp]
                y_color = y_train.copy()
                y_color[selected_class] = class_tmp+class_size

                # negatives = y_color == 0
                # positives = y_color == 1
                # negatives_chosen = y_color == 2
                # positives_chosen = y_color == 3
                ax_tmp = ax[i // total_rows, i % total_rows]
                ax_tmp.scatter(X_tr_umap[y_color == 0, 0], X_tr_umap[y_color == 0, 1], c="#EE99AA",
                               s=15, label='Negative')
                ax_tmp.scatter(X_tr_umap[y_color == 1, 0], X_tr_umap[y_color == 1, 1], c="#6699CC",
                               s=15, label='Positive')
                ax_tmp.scatter(X_tr_umap[y_color == 2, 0], X_tr_umap[y_color == 2, 1], c="#994455",
                               s=15, label=f'Negative_chosen')
                ax_tmp.scatter(X_tr_umap[y_color == 3, 0], X_tr_umap[y_color == 3, 1], c="#004488",
                               s=15, label=f'Positive_chosen')
                ax_tmp.set_title(f'{bias["name"]} - Pos({len(selected_positives)}/{sum(y_train == 1)} - '
                                 f'Neg({len(selected_negatives)}/{sum(y_train == 0)})', y=0)
            ax_tmp.set_xticks([])
            ax_tmp.set_xticks([])
            bias_strs.append(f'_{bias["name"]}'
                             f'({"|".join([str(val) for key, val in bias.items() if "name" not in key])})')

        plt.legend()
        fig.tight_layout()
        bias_str = '_'.join(bias_strs)
        # shift subplots down:
        st.set_y(0.98)
        fig.subplots_adjust(top=0.95)
        png_loc = config.RESULT_DIR / f'{os.path.basename(__file__)[:-3]}' / f'{model_dict["params"]["dataset"]["name"]}{bias_str}_run{fold}.png'
        config.ensure_dir(png_loc)
        # plt.show()
        plt.savefig(png_loc, type='png', dpi=300, bbox_inches='tight')


def bias_vis(model_dict, vis_method='umap'):
    X, y = ld.load_dataset(model_dict['params']['dataset']['name'])
    for fold in range(model_dict['params']['dataset']['runs']):
        model_dict['models'][f'{fold}'] = {}
        p_data = config.split_dataset(X, y, model_dict['params']['dataset']['train'],
                                      model_dict['params']['dataset']['test'],
                                      model_dict['params']['dataset']['unlabeled'], r_seed=R_STATE + fold)
        X_train, y_train = p_data['x_train'], p_data['y_train']
        X_tr_umap = umap.UMAP(n_neighbors=15, min_dist=0.25).fit_transform(X_train)
        bias_size = len(model_dict['params']['biases'])
        best_facts = factors(bias_size, typ='best_pair')
        plt.clf()
        plt.rcParams["figure.figsize"] = ((bias_size/best_facts[0])*6, (bias_size/best_facts[1])*6)
        fig, ax = plt.subplots(*best_facts)
        st = fig.suptitle(f'Fold {fold}')
        total_rows = best_facts[0]
        bias_strs = []
        for i, bias in enumerate(model_dict['params']['biases']):
            bias_params = {key: val for key, val in bias.items()
                           if ('name' not in key) and ('y' != key)}
            if 'y' in bias and bias['y']:
                selected_ids = bt.get_bias(bias['name'], X_train, y=y_train,
                                           **bias_params).astype(int)
            else:
                selected_ids = bt.get_bias(bias['name'], X=X_train, **bias_params).astype(int)
            selected_positives = selected_ids[y_train[selected_ids] == 1]
            selected_negatives = selected_ids[y_train[selected_ids] == 0]
            y_color = y_train.copy()
            y_color[selected_negatives] = 2
            y_color[selected_positives] = 3

            #negatives = y_color == 0
            #positives = y_color == 1
            #negatives_chosen = y_color == 2
            #positives_chosen = y_color == 3
            ax_tmp = ax[i//total_rows, i%total_rows]
            ax_tmp.scatter(X_tr_umap[y_color == 0, 0], X_tr_umap[y_color == 0, 1], c="#EE99AA",
                        s=15, label='Negative')
            ax_tmp.scatter(X_tr_umap[y_color == 1, 0], X_tr_umap[y_color == 1, 1], c="#6699CC",
                        s=15, label='Positive')
            ax_tmp.scatter(X_tr_umap[y_color == 2, 0], X_tr_umap[y_color == 2, 1], c="#994455",
                        s=15, label=f'Negative_chosen')
            ax_tmp.scatter(X_tr_umap[y_color == 3, 0], X_tr_umap[y_color == 3, 1], c="#004488",
                        s=15, label=f'Positive_chosen')
            ax_tmp.set_title(f'{bias["name"]} - Pos({len(selected_positives)}/{sum(y_train==1)} - '
                            f'Neg({len(selected_negatives)}/{sum(y_train==0)})', y=0)
            ax_tmp.set_xticks([])
            ax_tmp.set_xticks([])
            bias_strs.append(f'_{bias["name"]}'
                             f'({"|".join([str(val) for key, val in bias.items() if "name" not in key])})')

        plt.legend()
        fig.tight_layout()
        bias_str = '_'.join(bias_strs)
        # shift subplots down:
        st.set_y(0.98)
        fig.subplots_adjust(top=0.95)
        png_loc = config.RESULT_DIR / f'{os.path.basename(__file__)[:-3]}' / f'{model_dict["params"]["dataset"]["name"]}{bias_str}_run{fold}.png'
        config.ensure_dir(png_loc)
        #plt.show()
        plt.savefig(png_loc, type='png', dpi=300, bbox_inches='tight')


def bias_vis_final(model_dict, n_neigh, vis_method='umap'):
    X, y, X_main_test, y_main_test = ld.load_dataset(model_dict['params']['dataset']['name'], **model_dict['params']['dataset']['args'], test=True)
    class_dict = {'breast_cancer': {0: 'Benign (B)', 1: 'Tumor (T)'}, 'wine_uci2': {0: 'Red (R)', 1: 'White (W)'}, 
                  'mushroom': {0: 'Safe (S)', 1: 'Poison (P)'}, 'drug': {0: 'Resistant (R)', 1: 'Sensitive (S)'},
                 'mnist': {0: '0', 1: '1'}, 'fire': {1: 'Extinguished (E)', 0: 'Non-ext. (N)'}, 'spam': {0: 'Not-Spam (N)', 1: 'Spam (S)'},
                 'pumpkin': {0: 'Cercevelik (C)', 1: 'Urgup (U)'}, 'pistachio': {0: 'Kirmizi (K)', 1: 'Siirt (S)'}, 
                  'rice': {0: 'Cammeo (C)', 1: 'Osmancik (O)'}, 'raisin': {0: 'Kecimen (K)', 1: 'Besni (B)'},
                 'adult': {0: '<=50K', 1: '>50K'}}
    name2short = {'Benign (B)': 'B', 'Tumor (T)': 'T', 'Red (R)': 'R', 'White (W)': 'W', 'Safe (S)': 'S', 'Poison (P)': 'P', 'Resistant (R)': 'R', 'Sensitive (S)': 'S', '0': '0', '1': '1',
                 'Extinguished (E)':'E', 'Non-ext. (N)':'N', 'Spam (S)': 'S', 'Not-Spam (N)': 'N',
                 'Cercevelik (C)': 'C', 'Urgup (U)': 'U', 'Kirmizi (K)': 'K', 'Siirt (S)': 'S',
                  'Cammeo (C)': 'C', 'Osmancik (O)': 'O', 'Kecimen (K)': 'K', 'Besni (B)': 'B', '<=50K': '<=50K', '>50K': '>50K'}
    
    classid2name = class_dict[model_dict['params']['dataset']['name']]
    bias_name_fix = {'hierarchyy':"Hierarchy (0.9)", 'random': "Random", "joint": "Joint", "dirichlet": "Dirichlet"}
    ds_name_fix = {'breast_cancer':"Breast Cancer", 'wine_uci2': "Wine", "mushroom": "Mushroom", "mnist": "MNIST", "drug_CX-5461": "Drug", 'spam': 'Spam', 
               'adult': 'Adult', 'rice': 'Rice', 'fire': 'Fire', 'pumpkin': 'Pumpkin', 'pistachio': 'Pistachio', 'raisin': 'Raisin'}
    for fold in range(model_dict['params']['dataset']['runs']):
        if fold !=11:
            continue 
        model_dict['models'][f'{fold}'] = {}
        X_train, X_unk, y_train, y_unk = split_dataset(X, y, model_dict['params']['dataset']['train'],
                               model_dict['params']['dataset']['unlabeled'], r_seed=R_STATE + fold)
        class_size = len(np.unique(y_train))
        X_tr_umap = umap.UMAP(n_neighbors=n_neigh, min_dist=0.25, random_state=R_STATE).fit_transform(X_train)
        bias_size = len(model_dict['params']['biases'])
        best_facts = factors(bias_size, typ='best_pair')
        plt.clf()
        plt.rcParams["figure.figsize"] = ((bias_size/best_facts[0])*6, (bias_size/best_facts[1])*6*0.8)
        fig, ax = plt.subplots(*best_facts)
        if len(model_dict['params']['biases'])==1:
            st = fig.suptitle(f'{ds_name_fix[model_dict["params"]["dataset"]["name"]]} | {bias_name_fix[model_dict["params"]["biases"][0]["name"]]} | Run {fold}')
        else:
            st = fig.suptitle(f'{ds_name_fix[model_dict["params"]["dataset"]["name"]]} | Run {fold}')
        total_rows = best_facts[0]
        bias_strs = []
        for i, bias in enumerate(model_dict['params']['biases']):
            bias_params = {key: val for key, val in bias.items()
                           if ('name' not in key) and ('y' != key)}
            if 'y' in bias and bias['y']:
                selected_ids = bt.get_bias(bias['name'], X_train, y=y_train,
                                           **bias_params).astype(int)
            else:
                selected_ids = bt.get_bias(bias['name'], X=X_train, **bias_params).astype(int)
            classes = np.unique(y_train)
            class_converter = {classes[fake_k]:fake_k for fake_k in range(len(classes))}
            selected_negatives = selected_ids[y_train[selected_ids] == classes[0]]
            selected_positives = selected_ids[y_train[selected_ids] == classes[1]]
            y_color = y_train.copy()
            for k, v in class_converter.items(): y_color[y_train==k] = v
            y_color[selected_negatives] = 2
            y_color[selected_positives] = 3

            #negatives = y_color == 0
            #positives = y_color == 1
            #negatives_chosen = y_color == 2
            #positives_chosen = y_color == 3
            if len(model_dict['params']['biases'])==1:
                ax_tmp = ax
            else:
                ax_tmp = ax[i//total_rows, i%total_rows]
            #(0.945, 0.639, 0.251, 1.0),(0.6, 0.557, 0.765, 1.0) "#EE99AA", "#6699CC", "#994455", "#004488",
            ax_tmp.scatter(X_tr_umap[y_color == 0, 0], X_tr_umap[y_color == 0, 1], c="#EE99AA",
                        s=14, label=classid2name[0])
            ax_tmp.scatter(X_tr_umap[y_color == 1, 0], X_tr_umap[y_color == 1, 1], c="#6699CC",
                        s=14, label=classid2name[1])
            ax_tmp.scatter(X_tr_umap[y_color == 2, 0], X_tr_umap[y_color == 2, 1], c="#994455",
                        s=14, label=f'Chosen {name2short[classid2name[0]]}')
            ax_tmp.scatter(X_tr_umap[y_color == 3, 0], X_tr_umap[y_color == 3, 1], c="#004488",
                        s=14, label=f'Chosen {name2short[classid2name[1]]}')
            
            if len(model_dict['params']['biases'])==1:
                ax_tmp.set_title(f'{name2short[classid2name[1]]}({len(selected_positives)}/{sum(y_train==classes[1])}) - '
                            f'{name2short[classid2name[0]]}({len(selected_negatives)}/{sum(y_train==classes[0])})', y=0)
            else:
                ax_tmp.set_title(f'{ds_name_fix[model_dict["params"]["dataset"]["name"]]} | {name2short[classid2name[1]]}({len(selected_positives)}/{sum(y_train==classes[1])}) - '
                            f'{name2short[classid2name[0]]}({len(selected_negatives)}/{sum(y_train==classes[0])})', y=0)
            ax_tmp.set_xticks([])
            ax_tmp.set_yticks([])
            bias_strs.append(f'{bias["name"]}({"|".join([str(val) for key, val in bias.items() if "name" not in key])})')

        plt.legend(frameon=False, fontsize=14)
        fig.tight_layout()
        bias_str = '_'.join(bias_strs)
        # shift subplots down:
        st.set_y(0.98)
        fig.subplots_adjust(top=0.95)
        notype_loc = config.ROOT_DIR / f'final_bias_umaps' / f'{model_dict["params"]["dataset"]["name"]}' / f'{model_dict["params"]["dataset"]["name"]}{bias_str}_nn{n_neigh}_run{fold}'
        #png_loc = config.RESULT_DIR / f'{os.path.basename(__file__)[:-3]}' / f'{model_dict["params"]["dataset"]["name"]}{bias_str}_run{fold}.png'
        config.ensure_dir(notype_loc)
        #plt.show()
        plt.savefig(f'{notype_loc}.png', type='png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{notype_loc}.pdf', type='png', dpi=300, bbox_inches='tight')

def visualize_umap_final():
    for ds_name in ['mnist', 'breast_cancer', 'wine_uci2', 'mushroom', 'fire', 'adult', 'spam', 'rice', 'raisin', 'pumpkin', 'pistachio']:
        bias_per_class = 30
        unique_class =2
        if ds_name=='mnist':
            unique_class=10
        bias_list = []
        #bias_list.append({'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.9})
        #bias_list.append({'name': 'random', 'y': True, 'size':bias_per_class})
        #bias_list.append({'name': 'joint'})
        bias_list.append({'name': 'dirichlet', 'n': bias_per_class * unique_class})
        model_dict = {'models': {}, 'params': {}}
        model_dict['params']['biases'] = bias_list

        bias_per_class = args.bias_size
        dataset = {'name': ds_name,'args':{}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
        if 'drug_' in ds_name:
            dataset = {'name': 'drug','args':{'drug': ds_name.split('_')[1]}, 'order': 'train_test_unlabeled_bias_validation',
                                                               'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
        model_dict['params']['dataset'] = dataset
        for n_neigh in [15, 100]:#, 50, 200]:
            bias_vis_final(model_dict, n_neigh, 'umap')
        print(f'{ds_name} done')

def visualize_umap():
    selection_per_class = 25
    class_size = 10
    bias_list = []
    #bias_list.append({'name': 'cluster', 'y': True, 'k': 3, 'n_k': 1})
    bias_list.append({'name': 'hierarchy', 'y':True, 'max_size':selection_per_class, 'metric':'ward', 'prob':0.8})
    #bias_list.append({'name': 'dirichlet', 'n': selection_per_class*class_size})
    #bias_list.append({'name': 'joint'})
    model_dict = {'models': {}, 'params': {}}
    model_dict['params']['biases'] = bias_list
    model_dict['params']['dataset'] = {'name': 'mnist', 'order': 'train_test_unlabeled_bias_validation',
                                       'train': 0.35, 'test': 0.25, 'unlabeled': 0.40, 'val': 0.1, 'runs': 30}
    bias_vis(model_dict, 'umap')
    
def split_dataset(X, y, train_ratio=0.20, test_ratio=0.20, unknown_ratio=None, r_seed=123):
    x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                        test_size=1 - train_ratio, random_state=r_seed)
    if unknown_ratio is None:
        return x_train, x_test, y_train, y_test
    else:
        x_unk, x_test, y_unk, y_test = train_test_split(x_test, y_test, shuffle=True, stratify=y_test,
                                                        test_size=test_ratio / (test_ratio + unknown_ratio),
                                                        random_state=r_seed)
        return {'x_train': x_train, 'y_train': y_train,
                'x_unk': x_unk, 'y_unk': y_unk,
                'x_test': x_test, 'y_test': y_test}

def get_distances(model_dict):
    X, y, X_main_test, y_main_test = ld.load_dataset(model_dict['params']['dataset']['name'], **model_dict['params']['dataset']['args'], test=True)
    all_distances = []
    all_classes = []
    df_res = []
    for fold in range(model_dict['params']['dataset']['runs']):
        X_test, y_test = X_main_test.copy(), y_main_test.copy()
        model_dict['models'][f'{fold}'] = {}
        X_train, X_unk, y_train, y_unk = split_dataset(X, y, model_dict['params']['dataset']['train'],
                               model_dict['params']['dataset']['unlabeled'], r_seed=R_STATE + fold)
        class_size = len(np.unique(y_train))
        if model_dict['params']['bias']["name"] is not None:
            bias_params = {key: val for key, val in model_dict['params']['bias'].items()
                           if ('name' not in key) and ('y' != key)}
            if 'y' in model_dict['params']['bias'] and model_dict['params']['bias']['y']:
                selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X_train, y=y_train,
                                           **bias_params).astype(int)
            else:
                selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X=X_train, **bias_params).astype(int)
        else:
            selected_ids = np.arange(len(X_train))
        pos_dis_to_others_list = []
        class_to_others_list = []
        for selected_id in selected_ids:
            selected_class = y_train[selected_id]
            pos_dist_to_others = []
            for other_point in selected_ids:
                if selected_id != other_point and y_train[other_point] == selected_class:
                    pos_dist_to_others.append(np.linalg.norm(X_train[selected_id] - X_train[other_point]))
            pos_dis_to_others_list.append(np.sum(pos_dist_to_others) / len(pos_dist_to_others))
            class_to_others_list.append(selected_class)
            df_res.append([selected_id, selected_class, fold, np.sum(pos_dist_to_others) / len(pos_dist_to_others)])
        all_distances.append(pos_dis_to_others_list)
        all_classes.append(class_to_others_list)

    df_res = pd.DataFrame(df_res, columns=['selected_id', 'selected_class', 'fold', 'score'])
    return df_res, all_distances, all_classes

def get_selected_similarities(model_dict, out_loc):
    df_res, all_distances, all_classes = get_distances(model_dict)
    plt.clf()
    png_loc = config.RESULT_DIR / f'{model_dict["params"]["bias"]["name"]}' / \
                        f'{model_dict["params"]["dataset"]["name"]}' / 'bias_histograms' / f'{out_loc}.png'
    config.ensure_dir(png_loc)
    all_res = np.concatenate(all_distances, axis=0)
    #all_class = np.concatenate(all_classes, axis=0)
    #colors = ['blue' if clas==1 else 'red' for clas in all_class]
    #plt.hist(all_res,50, rwidth=0.8, hue='colors')
    maxr,minr = max(df_res['score'].values), min(df_res['score'].values)
    binrange_dict={'breast_cancer':[0,2500], 'wine_uci2':[0,230], 'mnist':[0,75], 'mushroom':[400, 1100], 'drug':[100, 168]}
    ranger = binrange_dict[model_dict["params"]["dataset"]["name"]]
    if model_dict["params"]["dataset"]["name"] == 'mnist':
        df_res = df_res[np.isin(df_res['selected_class'],[1,2])]
    bins = 50
    hist = sns.histplot(df_res,x='score', bins=bins, shrink=0.8, hue='selected_class', binrange = ranger, palette='colorblind')
    plt.title(f'{model_dict["params"]["dataset"]["name"]} - {model_dict["params"]["bias"]["name"]}')
    plt.ylabel(f'Counts')
    plt.xlabel(f'Distances')
    heights = [pat.get_height() for pat in hist.patches]


    new_model_dict = {'models':{}, 'params':{}}
    new_model_dict['params']['bias'] = {'name':None}
    new_model_dict['params']['dataset'] = model_dict['params']['dataset'].copy()
    df_res_none, all_distances_none, all_classes_none = get_distances(new_model_dict)
    df_res_none = df_res_none.set_index('selected_id').loc[df_res['selected_id'].values].reset_index()
    full_res = df_res.merge(df_res_none, how='left', left_on=['selected_id', 'selected_class', 'fold'],
                     right_on=['selected_id', 'selected_class', 'fold'],
                     suffixes=('_bias', '_none'))
    test_res = {}
    test_res2 = {}
    for clas_id, clas in enumerate(df_res['selected_class'].unique()):
        biased_samples = df_res[df_res['selected_class']==clas]['score'].values
        nonbiased_samples = df_res_none[df_res_none['selected_class']==clas]['score'].values
        res = ks_2samp(biased_samples, nonbiased_samples)
        res2 = mannwhitneyu(biased_samples, nonbiased_samples)
        test_res[clas] = res
        test_res2[clas] = res2
        res_txt = f'Class {clas}: {res.pvalue:.2e}'
        plt.text(plt.xlim()[1]-plt.xlim()[1]/40, plt.ylim()[1] - plt.ylim()[1] * 2 / 10 - plt.ylim()[1] * clas_id / 20 - plt.ylim()[1]/30, res_txt,
                 fontsize=12, ha='right', va='top', wrap=True)
        #max_id = np.argmax(heights[clas_id*bins:(clas_id+1)*bins])
        max_id = np.argmax(np.histogram(nonbiased_samples, bins=bins, range=ranger)[0])
        x_coord = hist.patches[clas_id*bins+max_id].get_x()+hist.patches[clas_id*bins+max_id].get_width()/2
        plt.axvline(x_coord, c='red')


    plt.xticks(np.arange(ranger[0], ranger[1], int(ranger[1]/10.0)))
    plt.savefig(png_loc, dpi=300, bbox_inches='tight')
    #plt.show()

def get_multi_selected_similarities(model_dict, out_loc):
    notype_loc = config.ROOT_DIR / 'bias_histograms' / f'{out_loc}_T'#f'{model_dict["params"]["dataset"]["name"]}' / 'bias_histograms' / f'{out_loc}.png'
    config.ensure_dir(notype_loc)
    
    #Plot settings
    plt.clf()
    if len(model_dict['params']['datasets'])==3:
        plt.rcParams["figure.figsize"] = (6,6)#(24,25)#((bias_size/best_facts[0])*6, (bias_size/best_facts[1])*6)
    elif len(model_dict['params']['datasets'])==4:
        plt.rcParams["figure.figsize"] = (8,6) #(24,25)#((bias_size/best_facts[0])*6, (bias_size/best_facts[1])*6)
    plt.rcParams["legend.labelspacing"] = 0.05
    plt.rcParams["legend.handletextpad"] = 0
    total_columns = len(model_dict['params']['biases'])
    total_rows=len(model_dict['params']['datasets'])
    fig, ax = plt.subplots(total_columns, total_rows, sharex='col')#fig, ax = plt.subplots(total_rows,total_columns)
    plt.subplots_adjust(wspace=0.18, hspace=0.13)
    #fig.subplots_adjust(hspace=0.075, wspace=0.075)
    binrange_dict={'breast_cancer':[0,2500], 'wine_uci2':[0,230], 'mnist':[10,70], 'mushroom':[400, 1100], 'drug':[100, 168], 'fire':[10, 180]}
    class_dict = {'breast_cancer': {0: 'Benign (B)', 1: 'Tumor (T)'}, 'wine_uci2': {0: 'Red (R)', 1: 'White (W)'}, 
                  'mushroom': {0: 'Safe (S)', 1: 'Poison (P)'}, 'drug': {0: 'Resistant (R)', 1: 'Sensitive (S)'},
                 'mnist': {0: '0', 1: '1'}, 'fire': {1: 'Extinguished (E)', 0: 'Non-ext. (N)'}, 'spam': {0: 'Not-Spam (N)', 1: 'Spam (S)'},
                 'pumpkin': {0: 'Cercevelik (C)', 1: 'Urgup (U)'}, 'pistachio': {0: 'Kirmizi (K)', 1: 'Siirt (S)'}, 
                  'rice': {0: 'Cammeo (C)', 1: 'Osmancik (O)'}, 'raisin': {0: 'Kecimen (K)', 1: 'Besni (B)'},
                 'adult': {0: '<=50K', 1: '>50K'}}
    ds_name_fix = {'breast_cancer':"Breast Cancer", 'wine_uci2': "Wine", "mushroom": "Mushroom", "mnist": "MNIST", "drug_CX-5461": "Drug", 'spam': 'Spam', 
               'adult': 'Adult', 'rice': 'Rice', 'fire': 'Fire', 'pumpkin': 'Pumpkin', 'pistachio': 'Pistachio', 'raisin': 'Raisin'}
    bias_name_fix = {'hierarchyy':"Hierarchy (0.9)", 'random': "Random", "joint": "Joint", "dirichlet": "Dirichlet"}
    
    for i_ds, dataset in enumerate(model_dict['params']['datasets']):
        unique_class =2
        if dataset['name']=='mnist':
            unique_class =10
        #Get None bias histogram for the dataset
        new_model_dict = {'models':{}, 'params':{}}
        new_model_dict['params']['bias'] = {'name':None}
        new_model_dict['params']['dataset'] = dataset.copy()
        none_bias_str = f'{new_model_dict["params"]["bias"]["name"]}' \
           f'({"|".join([str(val) for key, val in new_model_dict["params"]["bias"].items() if "name" not in key])})'
        ds_str_name = f'{new_model_dict["params"]["dataset"]["name"]}_{"_".join(str(val) for val in new_model_dict["params"]["dataset"]["args"].values())}'
        ds_str = f'{ds_str_name}' \
                 f'(tr{new_model_dict["params"]["dataset"]["train"]}' \
                 f'|val{new_model_dict["params"]["dataset"]["val"]}' \
                 f'|te{new_model_dict["params"]["dataset"]["test"]}' \
                 f'|unk{new_model_dict["params"]["dataset"]["unlabeled"]})'
        df_res_none_loc = config.RESULT_DIR / 'bias_histograms' / f'data' / f'{none_bias_str}_{ds_str}_df.csv'
        config.ensure_dir(df_res_none_loc)
        if os.path.exists(df_res_none_loc):
            df_res_none = pd.read_csv(df_res_none_loc)
        else:
            df_res_none, all_distances_none, all_classes_none = get_distances(new_model_dict)
            df_res_none.to_csv(df_res_none_loc)
        print(df_res_none.head())
        for i, bias in enumerate(model_dict['params']['biases']):
            ax_tmp = ax[i, i_ds]#ax[i_ds,i]#//total_rows, i%total_rows]
            bias_specific_model_dict = {'models':{}, 'params':{}}
            bias_specific_model_dict['params']['bias'] = bias.copy()
            if bias["name"]=='dirichlet':
                bias_specific_model_dict['params']['bias']['n'] = bias['n']*unique_class
            bias_specific_model_dict['params']['dataset'] = dataset.copy()#model_dict['params']['dataset'].copy()
            normal_bias_str = f'{bias_specific_model_dict["params"]["bias"]["name"]}({"|".join([str(val) for key, val in bias_specific_model_dict["params"]["bias"].items() if "name" not in key])})'
            df_res_loc = config.RESULT_DIR / 'bias_histograms' / f'data' / f'{normal_bias_str}_{ds_str}_df.csv'
            if os.path.exists(df_res_loc):
                df_res = pd.read_csv(df_res_loc)
            else:
                df_res, all_distances, all_classes = get_distances(bias_specific_model_dict)
                df_res.to_csv(df_res_loc)
            print(df_res['selected_id'].values)
            print(df_res_none['selected_id'].values)
            df_res_none_tmp = df_res_none.set_index('selected_id').loc[df_res['selected_id'].values].reset_index()
            print('xd')
            print(df_res['selected_id'].values)
            print(df_res_none['selected_id'].values)

            #all_res = np.concatenate(all_distances, axis=0)
            #all_class = np.concatenate(all_classes, axis=0)
            #colors = ['blue' if clas==1 else 'red' for clas in all_class]
            #plt.hist(all_res,50, rwidth=0.8, hue='colors')
            maxr,minr = max(df_res['score'].values), min(df_res['score'].values)
            if dataset["name"] not in binrange_dict:
                binrange_dict[dataset["name"]] = [minr, maxr]
            ranger = binrange_dict[dataset["name"]]#binrange_dict[model_dict["params"]["dataset"]["name"]]
            if dataset["name"] == 'mnist':
                df_res = df_res[np.isin(df_res['selected_class'],[0,1])]
                df_res['Class'] = df_res['selected_class'].copy()
            else:
                df_res['Class'] = df_res['selected_class'].replace(class_dict[dataset['name']])
            bins = 50
            hist = sns.histplot(df_res,x='score', bins=bins, shrink=0.8, hue='Class', binrange = ranger, palette=["#994455","#004488"], ax=ax_tmp, edgecolor="white", linewidth=0.1)#'colorblind'
            
            
            ax_tmp.set_xticks(np.arange(ranger[0], ranger[1], int((ranger[1]-ranger[0])/7.0)))
            ax_tmp.tick_params(axis="x", labelsize=5.5, width=0.3,length=1.2, pad=1.8)
            ax_tmp.tick_params(axis="y", labelsize=5.5, width=0.3,length=1.2, pad=1.8)
            #ax_tmp.set_title(f'{ds_name_fix[dataset["name"]]} - {bias_name_fix[bias["name"]]}', size=6, pad=-0.2, family='Arial')
            ax_tmp.set_title(f'{bias_name_fix[bias["name"]]}', size=6, pad=-0.05, family='Arial')
            ax_tmp.spines.right.set_visible(False)
            ax_tmp.spines.top.set_visible(False)
            ax_tmp.spines['left'].set_linewidth(0.3)
            ax_tmp.spines['bottom'].set_linewidth(0.3)
            #hist.legend_.set_bbox_to_anchor((1,1))
            if 'hierarchy' in bias['name']:
                sns.move_legend(hist, "lower center", bbox_to_anchor=(.5, 1.03), ncol=2, frameon=False, handlelength = 1, handleheight = 0.5, labelspacing = 0.05, handletextpad = 0.1, columnspacing=0.5,
                               title=ds_name_fix[dataset["name"]], title_fontsize=6.5, fontsize=5.6)
                #hist.legend_.get_frame().set_alpha(0)
                #hist.legend_.set_title(ds_name_fix[dataset["name"]], fontsize=6, family='Arial')
                #plt.setp(hist.legend_.get_texts(), fontsize='5.6', family='Arial')
                #hist.legend_.labelspacing = 0.05
                #hist.legend_.handletextpad=0
                #for lh in hist.legend_.legendHandles: 
                #    lh.set_height(4)
                #    lh.set_width(8) 
                stats_y_loc = 0.95
            else:
                hist.get_legend().remove() 
                stats_y_loc = 0.95
            #hist.legend_.get_frame().set_alpha(0)
            #handles_tmp, labels_tmp = ax_tmp.get_legend_handles_labels()
            #ax_tmp.legend(handles_tmp, labels_tmp, frameon=False)
            heights = [pat.get_height() for pat in hist.patches] 

            #full_res = df_res.merge(df_res_none, how='left', left_on=['selected_id', 'selected_class', 'fold'],
            #                 right_on=['selected_id', 'selected_class', 'fold'],
            #                 suffixes=('_bias', '_none'))
            test_res = {}
            test_res2 = {}
            legx, legy = ax_tmp.transLimits.inverted().transform((0.62,stats_y_loc))
            #ax_tmp.text(ax_tmp.get_xlim()[1]-ax_tmp.get_xlim()[1]/5, ax_tmp.get_ylim()[1] - ax_tmp.get_ylim()[1] * 2 / 10 - ax_tmp.get_ylim()[1] * -1 / 30 , 'KS Statistics',
            #         fontsize=16, ha='left', va='top', wrap=True, font='Arial', fontweight='bold') #- ax_tmp.get_ylim()[1]/40
            ax_tmp.text(legx, legy , 'KS Statistics',
                     fontsize=6, ha='left', va='top', wrap=True, family='Arial') #- ax_tmp.get_ylim()[1]/40
            color_palette = ["#994455","#004488"]
            for clas_id, clas in enumerate(df_res['selected_class'].unique()):
                biased_samples = df_res[df_res['selected_class']==clas]['score'].values
                nonbiased_samples = df_res_none_tmp[df_res_none_tmp['selected_class']==clas]['score'].values
                res = ks_2samp(biased_samples, nonbiased_samples)
                res2 = mannwhitneyu(biased_samples, nonbiased_samples)
                test_res[clas] = res
                test_res2[clas] = res2
                #res_txt = f'Class {clas}: {res.pvalue:.2e}'
                if dataset["name"] == 'mnist':
                    res_txt = f'Class {clas}: {res.statistic:.3f}'
                else:
                    res_txt = f'Class {class_dict[dataset["name"]][clas][-2]}: {res.statistic:.2f}'
                #ax_tmp.text(ax_tmp.get_xlim()[1]-ax_tmp.get_xlim()[1]/5, ax_tmp.get_ylim()[1] - ax_tmp.get_ylim()[1] * 2 / 10 - ax_tmp.get_ylim()[1] * clas_id / 20, res_txt,
                #         fontsize=16, ha='left', va='top', wrap=True, font='Arial')# - ax_tmp.get_ylim()[1]/40
                legx_tmp, legy_tmp = ax_tmp.transLimits.inverted().transform((0.62,stats_y_loc-0.08-clas_id*0.07))
                ax_tmp.text(legx_tmp, legy_tmp, res_txt,
                         fontsize=5.6, ha='left', va='top', wrap=True, family='Arial')# - ax_tmp.get_ylim()[1]/40
                #max_id = np.argmax(heights[clas_id*bins:(clas_id+1)*bins])
                max_id = np.argmax(np.histogram(nonbiased_samples, bins=bins, range=ranger)[0])
                x_coord = hist.patches[clas_id*bins+max_id].get_x()+hist.patches[clas_id*bins+max_id].get_width()/2
                ax_tmp.axvline(x_coord, c=color_palette[clas_id], linewidth=0.3) 

            if i_ds!=0:
                ax_tmp.get_yaxis().get_label().set_visible(False)
            else:
                ax_tmp.set_ylabel('Number of Samples', fontsize=6, family='Arial', fontdict={'weight':'bold'})
            if i!=len(model_dict['params']['biases'])-1:
                ax_tmp.get_xaxis().get_label().set_visible(False)
            else:
                ax_tmp.set_xlabel('Average Euclidian Distances', fontsize=6, family='Arial', fontdict={'weight':'bold'})

            #ax_tmp.set_xlabel('Distances')
    
    plt.margins(0,0)
    plt.savefig(f'{notype_loc}x.png', dpi=300, bbox_inches='tight', pad_inches = 0.02)
    plt.savefig(f'{notype_loc}x.pdf', dpi=300, bbox_inches='tight', pad_inches = 0.02)
    #plt.savefig(f'{notype_loc}.eps', dpi=300, bbox_inches='tight')
    
def draw_distr():
    datasets = ['mushroom']#['drug_CX-5461']#['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'drug_CX-5461']
    for ds_name in datasets:
        bias_per_class = args.bias_size
        dataset = {'name': ds_name,'args':{}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
        if 'drug_' in ds_name:
            dataset = {'name': 'drug','args':{'drug': ds_name.split('_')[1]}, 'order': 'train_test_unlabeled_bias_validation',
                                                               'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}

        unique_class =2
        if ds_name=='mnist':
            unique_class=10

        if args.bias=='hierarchyy9':
            bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.9} 
        if args.bias=='hierarchyy8':
            bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.8} 
        if 'hierarchyy_' in args.bias:
            bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': float(args.bias.split('_')[1])}
        if args.bias == 'none':
            bias = {'name': None}
        if args.bias == 'random':
            bias = {'name': 'random', 'y': True, 'size':bias_per_class}
        if args.bias == 'joint':
            bias = {'name': 'joint'}
        if args.bias == 'dirichlet':
            bias = {'name': 'dirichlet', 'n': bias_per_class * unique_class} 

        model_dict = {'models': {}, 'params': {}}
        model_dict['params']['bias'] = bias
        model_dict['params']['dataset'] = dataset
        bias_str = f'{model_dict["params"]["bias"]["name"]}' \
                   f'({"|".join([str(val) for key, val in model_dict["params"]["bias"].items() if "name" not in key])})'

        ds_str_name = f'{model_dict["params"]["dataset"]["name"]}_{"_".join(str(val) for val in model_dict["params"]["dataset"]["args"].values())}'
        ds_str = f'{ds_str_name}' \
                 f'(tr{model_dict["params"]["dataset"]["train"]}' \
                 f'|val{model_dict["params"]["dataset"]["val"]}' \
                 f'|te{model_dict["params"]["dataset"]["test"]}' \
                 f'|unk{model_dict["params"]["dataset"]["unlabeled"]})'
        out_loc = f'{bias_str}_{ds_str}_xx'
        get_selected_similarities(model_dict, out_loc)

def draw_multi_distr():
    if args.ds_group == 0:
        datasets = ['wine_uci2', 'mushroom', 'fire']
    elif args.ds_group == 1:
        datasets = ['breast_cancer', 'pumpkin', 'mnist', 'spam']
    elif args.ds_group == 2:
        datasets = ['raisin', 'rice', 'pistachio', 'adult']
    elif args.ds_group == 3:
        datasets = ['mnist', 'breast_cancer']#['breast_cancer', 'mnist']#['wine_uci2', 'mushroom', 'drug_CX-5461']#['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'drug_CX-5461']
    
    #for ds_name in datasets:
    bias_per_class = args.bias_size
    #dataset = {'name': ds_name,'args':{}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
    #if 'drug_' in ds_name:
    #    dataset = {'name': 'drug','args':{'drug': ds_name.split('_')[1]}, 'order': 'train_test_unlabeled_bias_validation',
    #                                                       'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}

    #unique_class =2
    #if ds_name=='mnist':
    #    unique_class=10
    bias_list = []
    bias_list.append({'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.9})
    #bias_list.append({'name': None}
    bias_list.append({'name': 'random', 'y': True, 'size':bias_per_class})
    bias_list.append({'name': 'joint'})
    bias_list.append({'name': 'dirichlet', 'n': bias_per_class})# * unique_class})

    model_dict = {'models': {}, 'params': {}}
    model_dict['params']['biases'] = bias_list
    bias_str_list = []
    for bias in bias_list:
        bias_str_list.append(f'{bias["name"]}' \
               f'({"|".join([str(val) for key, val in bias.items() if "name" not in key])})')
    bias_str = f'({"_".join(bias_str_list)})'
    print(bias_str)

    ds_str_list = []
    ds_list = []
    for ds_name in datasets:
        dataset = {'name': ds_name,'args':{}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
        if 'drug_' in ds_name:
            dataset = {'name': 'drug','args':{'drug': ds_name.split('_')[1]}, 'order': 'train_test_unlabeled_bias_validation',
                                                           'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
        ds_list.append(dataset)
        ds_str_list.append(f'{dataset["name"]}_{"_".join(str(val) for val in dataset["args"].values())}')
    ds_str = f'({"_".join(ds_str_list)})'
    
    model_dict['params']['datasets'] = ds_list

    #ds_str_name = f'{model_dict["params"]["dataset"]["name"]}_{"_".join(str(val) for val in model_dict["params"]["dataset"]["args"].values())}'
    #ds_str = f'{ds_str_name}' \
    #         f'(tr{model_dict["params"]["dataset"]["train"]}' \
    #         f'|val{model_dict["params"]["dataset"]["val"]}' \
    #         f'|te{model_dict["params"]["dataset"]["test"]}' \
    #         f'|unk{model_dict["params"]["dataset"]["unlabeled"]})'
    out_loc = f'{bias_str}_{ds_str}_same_stats' #Same selected samples are used to calculate the distance to others.
    print(out_loc)
    get_multi_selected_similarities(model_dict, out_loc)

#draw_multi_distr()
#draw_distr()
if 'umap' in args.task:
    visualize_umap_final()
elif 'multi_dist' in args.task:
    draw_multi_distr()
elif 'dist' in args.task:
    draw_distr()