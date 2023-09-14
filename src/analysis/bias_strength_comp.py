import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower() == 'diversepsuedolabeling':
        project_path = '/'.join(path2this[:i + 1])
sys.path.insert(0, project_path)
import numpy as np
import pandas as pd
from src import config
from src.lib import visual_fnc
import warnings
warnings.filterwarnings('ignore')
from matplotlib import rcParams
import math
import argparse

parser = argparse.ArgumentParser(description='RFE arguments')
parser.add_argument('--bias_size', '-bs', metavar='the-bias-size', dest='bias_size', type=int, help='Choose the bias size per class', default=30)
parser.add_argument('--folder', '-f', metavar='the-result-folder', dest='res_fold', type=str, help='Choose the main result folder', default='results_test_nb_imb')#results_nn_test_nb_imb_fin
parser.add_argument('--thold', '-t', metavar='the-threshold', dest='threshold', type=float, help='Choose the main threshold', default=97)
parser.add_argument('--valbase', '-v', metavar='the-val-base', dest='val_base', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--dgroup', '-dg', metavar='d-group', dest='d_group', type=float, help='Choose the dataset group', default=0)
#parser.add_argument('--feature', default=True, action=argparse.BooleanOptionalAction)


args = parser.parse_args()
print(f'Running args:{args}')

R_STATE = 123

params = {
    # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    "subsample": 0.9, "subsample_freq": 1,
    #"min_child_weight":0.01,
    "reg_lambda": 5,
    "class_weight": 'balanced',
    "random_state": R_STATE, "verbose": -1, "n_jobs":-1, 
}


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 1])
                xmax = np.max(verts_sub[:, 1])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 1] == xmin, 1] = xmin_new
                verts_sub[verts_sub[:, 1] == xmax, 1] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

def get_res(res_fold, bias_name, ds_folder_name, out_loc):
    #try:
    res_loc = config.ROOT_DIR / res_fold / f'{bias_name}' / ds_folder_name / f'{out_loc}.csv'
    if not os.path.exists(res_loc):
        res_loc = config.ROOT_DIR / res_fold / f'{bias_name}' / ds_folder_name / f'{out_loc}).csv'
    res_df = pd.read_csv(res_loc)
    #except:
    #    if args.res_fold == 'results_test_nb_imb':
    #        res_loc = config.ROOT_DIR / 'resultsxxx' / f'{bias_name}' / ds_folder_name / f'{out_loc}.csv'
    #        res_df = pd.read_csv(res_loc)
    return res_df


def create_model_dict(ds_name):
    bias_per_class = args.bias_size
    dataset = {'name': ds_name,'args':{}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
    if 'drug_' in ds_name:
        bias_per_class = 50
        dataset = {'name': 'drug','args':{'drug': ds_name.split('_')[1]}, 'order': 'train_test_unlabeled_bias_validation',
                                                           'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
    if args.bias=='hierarchyy9':
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.9} 
    if args.bias=='hierarchyy8':
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.8} 
    if 'hierarchyy_' in args.bias:
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': float(args.bias.split('_')[1])}
    th, kb, mi, balance, val_base = args.threshold, 6, 100, 'ratio', args.val_base
    if ds_name == 'mnist':
        #dataset = {'name': ds_name, 'args': {}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.35,
        #           'test': 0.25, 'unlabeled': 0.4, 'val': 0.2, 'runs': 30}
        dataset = {'name': ds_name, 'args': {}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.30, 'test': 0.2, 'unlabeled': 0.70, 'val': 0.2, 'runs': 30}
        
        th, kb, mi, balance, val_base = 85, 30, 100, 'ratio', args.val_base
        if 'nn' in args.res_fold and 'kmm' not in args.res_fold:
            th, kb, mi, balance, val_base = args.threshold, 30, 100, 'ratio', args.val_base
        #th=85
        
    if th>1:
        th=int(th)
    model_dict = {'models': {}, 'params': {}}
    model_dict['params']['bias'] = bias
    model_dict['params']['dataset'] = dataset
    model_dict['params']['base_model'] = params
    model_dict['params']['base_model']['name'] = 'BRRF1'
    model_dict['params']['model'] = {'name': 'DST', 'threshold': th, 'k_best': kb, 'max_iter': mi, 'balance': balance, 'val_base': val_base}
    model_dict['params']['model']['full_name'] = f'{model_dict["params"]["model"]["name"]}-{model_dict["params"]["base_model"]["name"]}'
    bias_str = f'{model_dict["params"]["bias"]["name"]}({"|".join([str(val) for key, val in model_dict["params"]["bias"].items() if "name" not in key])})'

    ds_str = f'{model_dict["params"]["dataset"]["name"]}' \
             f'({model_dict["params"]["dataset"]["train"]}' \
             f'|{model_dict["params"]["dataset"]["val"]}' \
             f'|{model_dict["params"]["dataset"]["test"]}' \
             f'|{model_dict["params"]["dataset"]["unlabeled"]})'
    model_str = f'{model_dict["params"]["model"]["full_name"]}' \
                f'(th={model_dict["params"]["model"]["threshold"]}' \
                f'|kb={model_dict["params"]["model"]["k_best"]}' \
                f'|mi={model_dict["params"]["model"]["max_iter"]}' \
                f'|vb={model_dict["params"]["model"]["val_base"]}' \
                f'|b={model_dict["params"]["model"]["balance"]})'
    out_loc = f'{bias_str}_{ds_str}_{model_str}'
    
    return model_dict, out_loc

def get_bias_dict(bias_name):
    bias_per_class = args.bias_size
    if bias_name=='hierarchyy9':
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.9} 
    if bias_name=='hierarchyy8':
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.8} 
    if 'hierarchyy_' in bias_name:
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': float(bias_name.split('_')[1])}
    if bias_name == 'none':
        bias = {'name': None}
    if bias_name == 'random':
        bias = {'name': 'random', 'y': True, 'size': bias_per_class}
    if bias_name == 'joint':
        bias = {'name': 'joint'}
    if bias_name == 'dirichlet':
        bias = {'name': 'dirichlet', 'n': bias_per_class}
    return bias
    
    
def none_hierarchy_multi_datasets_facet(datasets, bias_list, models, metric="LogLoss", swarmed=True):
    colsize = 11
    rowsize = int(len(datasets)/colsize)
    mutids_name = '|'.join(datasets)
    plt.clf()
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    last_ds = datasets[-1]
    res_folder = config.ROOT_DIR / 'bias_strength_all_comparisons' 
    all_datasets_lst = []
    models_all_str = '_'.join(models)
    ds_name_fix = {'breast_cancer':"Breast Cancer", 'wine_uci2': "Wine", "mushroom": "Mushroom", "mnist": "MNIST", "drug_CX-5461": "Drug", 'spam': 'Spam', 
                   'adult': 'Adult', 'rice': 'Rice', 'fire': 'Fire', 'pumpkin': 'Pumpkin', 'pistachio': 'Pistachio', 'raisin': 'Raisin'}
    for dataset in datasets:
        print(f'Dataset {dataset} started!')
        lst_to_concat =[]
        kb=6
        u_classes = 2
        th_rf=97
        th_lr=0.9
        ds_str = f'{dataset}({0.3}|{0.2}|{0.2}|{0.7})'
        if 'drug_' in dataset:
            ds_str = f'drug({0.3}|{0.2}|{0.2}|{0.7})'
        if 'mnist' in dataset:
            th_rf=85
            kb=30
            u_classes=10
        
        if 'RF' in models:
            #Main RF
            dictre_rf = {"SU": 'Biased-RF', "DST-None": 'BaST-RF', 'DST-10':'DBaST-10-RF', 'DST-100':'DBaST-100-RF'}
            for bias_name in bias_list:# 0.6, 0.7, 0.8, 0.9]:
                bias = get_bias_dict(bias_name)
                model_rf_str = f'DST-BRRF1(th=97|kb={kb}|mi=100|vb=False|b=ratio)_es=True'
                if dataset=='mnist':
                    model_rf_str = f'DST-BRRF1(th=85|kb={kb}|mi=100|vb=False|b=ratio)_es=True'
                if bias_name =='dirichlet':
                    bias['n']=args.bias_size*2
                    if dataset=='mnist':
                        bias['n']=args.bias_size*10
                bias_str = f'{bias["name"]}({"|".join([str(val) for key, val in bias.items() if "name" not in key])})'
                out_rf_loc = f'{bias_str}_{ds_str}_{model_rf_str}'
                rf_df = get_res('results_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_rf_loc)
                if 'ST_None' in rf_df['Method'].unique():
                    dictre_rf = {"SU": 'Biased-RF', "ST_None": 'BaST-RF', 'ST_10':'DBaST-10-RF', 'ST_100':'DBaST-100-RF'}
                    rf_df = rf_df[~np.isin(rf_df['Method'],['ST_50', 'ST_200'])]
                rf_df = rf_df[~np.isin(rf_df['Method'],['DST-50', 'DST-200'])]
                rf_df = rf_df.replace({"Method": dictre_rf})
                rf_df['Bias'] = bias_name
                rf_df['Main Model'] = 'RF'
                lst_to_concat.append(rf_df)
        
        if 'NN' in models:    
            #Main NN
            for bias_name in bias_list:# 0.6, 0.7, 0.8, 0.9]:
                bias = get_bias_dict(bias_name)
                if bias_name =='dirichlet':
                    bias['n']=args.bias_size*2
                    if dataset=='mnist':
                        bias['n']=args.bias_size*10
                model_nn_str = f'DST-BRRF1(th=0.9|kb={kb}|mi=100|vb=False|b=ratio)_es=True'
                bias_str = f'{bias["name"]}({"|".join([str(val) for key, val in bias.items() if "name" not in key])})'
                out_nn_loc = f'{bias_str}_{ds_str}_{model_nn_str}'
                nn_df = get_res('results_nn_test_nb_imb_fin_cw3_ss8', f'{bias["name"]}', dataset, out_nn_loc)
                nn_df = nn_df[~np.isin(nn_df['Method'],['DST-50', 'DST-200'])]
                dictre_nn = {"SU": 'Biased-NN', "DST-None": 'BaST-NN', 'DST-10':'DBaST-10-NN', 'DST-100':'DBaST-100-NN'}
                if bias_name =='none':
                    nn_df = nn_df[~np.isin(nn_df['Method'],["DST-None", "DST-10", 'DST-50', "DST-100", 'DST-200'])]
                    dictre_nn = {"SU": 'No Bias-NN', "DST-None": 'BaST-NN', 'DST-10':'DBaST-10-NN', 'DST-100':'DBaST-100-NN'}
                nn_df = nn_df.replace({"Method": dictre_nn})
                nn_df['Bias'] = bias_name
                nn_df['Main Model'] = 'NN'
                lst_to_concat.append(nn_df)
        
        if 'LR' in models:
            #LR - DBaST
            dictre_lr_dbst = {"DST-None": 'BaST-LR', 'DST-100':'DBaST-100-LR'} 
            for bias_name in bias_list:
                bias = get_bias_dict(bias_name)
                if bias_name =='dirichlet':
                    bias['n']=args.bias_size*2
                    if dataset=='mnist':
                        bias['n']=args.bias_size*10
                bias_str = f'{bias["name"]}({"|".join([str(val) for key, val in bias.items() if "name" not in key])})'
                model_lr_dbst_str = f'DST-BRRF1(th=0.9|kb={kb}|mi=100|vb=False|b=ratio)'
                try:
                    out_lr_dbst_loc = f'{bias_str}_{ds_str}_{model_lr_dbst_str}_es=True'
                    lr_dbst_df = get_res('results_lr_fsd_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_lr_dbst_loc)
                except:
                    out_lr_dbst_loc = f'{bias_str}_{ds_str}_{model_lr_dbst_str}'
                    lr_dbst_df = get_res('results_lr_fsd_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_lr_dbst_loc)
                if 'ST_None' in lr_dbst_df['Method'].unique():
                    dictre_lr_dbst = {"ST_None": 'BaST-LR', 'ST_100':'DBaST-100-LR'}
                    lr_dbst_df = lr_dbst_df[~np.isin(lr_dbst_df['Method'],['SU', 'ST_10', 'ST_50', 'ST_200'])]
                lr_dbst_df = lr_dbst_df[~np.isin(lr_dbst_df['Method'],['SU', 'DST-10', 'DST-50', 'DST-200'])]
                lr_dbst_df = lr_dbst_df.replace({"Method": dictre_lr_dbst})
                lr_dbst_df['Bias'] = bias_name
                lr_dbst_df['Main Model'] = 'LR'
                lst_to_concat.append(lr_dbst_df) 
                
        a = pd.concat(lst_to_concat)
        a['Dataset'] = ds_name_fix[dataset]
        all_datasets_lst.append(a)
        #print(a.head())
    print('uu')
    accepted_methods = []
    for main_method in models:
        accepted_methods.append(f'No Bias-{main_method}')
        accepted_methods.append(f'Biased-{main_method}')
        accepted_methods.append(f'BaST-{main_method}')
        accepted_methods.append(f'DBaST-100-{main_method}')
    #accepted_methods = ['No Bias-RF', 'Biased-RF', 'BaST-RF',  'DBaST-100-RF', 'KMM-RF', 'No Bias-RF', 'Biased-NN', 'BaST-NN', 'DBaST-100-NN', 'KMM-#NN', 'No Bias-RF', 'Biased-LR', 'BaST-LR', 'DBaST-100-LR', 'KMM-LR', 'KDE-LR', 'RBA-LR', 'FLDA-LR', 'TCPR-LR', 'SUBA-LR']
    #'DBaST-10-NN','DBaST-10-RF',
    #color_dict = {'RF':'#DDAA33', 'BaST-RF':'#bdd7e7', 'DBaST-10-RF':'#6baed6', 'DBaST-100-RF':'#3182bd', 'KMM-RF':'#08519c',
    #            'NN':'#DDAA33', 'BaST-NN':'#bae4b3', 'DBaST-10-NN':'#74c476', 'DBaST-100-NN':'#31a354', 'KMM-NN':'#006d2c',
    #            'Biased-LR':'#DDAA33', 'KMM-LR':'#fcbba1', 'KDE-LR':'#fc9272', 'RBA-LR':'#fb6a4a', 'FLDA-LR':'#ef3b2c', 'TCPR-LR':'#cb181d', 'SUBA-LR':'#99000d'}
    #Supervised: '#DDAA33'
    
    '''
    color_dict = {'Biased-RF':'#DDAA33', 'BaST-RF':'#08519c', 'DBaST-10-RF':'#08519c', 'DBaST-100-RF':'#08519c', 'KMM-RF':'#08519c',
                    'Biased-NN':'#DDAA33', 'BaST-NN':'#006d2c', 'DBaST-10-NN':'#006d2c', 'DBaST-100-NN':'#006d2c', 'KMM-NN':'#006d2c',
                  'Biased-LR':'#DDAA33', 'BaST-LR':'#99000d', 'DBaST-100-LR':'#99000d', 'KMM-LR':'#99000d', 'KDE-LR':'#99000d', 'RBA-LR':'#99000d', 'FLDA-LR':'#99000d', 'TCPR-LR':'#99000d', 'SUBA-LR':'#99000d'}
    
    bp_color_dict = {'Biased-RF':'#DDAA33', 'BaST-RF':'#08519c', 'DBaST-10-RF':'#08519c', 'DBaST-100-RF':'#08519c', 'KMM-RF':'#08519c',
                'Biased-NN':'#DDAA33', 'BaST-NN':'#006d2c', 'DBaST-10-NN':'#006d2c', 'DBaST-100-NN':'#006d2c', 'KMM-NN':'#006d2c',
                'Biased-LR':'#DDAA33', 'BaST-LR':'#99000d', 'DBaST-100-LR':'#99000d', 'KMM-LR':'#99000d', 'KDE-LR':'#99000d', 'RBA-LR':'#99000d', 'FLDA-LR':'#99000d', 'TCPR-LR':'#99000d', 'SUBA-LR':'#99000d'}
    '''
    color_dict = {'Biased-RF':'#000000', 'BaST-RF':'#08519c', 'DBaST-10-RF':'#08519c', 'DBaST-100-RF':'#08519c', 'KMM-RF':'#DDAA33', 
                  'No Bias-NN': 'gray',
                    'Biased-NN':'#000000', 'BaST-NN':'#238b45', 'DBaST-10-NN':'#238b45', 'DBaST-100-NN':'#238b45', 'KMM-NN':'#DDAA33',
                  'Biased-LR':'#000000', 'BaST-LR':'#cb181d', 'DBaST-100-LR':'#cb181d', 'KMM-LR':'#DDAA33', 'KDE-LR':'#DDAA33', 'RBA-LR':'#54278f', 'FLDA-LR':'#54278f', 'TCPR-LR':'#54278f', 'SUBA-LR':'#54278f'}
    
    bp_color_dict = {'Biased-RF':'#FFFFFF', 'BaST-RF':'#6baed6', 'DBaST-10-RF':'#6baed6', 'DBaST-100-RF':'#6baed6', 'KMM-RF':'#DDAA33',
                'No Bias-NN': 'gray','Biased-NN':'#FFFFFF', 'BaST-NN':'#41ab5d', 'DBaST-10-NN':'#41ab5d', 'DBaST-100-NN':'#41ab5d', 'KMM-NN':'#DDAA33',
                'Biased-LR':'#FFFFFF', 'BaST-LR':'#ef3b2c', 'DBaST-100-LR':'#ef3b2c', 'KMM-LR':'#DDAA33', 'KDE-LR':'#DDAA33', 'RBA-LR':'#6a51a3', 'FLDA-LR':'#6a51a3', 'TCPR-LR':'#6a51a3', 'SUBA-LR':'#6a51a3'}
    
    
    met_hue = {'Biased-RF':'Biased', 'BaST-RF':'SS', 'DBaST-10-RF':'SS', 'DBaST-100-RF':'SS', 'KMM-RF':'KMM',
                'Biased-NN':'Biased', 'BaST-NN':'SS', 'DBaST-10-NN':'SS', 'DBaST-100-NN':'SS', 'KMM-NN':'KMM',
                'Biased-LR':'Biased', 'BaST-LR':'SS', 'DBaST-100-LR':'SS', 'KMM-LR':'KMM', 'RBA-LR':'RBA', 'KDE-LR':'KDE', 'FLDA-LR':'FLDA', 'TCPR-LR':'TCPR', 'SUBA-LR':'SUBA'}
    
    #ds_sort = {"Breast Cancer":0, "Wine":1, "Mushroom":2, "MNIST":3, "Drug":4, 'Adult':5}#'spam', 'adult', 'fire', 'rice', 'raisin', 'pistachio', 'pumpkin'
    #ds_sort = {"Breast Cancer":0, "Wine":1, "Mushroom":2, "MNIST":3, "Drug":4, 'Adult':5, 
    #          "Spam":6, "Adult":7, "Fire":8, "Rice":9, "Raisin":10, 'Pistachio':11, 'Pumpkin':12}
    ds_sort = {ds_name_fix[dssname]:idds for idds, dssname in enumerate(
        datasets)}
    row_no = {key: int(val/colsize) for key,val in ds_sort.items()}
    col_no = {key: int(val%colsize) for key,val in ds_sort.items()}
    met_sort = {metname:idmet for idmet, metname in enumerate(
        accepted_methods)}
    bias_sort = {biasname:idbias for idbias, biasname in enumerate(
        bias_list)}
    #met_sort = {'RF':0, 'BaST-RF':1, 'DBaST-10-RF':2, 'DBaST-100-RF':3, 'KMM-RF':4,
    #            'NN':5, 'BaST-NN':6, 'DBaST-10-NN':7, 'DBaST-100-NN':8, 'KMM-NN':9,
    #            'Biased-LR':10, '' 'KMM-LR':11, 'KDE-LR':12, 'RBA-LR':13, 'FLDA-LR':14, 'TCPR-LR':15, 'SUBA-LR':16}
    
    #row_no = {"Breast Cancer":0, "Wine":0, "Mushroom":0,  "MNIST":1, "Spam":1, "Fire":1}#,  "Raisin":2, 'Pistachio':2, 'Pumpkin':2}
    row_col_dict = {}
    for rr_id in np.unique(list(row_no.values())):
        row_col_dict[rr_id]={}
    print(row_col_dict)
    #col_no = {"Breast Cancer":0, "Wine":1, "Mushroom":2, 
    #          "MNIST":0, "Spam":1, "Fire":2}#, 
              #"Raisin":0, 'Pistachio':1, 'Pumpkin':2}
    for dss_name, row_id in row_no.items():
        col_id = col_no[dss_name]
        row_col_dict[row_id][col_id] = dss_name
    print(row_col_dict)
    all_ds = pd.concat(all_datasets_lst)
    all_ds = all_ds.groupby(["Method", "Dataset", "Iteration", "Bias"])[['Accuracy', 'LogLoss', 'AUROC', 'AUPRC']].median().reset_index()
    all_ds = all_ds[all_ds['Method'].isin(accepted_methods)]
    all_ds['met_sort'] = all_ds['Method'].map(met_sort)
    all_ds['Category'] = all_ds['Method'].map(met_hue)
    all_ds['ds_sort'] = all_ds['Dataset'].map(ds_sort)
    all_ds['bias_sort'] = all_ds['Bias'].map(bias_sort)
    all_ds['row_no'] = all_ds['Dataset'].map(row_no)
    all_ds['col_no'] = all_ds['Dataset'].map(col_no)
    
    all_ds = all_ds.sort_values(['bias_sort', 'ds_sort', 'met_sort', 'Iteration'], ascending=[True, True, True, True])
    print(all_ds.head())
    #print(all_ds.head())
    #print(all_ds.shape)
    if not swarmed:
        PROPS = {
            'boxprops': {'edgecolor': 'black', 'linewidth':0.2},
            'medianprops': {'color': 'black', 'linewidth':0.4},
            'whiskerprops': {'color': 'black', 'linewidth':0.2 },
            'capprops': {'color': 'black', 'linewidth':0.2 },
            'flierprops': {'marker':'o', 'markerfacecolor':'None', 'markersize':1, 'linestyle':'none', 'markeredgewidth':0.3}
        }    
    else:
        PROPS = {
            'boxprops': {'edgecolor': 'black', 'linewidth':0.2},
            'medianprops': {'color': 'black', 'linewidth':0.4},
            'whiskerprops': {'color': 'black', 'linewidth':0.2 },
            'capprops': {'color': 'black', 'linewidth':0.2 },
            'flierprops': {'marker':'o', 'markerfacecolor':'None', 'markersize':1, 'linestyle':'none', 'markeredgewidth':0.3},
            'zorder': 10
        }  
    rc = {'xtick.bottom': True, 'xtick.left': True}
    #fig = plt.figure(figsize=(7, 10))
    #plt.rcParams["figure.figsize"] = (5,6)
    sns.axes_style(style="white", rc=rc) #font='Arial'
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "Arial"
    #sbox = sns.boxplot(y="Dataset", x=metric, hue="Method", data=all_ds, showfliers=True, **PROPS)
    
    def fixed_boxplot(*args, label=None, **kwargs):
        sns.boxplot(*args, **kwargs, labels=[label])
    gf = sns.FacetGrid(all_ds, col='Bias', row='Dataset', height=0.9, aspect=1.1, margin_titles=True, sharex='col', sharey='row', gridspec_kws={"wspace":0.1, 'hspace':0.2, 'width_ratios': [0.3, 1, 1, 1, 1, 1, 1]})
    gmapped = gf.map_dataframe(fixed_boxplot, "Method", metric, showfliers=not swarmed, width=0.6, palette=bp_color_dict, **PROPS)
    for ax in gf.axes.flat:
        for idd, patch in enumerate(ax.artists):
            r, g, b, a = patch.get_facecolor()
            if swarmed:
                patch.set_facecolor((r, g, b, .1))
            else:
                patch.set_facecolor((r, g, b, .95))
    if swarmed:
        gmapped2 = gf.map_dataframe(sns.swarmplot, "Method", metric, palette=color_dict, size=1, zorder=5)
    grouped_summary = all_ds.groupby(['Dataset', 'Method']).agg({'Accuracy':['mean', 'median', 'std', 'min', 'max'], 
                                                                 'LogLoss':['mean', 'median', 'std', 'min', 'max'],
                                                                 'AUROC':['mean', 'median', 'std', 'min', 'max'],
                                                                 'AUPRC':['mean', 'median', 'std', 'min', 'max']})
    print(grouped_summary) 
    print(gf.axes_dict)

    '''
    sbox.legend_.labelspacing = 0.1 
    for lh in sbox.legend_.legendHandles:  
        lh.set_height(5)
        lh.set_width(10) 
    adjust_box_widths(fig, 0.90)
    #ax2 = sns.swarmplot(x="Dataset", y="LogLoss", hue="Method", data=all_ds, size=3, color="#2596be") 
    sbox.legend(frameon=False)
    sbox.axes.tick_params(axis="x", labelsize=6, width=0.3)
    sbox.axes.tick_params(axis="y", labelsize=6, width=0.3)
    sbox.axes.spines['left'].set_linewidth(0.3)
    sbox.axes.spines['bottom'].set_linewidth(0.3)
    sbox.axes.spines.right.set_visible(False)
    sbox.axes.spines.top.set_visible(False)
    sbox.set_xlabel('Dataset', fontsize=6.5, family='Arial', fontdict={'weight':'bold'})
    sbox.set_ylabel(metric, fontsize=6.5, family='Arial', fontdict={'weight':'bold'})
    plt.setp(sbox.legend_.get_texts(), fontsize='6')
    '''
    

    #ax.axhline(y=med_ds, c='black', linewidth=0.2)#, ls='--')
    
    #for datset in all_ds['Dataset'].unique():
    #    med_ds = all_ds[(all_ds['Dataset']==datset)& (all_ds['Method']==f'No Bias-{main_method}')].median()
    #    gf.axes_dict[(datset, 'none')].axhline(y=med_ds, c='black', linewidth=0.2, ls='--')
    
    
        
    for ax in gf.axes.flat:
        if ax.get_ylabel():
            ylim = ax.get_ylim()
            print(f'{ax.get_title()}: {ylim}') 
            upper_ylim = min(math.ceil(ylim[1] * 10)/10, 1.0)
            lower_ylim = max(math.floor(ylim[0] * 10)/10, 0.0)
            if ylim[0]> lower_ylim+0.05:
                lower_ylim2 = lower_ylim+0.05
                lower_ylim = lower_ylim+0.1
            else:
                lower_ylim2 = lower_ylim
            if ylim[1]< upper_ylim-0.05:
                upper_ylim2 = upper_ylim-0.05
                upper_ylim = upper_ylim-0.1
            else:
                upper_ylim2 = upper_ylim
            ax.set(ylim=(lower_ylim2 -0.015, upper_ylim2+0.015))
            ax.set_yticks(np.arange(lower_ylim, upper_ylim+0.0001, 0.1))
        ax.set_xlabel(ax.get_xlabel(), family='Arial', fontdict={'weight':'bold', 'size':6.5}, labelpad=1.6)
        ax.set_ylabel(ax.get_ylabel(), family='Arial', fontdict={'weight':'bold', 'size':6.5}, labelpad=1.6)
        #ax.set_xticklabels(ax.get_xticklabels(), family='Arial', fontdict={ 'size':5.4})
        #ax.set_yticklabels(ax.get_yticklabels(), family='Arial', fontdict={ 'size':5.4})
        plt.setp(ax.get_xticklabels(), size=5.4, rotation=80)
        
        #plt.setp(ax.get_yticklabels(), size=5.4)
        #ax.tick_params(width=0.3)
        #if ax.get_yticklabels():
        #ax.set_yticklabels(ax.get_yticklabels(), family='Arial', fontdict={ 'size':6})
        #print(ax.get_ylabel())
        if False:#not ax.get_ylabel():
            ax.spines['left'].set_linewidth(0.05)
            ax.tick_params(axis='y', width=0.1, length=1)
        else:
            ax.spines['left'].set_linewidth(0.2)
            ax.tick_params(axis='y', width=0.3, length=1.2, pad=2, labelsize=5.4)
            for tick in ax.yaxis.get_major_ticks():
                if 'BaST' in tick.label.get_text():
                    tick.label.set_weight('bold')
        
        if False:#not ax.get_xlabel():
            ax.spines['bottom'].set_linewidth(0.05)
            ax.tick_params(axis='x', width=0.1, length=1)
        else:
            ax.spines['bottom'].set_linewidth(0.2)
            ax.tick_params(axis='x', width=0.3, length=1.2, pad=2)
        #ax.spines['bottom'].set_linewidth(0.2)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        #ax.title.set_position([.5, 0.9])
        #print(ax.get_title())
        #row_id = int(ax.get_title()[9])
        #col_id = int(ax.get_title()[-1])
        #try:
        #    ax.set_title(ax.get_title(), family='Arial', size=6.5, pad=-0.6)
        #    y_axis_ticks = ax.get_yaxis().get_majorticklocs()
        #    ax.axhline(y=3.5, c='black', linewidth=0.2)#, ls='--')
        #    ax.axhline(y=7.5, c='black', linewidth=0.2)#, ls='--')
        #except:
        #    print('This facet is empty')
        #    ax.set_title('')
        #    ax.set_axis_off()
        #for patch in ax.artists:
        #    r, g, b, a = patch.get_facecolor()
        #    patch.set_facecolor((r, g, b, .3))
    #gf.set_titles(template = '{row_name} | {col_name}', family='Arial', size=5.2, pad=0.0)
    gf.set_titles(col_template="{col_name}", row_template="{row_name}", family='Arial', size=5.2)
    #plt.tight_layout()#(pad=0.7)
    
    for (datset, bbias), axx in gf.axes_dict.items():
        if bbias !='none':
            med_ds = all_ds[(all_ds['Dataset']==datset)& (all_ds['Method']==f'No Bias-{main_method}')][metric].median()
            print(f'{datset}: {med_ds}')
            axx.axhline(y=med_ds, c='black', linewidth=0.2, ls='--')
    out_loc = res_folder / f'{models_all_str}_{bias["name"]}_{metric}_{args.val_base}_facet_h{mutids_name}_s={swarmed}_ss8_v4_lr={th_lr}.png'
    out_loc_pdf = res_folder / f'{models_all_str}_{bias["name"]}_{metric}_{args.val_base}_facet_h{mutids_name}_s={swarmed}_ss8_v4_lr={th_lr}.pdf' 
    config.ensure_dir(out_loc)
    plt.margins(0,0)
    plt.savefig(out_loc, dpi=300, bbox_inches='tight', pad_inches = 0)
    plt.savefig(out_loc_pdf, dpi=300, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    

#    png_bp_loss_loc = res_folder / 'images_comp' / f'None_{out_loc}_loss.png'
#    png_bp_loss_match_loc = res_folder / 'images_comp' / f'None_{out_loc}_loss_match.png'
#    png_bp_acc_loc = res_folder / 'images_comp' / f'None_{out_loc}_acc.png'
#    png_bp_acc_match_loc = res_folder / 'images_comp' / f'None_{out_loc}_acc_match.png'
#    print(png_bp_acc_match_loc)
#    config.ensure_dir(png_bp_loss_loc)
#    visual_fnc.call_plot('boxplot_sign', png_bp_loss_loc, a, model_dict, metric='LogLoss')
#    visual_fnc.call_plot('boxplot_match', png_bp_loss_match_loc, a, model_dict, metric='LogLoss')
#    visual_fnc.call_plot('boxplot_sign', png_bp_acc_loc, a, model_dict, metric='Accuracy')
#    visual_fnc.call_plot('boxplot_match', png_bp_acc_match_loc, a, model_dict, metric='Accuracy')


#dataset = {'name': ds_name,'args':{}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.3, 'test': 0.25, 'unlabeled': 0.45, 'val': 0.2, 'runs': 30}

#datasets = ['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'drug_CX-5461']#, 'spam']
#datasets = ['adult', 'drug_CX-5461', 'rice', 'raisin', 'pistachio', 'pumpkin']
models = ['NN']#, 'RF', 'LR']
bias_list = ['none', 'random', 'dirichlet', 'joint', 'hierarchyy_0.5', 'hierarchyy_0.7', 'hierarchyy_0.9']
if args.d_group==0:
    datasets = ['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'rice', 'fire']
elif args.d_group==1:
    datasets = ['adult', 'spam', 'raisin', 'pistachio', 'pumpkin', 'fire']
elif args.d_group==2:
    datasets = ['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'rice', 'fire', 'spam', 'adult', 'raisin', 'pistachio', 'pumpkin'] 
strengths=[0.5, 0.7, 0.9]
#none_hierarchy_multi_datasets_facet(datasets, metric='LogLoss', swarmed=True)#model_dict, out_loc)
none_hierarchy_multi_datasets_facet(datasets, bias_list= bias_list, models=models, metric='Accuracy', swarmed=False)#model_dict, out_loc)
#none_hierarchy_multi_datasets_facet(datasets, metric='LogLoss', swarmed=False)#model_dict, out_loc)
#none_hierarchy_multi_datasets_facet(datasets, metric='Accuracy', swarmed=False)#model_dict, out_loc)
#none_hierarchy_multi_datasets_facet(datasets, metric='AUROC')#model_dict, out_loc)
#none_hierarchy_multi_datasets_facet(datasets, metric='AUPRC')#model_dict, out_loc)