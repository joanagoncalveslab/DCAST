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
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
from scipy.stats import wilcoxon
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches
from functools import reduce

import argparse

parser = argparse.ArgumentParser(description='RFE arguments')
parser.add_argument('--bias', '-b', metavar='the-bias', dest='bias', type=str, help='Choose bias', default='hierarchyy9')
parser.add_argument('--bias_size', '-bs', metavar='the-bias-size', dest='bias_size', type=int, help='Choose the bias size per class', default=30)
parser.add_argument('--folder', '-f', metavar='the-result-folder', dest='res_fold', type=str, help='Choose the result folder', default='results_test_nb_imb')#results_nn_test_nb_imb_fin
parser.add_argument('--thold', '-t', metavar='the-threshold', dest='threshold', type=float, help='Choose the threshold', default=97)
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
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

def get_res(bias_name, exp_fold, ds_folder_name, out_loc):
    #try:
    res_loc = config.ROOT_DIR / exp_fold / f'{bias_name}' / ds_folder_name / f'{out_loc}.csv'
    res_df = pd.read_csv(res_loc)
    #except:
    #    if args.res_fold == 'results_test_nb_imb':
    #        res_loc = config.ROOT_DIR / 'resultsxxx' / f'{bias_name}' / ds_folder_name / f'{out_loc}.csv'
    #        res_df = pd.read_csv(res_loc)
    return res_df


def create_model_dict(ds_name, isnone=False):
    bias_per_class = args.bias_size
    dataset = {'name': ds_name,'args':{}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
    if 'drug_' in ds_name:
        dataset = {'name': 'drug','args':{'drug': ds_name.split('_')[1]}, 'order': 'train_test_unlabeled_bias_validation',
                                                           'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
    if args.bias=='hierarchyy9':
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.9} 
    if args.bias=='hierarchyy8':
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.8} 
    if 'hierarchyy_' in args.bias:
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': float(args.bias.split('_')[1])}
    if args.bias == 'none':
        bias = {'name': None}
    if args.bias == 'random':
        bias = {'name': 'random', 'y': True, 'size': bias_per_class}
    if args.bias == 'joint':
        bias = {'name': 'joint'}
    if args.bias == 'dirichlet':
        bias_per_class = args.bias_size*2
        if ds_name=='mnist':
            bias_per_class = args.bias_size*10
        bias = {'name': 'dirichlet', 'n': bias_per_class}
    th, kb, mi, balance, val_base = args.threshold, 6, 100, 'ratio', args.val_base
    if ds_name == 'mnist':
        #dataset = {'name': ds_name, 'args': {}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.35,
        #           'test': 0.25, 'unlabeled': 0.4, 'val': 0.2, 'runs': 30}
        dataset = {'name': ds_name, 'args': {}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.30, 'test': 0.2, 'unlabeled': 0.70, 'val': 0.2, 'runs': 30}
        
        th, kb, mi, balance, val_base = 85, 30, 100, 'ratio', args.val_base
        if 'nn' in args.res_fold:
            th, kb, mi, balance, val_base = args.threshold, 30, 100, 'ratio', args.val_base
        if isnone:
            th, kb, mi, balance, val_base = 0.9, 30, 100, 'ratio', args.val_base 
    elif isnone:
        th, kb, mi, balance, val_base = 0.9, 6, 100, 'ratio', args.val_base
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

def none_hierarchy_multi_datasets(datasets, metric="LogLoss", base_m='RF', swarmed=False, version=0):
    mutids_name = '|'.join(datasets)
    width_bp = 0.8
    alpha=0.95
    if swarmed:
        alpha=0.1
    if base_m == 'RF':
        long_base_m = 'Random Forest'
    elif base_m == 'NN':
        long_base_m = 'Neural Network'
    print(f'{metric} started')
    if args.bias=='hierarchyy9':
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': args.bias_size, 'prob': 0.9} 
    if args.bias=='hierarchyy8':
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': args.bias_size, 'prob': 0.8} 
    if 'hierarchyy_' in args.bias:
        bias = {'name': 'hierarchyy', 'y': True, 'max_size': args.bias_size, 'prob': float(args.bias.split('_')[1])}
    if args.bias == 'none':
        bias = {'name': None}
    if args.bias == 'random':
        bias = {'name': 'random', 'y': True, 'size': args.bias_size}
    if args.bias == 'joint':
        bias = {'name': 'joint'}
    if args.bias == 'dirichlet':
        bias = {'name': 'dirichlet', 'n': args.bias_size}
    normal_bias_str = f'{bias["name"]}({"|".join([str(val) for key, val in bias.items() if "name" not in key])})'
    res_folder = config.ROOT_DIR / args.res_fold / f'{bias["name"]}'
    all_datasets_lst = []
    ds_name_fix = {'breast_cancer':"Breast Cancer", 'wine_uci2': "Wine", "mushroom": "Mushroom", "mnist": "MNIST", "drug_CX-5461": "Drug", 'spam': 'Spam', 
               'adult': 'Adult', 'rice': 'Rice', 'fire': 'Fire', 'pumpkin': 'Pumpkin', 'pistachio': 'Pistachio', 'raisin': 'Raisin'}
    
    if base_m == 'RF':
        long_base_m = 'Random Forest'
        color_dict = {'Random Forest (RF, No Bias)':'gray', 'Random Forest (RF, Bias)':'#DDAA33', 'Self-Training (ST, Bias)':'#bdd7e7', 'Balanced Self-Training (BST, Bias)':'#bdd7e7', 'Diverse BST 10 (DBST-10, Bias)':'#6baed6', 'Diverse BST 100 (DBST-100, Bias)':'#3182bd'}
        bp_color_dict = {'Random Forest (RF, No Bias)':'gray', 'Random Forest (RF, Bias)':'#DDAA33', 'Self-Training (ST, Bias)':'#6baed6', 'Balanced Self-Training (BST, Bias)':'#6baed6', 'Diverse BST 10 (DBST-10, Bias)':'#3182bd', 'Diverse BST 100 (DBST-100, Bias)':'#08519c'}
    elif base_m == 'NN':
        long_base_m = 'Neural Network'
        color_dict = {'Neural Network (NN, No Bias)':'gray', 'Neural Network (NN, Bias)':'#DDAA33', 'Self-Training (ST, Bias)':'#bae4b3', 'Balanced Self-Training (BST, Bias)':'#bae4b3', 'Diverse BST 10 (DBST-10, Bias)':'#74c476', 'Diverse BST 100 (DBST-100, Bias)':'#31a354',
                     f'NN Seed (NNS, Bias)':'#DDAA99', f'DBST Seed 100 (DBSTS-100, Bias)':'#31a399', }
        bp_color_dict = {'Neural Network (NN, No Bias)':'gray', 'Neural Network (NN, Bias)':'#DDAA33', 'Self-Training (ST, Bias)':'#006d2c', 'Balanced Self-Training (BST, Bias)':'#006d2c', 'Diverse BST 10 (DBST-10, Bias)':'#006d2c', 'Diverse BST 100 (DBST-100, Bias)':'#31a354',
                        f'NN Seed (NNS, Bias)': '#DDAA99', f'DBST Seed 100 (DBSTS-100, Bias)':'#31a399', }
    '''
    color_dict = {f'Random Forest (RF, No Bias)':'gray', 'Random Forest (RF, Bias)':'#DDAA33', 'Self-Training (ST, Bias)':'#bdd7e7', 
                  'Diverse ST 10 (DST-10, Bias)':'#6baed6', 'Diverse ST 100 (DST-100, Bias)':'#3182bd', 'KMM-RF':'#08519c',
            f'Neural Network (NN, No Bias)':'gray', 'Neural Network (NN, Bias)':'#DDAA33', 'Self-Training (ST, Bias)':'#bae4b3', 
                  'Diverse ST 10 (DST-10, Bias)':'#74c476', 'Diverse ST 100 (DST-100, Bias)':'#31a354', 'KMM-NN':'#006d2c',
            'Biased-LR':'#DDAA33', 'KMM-LR':'#fcbba1', 'KDE-LR':'#fc9272', 'RBA-LR':'#fb6a4a', 'FLDA-LR':'#ef3b2c', 'TCPR-LR':'#cb181d', 'SUBA-LR':'#99000d'}
    
    bp_color_dict = {f'Random Forest (RF, No Bias)':'gray',  'Random Forest (RF, Bias)':'#DDAA33', 'Self-Training (ST, Bias)':'#08519c', 
                     'Diverse ST 10 (DST-10, Bias)':'#08519c', 'Diverse ST 100 (DST-100, Bias)':'#08519c', 'KMM-RF':'#08519c',
                f'Neural Network (NN, No Bias)':'gray', 'Neural Network (NN, Bias)':'#DDAA33', 'Self-Training (ST, Bias)':'#006d2c', 
                     'Diverse ST 100 (DST-100, Bias)':'#006d2c', 'Diverse ST 100 (DST-100, Bias)':'#006d2c', 'KMM-NN':'#006d2c',
                'Biased-LR':'#DDAA33', 'KMM-LR':'#99000d', 'KDE-LR':'#99000d', 'RBA-LR':'#99000d', 'FLDA-LR':'#99000d', 'TCPR-LR':'#99000d', 'SUBA-LR':'#99000d'}
    '''
    test_res = {}
    for dataset in datasets:
        print(f'{dataset}')
        if args.bias=='dirichlet':
            bias['n']=args.bias_size*2
            if dataset=='mnist':
                bias['n']=args.bias_size*10
        model_dict, out_loc = create_model_dict(dataset)
        ds_folder_name_list = [model_dict["params"]["dataset"]["name"]]
        [ds_folder_name_list.append(str(val)) for val in model_dict["params"]["dataset"]["args"].values()]
        ds_folder_name = '_'.join(ds_folder_name_list)

        none_model_dict = {'models':{}, 'params':{}}
        none_model_dict['params']['bias'] = {'name':None}
        none_bias_str = f'{none_model_dict["params"]["bias"]["name"]}({"|".join([str(val) for key, val in none_model_dict["params"]["bias"].items() if "name" not in key])})'
        none_out_loc = f'{none_bias_str}_{"_".join(out_loc.split("_")[1:])}'
        try:
            none_df = get_res(f'{none_model_dict["params"]["bias"]["name"]}', args.res_fold[:-1], ds_folder_name, none_out_loc)
        except:
            tmp, none_full_out_loc = create_model_dict(dataset, isnone=True)
            none_out_loc = f'{none_bias_str}_{"_".join(none_full_out_loc.split("_")[1:])}'
            none_df = get_res(f'{none_model_dict["params"]["bias"]["name"]}', args.res_fold[:-1], ds_folder_name, none_out_loc)
        none_su = none_df[none_df['Method']=='SU']
        none_su['Method'] = 'No Bias'#'None_' + none_su['Method'].astype(str)
        none_su['Dataset'] = ds_name_fix[dataset]#'None_' + none_su['Method'].astype(str)
        

        #random_model_dict = {'models':{}, 'params':{}}
        #random_model_dict['params']['bias'] = {'name':'random', 'y': True, 'size':args.bias_size}
        #random_bias_str = f'{random_model_dict["params"]["bias"]["name"]}({"|".join([str(val) for key, val in random_model_dict["params"]["bias"].items() if "name" not in key])})'
        #random_out_loc = f'{random_bias_str}_{"_".join(out_loc.split("_")[1:])}'
        #random_df = get_res(f'{random_model_dict["params"]["bias"]["name"]}', ds_folder_name, random_out_loc)
        #random_su = random_df[random_df['Method']=='SU']
        #random_su['Method'] = 'Random\nSelection'#'Random_' + random_su['Method'].astype(str)
        #random_su['Dataset'] = ds_name_fix[dataset]#'None_' + none_su['Method'].astype(str)

        hierarchy_df = get_res(f'{model_dict["params"]["bias"]["name"]}', f'{args.res_fold}', ds_folder_name, out_loc+'_es=True')
        dict_orig = {"SU": 'Biased', "DST-None": 'Self Training'}#, "DST-10": 'H', "DST-100": 'P'}
        hierarchy_df = hierarchy_df[~np.isin(hierarchy_df['Method'],['DST-None', 'DST-10', 'DST-50', 'DST-200'])]
        hierarchy_df = hierarchy_df.replace({"Method": dict_orig})
        
        hierarchy_var_df = get_res(f'{model_dict["params"]["bias"]["name"]}', f'{args.res_fold}_seedvar', ds_folder_name, out_loc+'_es=True')
        hierarchy_var_df = hierarchy_var_df[~np.isin(hierarchy_var_df['Method'],[ 'DST-None', 'DST-10', 'DST-50', 'DST-200'])]
        dict_var = {"SU": 'Biased Seed', "DST-100": 'DST-100 Seed'}
        hierarchy_var_df = hierarchy_var_df.replace({"Method": dict_var})

        total_res = pd.concat([hierarchy_df.copy(),hierarchy_var_df.copy()])#hierarchy_df.copy()
        total_res['Dataset'] = ds_name_fix[dataset] 
        if version==5: 
            iters_lst = []
            iters_lst.append(total_res[total_res['Method']=='Biased']['Iteration'].unique())
            for comp_stat_ds in ['ST_kb', 'Self Training', 'DST-10', 'DST-100']:
                iters_lst.append(total_res[total_res['Method']==comp_stat_ds]['Iteration'].unique())
            removals = np.setdiff1d(np.arange(30), reduce(np.intersect1d,iters_lst))
            total_res = total_res[~np.isin(total_res['Iteration'],removals)]
        #biased_vals = total_res[total_res['Method']=='Biased'][metric].values
        test_res[dataset] = {}
        biased_iters = np.unique(total_res[total_res['Method']=='Biased']['Iteration'])
        for comp_stat_ds in ['DST-100']:
            comp_stat_ds_iters = np.unique(total_res[total_res['Method']==comp_stat_ds]['Iteration'])
            #print(f"Unique folds for {comp_stat_ds}: {comp_stat_ds_iters}")
            common_iters = np.intersect1d(biased_iters, comp_stat_ds_iters)
            #print(f'Common iters:{common_iters}')
            comp_stat_res = total_res[np.isin(total_res['Iteration'],common_iters)]
            comp_stat_vals=comp_stat_res[comp_stat_res['Method']==comp_stat_ds][metric].values
            comp_stat_biased_vals=comp_stat_res[comp_stat_res['Method']=='Biased'][metric].values
            #print(f'{comp_stat_vals.shape}, {comp_stat_biased_vals}')
            #removals = np.setdiff1d(np.arange(30), reduce(np.intersect1d,iters_lst))
            #total_res = total_res[~np.isin(total_res['Iteration'],removals)]
            #comp_stat_vals=total_res[total_res['Method']==comp_stat_ds][metric].values
            try:
                res = wilcoxon(comp_stat_biased_vals, comp_stat_vals, alternative='two-sided')
                res2 = wilcoxon(comp_stat_biased_vals, comp_stat_vals, alternative='greater')
                res3 = wilcoxon(comp_stat_biased_vals, comp_stat_vals, alternative='less')
            except:
                res = wilcoxon(comp_stat_biased_vals, comp_stat_vals, alternative='two-sided', zero_method='zsplit')
                res2 = wilcoxon(comp_stat_biased_vals, comp_stat_vals, alternative='greater', zero_method='zsplit')
                res3 = wilcoxon(comp_stat_biased_vals, comp_stat_vals, alternative='less', zero_method='zsplit')
                
            test_res[dataset][comp_stat_ds] = {'greater':res2, 'less':res3}
            print(f'Biased vs {comp_stat_ds}: s_t: {res.statistic}\tp: {res.pvalue}')
            print(f'Biased vs {comp_stat_ds}: s_g: {res2.statistic}\tp: {res2.pvalue}')
            print(f'Biased vs {comp_stat_ds}: s_l: {res3.statistic}\tp: {res3.pvalue}')
        
        if version==5: 
            iters_lst = []
            iters_lst.append(total_res[total_res['Method']=='Biased']['Iteration'].unique())
            for comp_stat_ds in ['ST_kb', 'Self Training', 'DST-10', 'DST-100']:
                iters_lst.append(total_res[total_res['Method']==comp_stat_ds]['Iteration'].unique())
            removals = np.setdiff1d(np.arange(30), reduce(np.intersect1d,iters_lst))
            total_res = total_res[~np.isin(total_res['Iteration'],removals)]
        #biased_vals = total_res[total_res['Method']=='Biased'][metric].values
        biased2_iters = np.unique(total_res[total_res['Method']=='Biased Seed']['Iteration'])
        for comp_stat_ds2 in ['DST-100 Seed']:
            comp_stat_ds2_iters = np.unique(total_res[total_res['Method']==comp_stat_ds2]['Iteration'])
            #print(f"Unique folds for {comp_stat_ds}: {comp_stat_ds_iters}")
            common2_iters = np.intersect1d(biased2_iters, comp_stat_ds2_iters)
            #print(f'Common iters:{common_iters}')
            comp_stat2_res = total_res[np.isin(total_res['Iteration'],common2_iters)]
            comp_stat2_vals=comp_stat2_res[comp_stat2_res['Method']==comp_stat_ds2][metric].values
            comp_stat2_biased_vals=comp_stat2_res[comp_stat2_res['Method']=='Biased Seed'][metric].values
            #print(f'{comp_stat_vals.shape}, {comp_stat_biased_vals}')
            #removals = np.setdiff1d(np.arange(30), reduce(np.intersect1d,iters_lst))
            #total_res = total_res[~np.isin(total_res['Iteration'],removals)]
            #comp_stat_vals=total_res[total_res['Method']==comp_stat_ds][metric].values
            try:
                res02 = wilcoxon(comp_stat2_biased_vals, comp_stat2_vals, alternative='two-sided')
                res22 = wilcoxon(comp_stat2_biased_vals, comp_stat2_vals, alternative='greater')
                res32 = wilcoxon(comp_stat2_biased_vals, comp_stat2_vals, alternative='less')
            except:
                res02 = wilcoxon(comp_stat2_biased_vals, comp_stat2_vals, alternative='two-sided', zero_method='zsplit')
                res22 = wilcoxon(comp_stat2_biased_vals, comp_stat2_vals, alternative='greater', zero_method='zsplit')
                res32 = wilcoxon(comp_stat2_biased_vals, comp_stat2_vals, alternative='less', zero_method='zsplit')
                
            test_res[dataset][comp_stat_ds2] = {'greater':res22, 'less':res32}
            print(f'Biased vs {comp_stat_ds2}: s_t: {res.statistic}\tp: {res02.pvalue}')
            print(f'Biased vs {comp_stat_ds2}: s_g: {res2.statistic}\tp: {res22.pvalue}')
            print(f'Biased vs {comp_stat_ds2}: s_l: {res3.statistic}\tp: {res32.pvalue}')
        
        
        a = pd.concat([none_su, total_res])
        all_datasets_lst.append(a)
        #print(a.head())
    all_ds = pd.concat(all_datasets_lst)
    #print(all_ds.head())
    #print(all_ds.shape)
    accepted_methods = ['No Bias', 'Biased', 'DST-100', 'Biased Seed', 'DST-100 Seed']
    all_ds = all_ds[all_ds['Method'].isin(accepted_methods)]
    
    met_sort = {met:metenum for metenum, met in enumerate(accepted_methods)}
    ds_sort = {ds_name_fix[dssname]:idds for idds, dssname in enumerate(datasets)}
    
    all_ds['met_sort'] = all_ds['Method'].map(met_sort)
    all_ds['ds_sort'] = all_ds['Dataset'].map(ds_sort)
    all_ds = all_ds.sort_values(['ds_sort', 'met_sort', 'Iteration'], ascending=[True, True, True])
    
    met_name_dict = {'No Bias': f'{long_base_m} ({base_m}, No Bias)', 'Biased': f'{long_base_m} ({base_m}, Bias)', 'Biased Seed': f'{base_m} Seed ({base_m}S, Bias)',
                     'DST-100': f'Diverse BST 100 (DBST-100, Bias)', 'DST-100 Seed': f'DBST Seed 100 (DBSTS-100, Bias)'}
    all_ds['Method'] = all_ds['Method'].map(met_name_dict)
    
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
    fig = plt.figure(figsize=(7, 4))
    sns.axes_style(style="white", rc=rc) #font='Arial'
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "Arial"
    sbox = sns.boxplot(x="Dataset", y=metric, hue="Method", data=all_ds, showfliers=not swarmed, width=width_bp, palette=bp_color_dict, **PROPS)
    for idd, patch in enumerate(sbox.axes.artists):
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))
    if swarmed:
        sns.stripplot(x="Dataset", y=metric, hue="Method", data=all_ds, jitter=True, dodge=True, palette=color_dict, size=1, zorder=5)
    
    all_ds_min_range = all_ds[metric].min()
    all_ds_max_range = all_ds[metric].max()
    change = (all_ds_max_range-all_ds_min_range)/10
    plt.ylim((all_ds_min_range-change,all_ds_max_range+change))
    grouped_summary = all_ds.groupby(['Dataset', 'Method']).agg({'Accuracy':['mean', 'median', 'std', 'min', 'max'], 
                                                                 'LogLoss':['mean', 'median', 'std', 'min', 'max'],
                                                                 'AUROC':['mean', 'median', 'std', 'min', 'max'],
                                                                 'AUPRC':['mean', 'median', 'std', 'min', 'max']})
    
    adjust_box_widths(fig, 0.9)
    #ax2 = sns.swarmplot(x="Dataset", y="LogLoss", hue="Method", data=all_ds, size=3, color="#2596be") 
    sbox.legend(frameon=False)
    plt.setp(sbox.axes.get_xticklabels(), size=6)
    plt.setp(sbox.axes.get_yticklabels(), size=6)
    sbox.axes.spines['left'].set_linewidth(0.2)
    sbox.axes.tick_params(axis='y', width=0.3, length=1.2, pad=2)
    sbox.axes.spines['bottom'].set_linewidth(0.2)
    sbox.axes.tick_params(axis='x', width=0.3, length=1.2, pad=2)
    #sbox.axes.tick_params(axis="x", labelsize=6, width=0.3)
    #sbox.axes.tick_params(axis="y", labelsize=6, width=0.3)
    
    sbox.axes.spines['left'].set_linewidth(0.2)
    sbox.axes.spines['bottom'].set_linewidth(0.2)
    sbox.axes.spines.right.set_visible(False)
    sbox.axes.spines.top.set_visible(False)
    sbox.axes.set_xlabel('Dataset', family='Arial', fontdict={'weight':'bold', 'size':6.5}, labelpad=1.6)
    sbox.axes.set_ylabel(metric, family='Arial', fontdict={'weight':'bold', 'size':6.5}, labelpad=1.6)
    
    
    x_axis_ticks = sbox.axes.get_xaxis().get_majorticklocs()
    y_axis_range = sbox.axes.get_yaxis().get_majorticklocs()[-1]-sbox.axes.get_yaxis().get_majorticklocs()[0]
    y_change_size = (0.015)*y_axis_range
    y_change_size2 = (0.019)*y_axis_range
    print(sbox.axes.get_yaxis().get_majorticklocs())
    print(sbox.axes.get_xaxis().get_minorticklocs())
    method_size = len(list(met_name_dict.keys()))
    majorwidth = width_bp/method_size
    minorwidth = majorwidth/8
    middle_idx = (method_size-1)/2.0
    start_loc_per_ds = -middle_idx*majorwidth
    print(f'Major: {majorwidth}, Minor: {minorwidth}, Middle: {middle_idx}, Start: {start_loc_per_ds}')
    for ds_t_id, ds_t_name in enumerate(datasets):
        xaxis_loc = x_axis_ticks[ds_t_id]
        print(xaxis_loc)
        ds_df = all_ds[all_ds['Dataset']==ds_name_fix[ds_t_name]]
        biased_df = ds_df[ds_df['Method']==met_name_dict['Biased']][metric]
        biased_middle_xloc = xaxis_loc+start_loc_per_ds+1*majorwidth
        biased_max_yloc = biased_df.max()
        biased_min_yloc = biased_df.min()
        except_df = ds_df[ds_df['Method']!=met_name_dict['No Bias']][metric]
        last_max = -999
        last_min = +999
        for met_t_id, met_t_name in enumerate(['DST-100']): 
            met_middle_xloc = xaxis_loc+start_loc_per_ds+(met_t_id+2)*majorwidth
            print(f'{met_t_id}: {met_middle_xloc}')
            met_df = ds_df[ds_df['Method']==met_name_dict[met_t_name]][metric]
            left_loc = (biased_middle_xloc+(1-met_t_id)*minorwidth)
            right_loc = met_middle_xloc
            met_max_yloc = met_df.max()
            met_min_yloc = met_df.min()
            max_loc = max(last_max, biased_max_yloc, met_max_yloc)+y_change_size
            min_loc = min(last_min, biased_min_yloc, met_min_yloc)-y_change_size 
            top_txt = None
            bot_txt = None
            
            if test_res[ds_t_name][met_t_name]['less'].pvalue<0.01:
                top_txt='**'
            elif test_res[ds_t_name][met_t_name]['less'].pvalue<0.05:
                top_txt='*'
            elif test_res[ds_t_name][met_t_name]['greater'].pvalue<0.01:
                bot_txt='**'
            elif test_res[ds_t_name][met_t_name]['greater'].pvalue<0.05:
                bot_txt='*'
            
            if top_txt is not None:
                plt.text((left_loc+right_loc)/2, max_loc-y_change_size/8, top_txt, horizontalalignment='center', fontsize=5.6, color='red')
                plt.vlines(x=biased_middle_xloc+(1-met_t_id)*minorwidth, ymin=biased_max_yloc+y_change_size/10, ymax=max_loc+y_change_size/10, color='red', linewidth=0.2)
                plt.vlines(x=right_loc, ymin=met_max_yloc+y_change_size/10, ymax=max_loc+y_change_size/10, color='red', linewidth=0.2)
                plt.hlines(y=max_loc+y_change_size/10, xmin=left_loc, xmax=right_loc, color='red', linewidth=0.2)
                last_max = max_loc
            else:
                last_max = met_max_yloc
            if bot_txt is not None:
                plt.text((left_loc+right_loc)/2, min_loc-y_change_size2, bot_txt, horizontalalignment='center', fontsize=5.6, color='red') #Bottom text
                plt.vlines(x=biased_middle_xloc+(1-met_t_id)*minorwidth, ymin=min_loc, ymax=biased_min_yloc-y_change_size/10, color='red', linewidth=0.2) #Left
                plt.vlines(x=right_loc, ymin=min_loc, ymax=met_min_yloc-y_change_size/10, color='red', linewidth=0.2) #Right
                plt.hlines(y=min_loc, xmin=left_loc, xmax=right_loc, color='red', linewidth=0.2) #Bottom
                last_min = min_loc
            else:
                last_min = met_min_yloc
            
        #max_loc=all_ds[all_ds['Dataset']==ds_name_fix[ds_t_name]][metric].max()
        #plt.text(xaxis_loc, max_loc, '**', horizontalalignment='center', fontsize=5.6,
        #                 color='red')#, weight='semibold')
        '''
        plt.text((best2_loc + best1_loc) / 2, -0.24, label_txt, horizontalalignment='center', fontsize=14,
                         color='black')#, weight='semibold')
        plt.hlines(y=-0.25, xmin=best1_loc, xmax=best2_loc, color='red', linewidth=0.5)
        plt.vlines(x=best1_loc, ymin=-0.25, ymax=lowest_best_others - 0.03, color='red', linewidth=0.5)
        plt.vlines(x=best2_loc, ymin=-0.25, ymax=lowest_best_el - 0.03, color='red', linewidth=0.5)
        '''
    leg_met_names = []
    leg_met_handles = []
    for met_name, met_color in color_dict.items():
        if met_name in met_name_dict.values():
            met_bp_color = bp_color_dict[met_name]
            p_tmp = Patch(facecolor=met_bp_color, edgecolor='black', linewidth=0.1, alpha=alpha)
            #c_tmp = mpatches.Circle((0,0),facecolor=met_color, edgecolor=None, linewidth=0.1, radius=0)
            l_tmp = Line2D([], [], marker='o', color='w', linewidth=0.001, markeredgewidth=0, markerfacecolor=met_color, markersize=3)#, label=met_name)
            leg_met_handles.append(tuple([p_tmp, l_tmp]))
            #leg_met_handles.append(l_tmp)
            leg_met_names.append(met_name) 
        

    leg = plt.legend(leg_met_handles, leg_met_names, frameon=False, scatterpoints=1, numpoints=1, 
                     handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)}, fontsize=6, labelspacing=0.2, handletextpad=0.1)#, handleheight=0.2)
    #sbox.axes.legend(handles=leg_met_handles, frameon=False, fontsize=5.6, labelspacing=0.2, handletextpad=0.1)
    #for handle in leg.legendHandles:
    #    print(handle)
    #    
    for patch in leg.get_patches():
        patch.set_height(4)
        patch.set_width(6)
    #    patch.set_y(-0.8)
    #for line in leg.get_lines():
    #    print(line)
    #    line.set_y(-1)
    #v2: With vanilla, v3: without early stopping
    out_loc = res_folder / 'images_comp_seed' / f'Multids_{normal_bias_str}_{metric}_{args.threshold}_{args.val_base}h{mutids_name}_s={swarmed}_ss_v{version}'
    config.ensure_dir(out_loc)
    #plt.setp(sbox.legend_.get_texts(), fontsize=5.1)
    plt.margins(0,0)
    plt.savefig(f'{out_loc}.png', dpi=300, bbox_inches='tight', pad_inches = 0.01)
    plt.savefig(f'{out_loc}.pdf', dpi=300, bbox_inches='tight', pad_inches = 0.01)
    plt.show()
    
    

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
if 'nn' in args.res_fold:
    basem='NN'
else:
    basem='RF'
#datasets = ['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'spam', 'fire']#'breast_cancer', 
#datasets = ['adult', 'pumpkin', 'raisin', 'pistachio', 'rice', 'fire']#adult
if args.d_group==0:
    datasets = ['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'rice', 'fire']
elif args.d_group==1:
    datasets = ['adult', 'spam', 'raisin', 'pistachio', 'pumpkin', 'fire']
none_hierarchy_multi_datasets(datasets, metric='LogLoss', base_m = basem, swarmed=True, version=4)#model_dict, out_loc)
none_hierarchy_multi_datasets(datasets, metric='Accuracy', base_m = basem, swarmed=True, version=4)#model_dict, out_loc)
#none_hierarchy_multi_datasets(datasets, metric='AUROC')#model_dict, out_loc)
#none_hierarchy_multi_datasets(datasets, metric='AUPRC')#model_dict, out_loc)
