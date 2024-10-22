import os
import sys

path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower() == 'diversepsuedolabeling':
        project_path = '/'.join(path2this[:i + 1])
sys.path.insert(0, project_path)
import numpy as np
import pandas as pd
from src import config
from src.lib import visual_fnc

import argparse

parser = argparse.ArgumentParser(description='RFE arguments')
parser.add_argument('--bias', '-b', metavar='the-bias', dest='bias', type=str, help='Choose bias', default='hierarchyy9')
parser.add_argument('--dataset', '-d', metavar='the-dataset', dest='dataset', type=str, help='Choose dataset', default='mnist')
parser.add_argument('--bias_size', '-bs', metavar='the-bias-size', dest='bias_size', type=int, help='Choose the bias size per class', default=30)
parser.add_argument('--k_best', '-kb', metavar='the-k-best', dest='k_best', type=int, help='Choose the k best per class', default=40)
parser.add_argument('--threshold', '-th', metavar='the-threshold', dest='threshold', type=float, help='Choose the threshold for confidence', default=97)
parser.add_argument('--folder', '-f', metavar='the-result-folder', dest='res_fold', type=str, help='Choose the result folder', default='results_test_nb_imb')#results_nn_test_nb_imb_fin

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

def get_res(bias_name, ds_folder_name, out_loc):
    #try:
    res_loc = config.ROOT_DIR / args.res_fold / f'{bias_name}' / ds_folder_name / f'{out_loc}.csv'
    res_df = pd.read_csv(res_loc)
    #except:
    #    if args.res_fold == 'results_test_nb_imb':
    #        res_loc = config.ROOT_DIR / 'resultsxxx' / f'{bias_name}' / ds_folder_name / f'{out_loc}.csv'
    #        res_df = pd.read_csv(res_loc)
    return res_df


def none_random_hierarchy(model_dict, out_loc):
    ds_folder_name_list = [model_dict["params"]["dataset"]["name"]]
    [ds_folder_name_list.append(str(val)) for val in model_dict["params"]["dataset"]["args"].values()]
    ds_folder_name = '_'.join(ds_folder_name_list)
    res_folder = config.ROOT_DIR / args.res_fold / f'{model_dict["params"]["bias"]["name"]}' / ds_folder_name

    none_model_dict = {'models':{}, 'params':{}}
    none_model_dict['params']['bias'] = {'name':None}
    none_bias_str = f'{none_model_dict["params"]["bias"]["name"]}({"|".join([str(val) for key, val in none_model_dict["params"]["bias"].items() if "name" not in key])})'
    none_out_loc = f'{none_bias_str}_{"_".join(out_loc.split("_")[1:])}'
    none_df = get_res(f'{none_model_dict["params"]["bias"]["name"]}', ds_folder_name, none_out_loc)
    none_su = none_df[none_df['Method']=='SU']
    none_su['Method'] = 'None_' + none_su['Method'].astype(str)

    random_model_dict = {'models':{}, 'params':{}}
    random_model_dict['params']['bias'] = {'name':'random', 'y': True, 'size':bias_per_class}
    random_bias_str = f'{random_model_dict["params"]["bias"]["name"]}({"|".join([str(val) for key, val in random_model_dict["params"]["bias"].items() if "name" not in key])})'
    random_out_loc = f'{random_bias_str}_{"_".join(out_loc.split("_")[1:])}'
    random_df = get_res(f'{random_model_dict["params"]["bias"]["name"]}', ds_folder_name, random_out_loc)
    random_su = random_df[random_df['Method']=='SU']
    random_su['Method'] = 'Random_' + random_su['Method'].astype(str)

    hierarchy_df = get_res(f'{model_dict["params"]["bias"]["name"]}', ds_folder_name, out_loc)

    total_res = hierarchy_df.copy()
    total_res = total_res[~np.isin(total_res['Method'],['DST-50', 'DST-100'])]
    a = pd.concat([none_su, random_su, total_res])

    png_bp_loss_loc = res_folder / 'images_comp' / f'None_random_{out_loc}_loss2.png'
    png_bp_loss_match_loc = res_folder / 'images_comp' / f'None_random_{out_loc}_loss_match2.png'
    png_bp_acc_loc = res_folder / 'images_comp' / f'None_random_{out_loc}_acc2.png'
    png_bp_acc_match_loc = res_folder / 'images_comp' / f'None_random_{out_loc}_acc_match2.png'
    config.ensure_dir(png_bp_loss_loc)
    visual_fnc.call_plot('boxplot_sign', png_bp_loss_loc, a, model_dict, metric='LogLoss')
    visual_fnc.call_plot('boxplot_match', png_bp_loss_match_loc, a, model_dict, metric='LogLoss')
    visual_fnc.call_plot('boxplot_sign', png_bp_acc_loc, a, model_dict, metric='Accuracy')
    visual_fnc.call_plot('boxplot_match', png_bp_acc_match_loc, a, model_dict, metric='Accuracy')


def none_hierarchy(model_dict, out_loc):
    ds_folder_name_list = [model_dict["params"]["dataset"]["name"]]
    [ds_folder_name_list.append(str(val)) for val in model_dict["params"]["dataset"]["args"].values()]
    ds_folder_name = '_'.join(ds_folder_name_list)
    res_folder = config.ROOT_DIR / args.res_fold / f'{model_dict["params"]["bias"]["name"]}' / ds_folder_name

    none_model_dict = {'models':{}, 'params':{}}
    none_model_dict['params']['bias'] = {'name':None}
    none_bias_str = f'{none_model_dict["params"]["bias"]["name"]}({"|".join([str(val) for key, val in none_model_dict["params"]["bias"].items() if "name" not in key])})'
    none_out_loc = f'{none_bias_str}_{"_".join(out_loc.split("_")[1:])}'
    none_df = get_res(f'{none_model_dict["params"]["bias"]["name"]}', ds_folder_name, none_out_loc)
    none_su = none_df[none_df['Method']=='SU']
    none_su['Method'] = 'No\nBias'#'None_' + none_su['Method'].astype(str)

    random_model_dict = {'models':{}, 'params':{}}
    random_model_dict['params']['bias'] = {'name':'random', 'y': True, 'size':bias_per_class}
    random_bias_str = f'{random_model_dict["params"]["bias"]["name"]}({"|".join([str(val) for key, val in random_model_dict["params"]["bias"].items() if "name" not in key])})'
    random_out_loc = f'{random_bias_str}_{"_".join(out_loc.split("_")[1:])}'
    random_df = get_res(f'{random_model_dict["params"]["bias"]["name"]}', ds_folder_name, random_out_loc)
    random_su = random_df[random_df['Method']=='SU']
    random_su['Method'] = 'Random\nSelection'#'Random_' + random_su['Method'].astype(str)

    hierarchy_df = get_res(f'{model_dict["params"]["bias"]["name"]}', ds_folder_name, out_loc)

    total_res = hierarchy_df.copy()
    total_res = total_res[~np.isin(total_res['Method'],['DST-50', 'DST-200'])]
    dict = {"SU": 'Biased', "DST-None": 'Self\nTraining'}#, "DST-10": 'H', "DST-100": 'P'}
    total_res = total_res.replace({"Method": dict})
    #a = pd.concat([none_su, random_su, total_res])
    a = pd.concat([none_su, total_res])
    print(a)

    png_bp_loss_loc = res_folder / 'images_comp' / f'None_{out_loc}_loss.png'
    png_bp_loss_match_loc = res_folder / 'images_comp' / f'None_{out_loc}_loss_match.png'
    png_bp_acc_loc = res_folder / 'images_comp' / f'None_{out_loc}_acc.png'
    png_bp_acc_match_loc = res_folder / 'images_comp' / f'None_{out_loc}_acc_match.png'
    print(png_bp_acc_match_loc)
    config.ensure_dir(png_bp_loss_loc)
    visual_fnc.call_plot('boxplot_sign', png_bp_loss_loc, a, model_dict, metric='LogLoss')
    visual_fnc.call_plot('boxplot_match', png_bp_loss_match_loc, a, model_dict, metric='LogLoss')
    visual_fnc.call_plot('boxplot_sign', png_bp_acc_loc, a, model_dict, metric='Accuracy')
    visual_fnc.call_plot('boxplot_match', png_bp_acc_match_loc, a, model_dict, metric='Accuracy')

    
bias_per_class = args.bias_size
#dataset = {'name': args.dataset,'args':{}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.3, 'test': 0.25, 'unlabeled': 0.45, 'val': 0.2, 'runs': 30}
dataset = {'name': args.dataset,'args':{}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
if 'drug_' in args.dataset:
    dataset = {'name': 'drug','args':{'drug': args.dataset.split('_')[1]}, 'order': 'train_test_unlabeled_bias_validation',
                                                       'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
if args.bias=='hierarchyy9':
    bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.9} 
if args.bias=='hierarchyy8':
    bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.8} 
if 'hierarchyy_' in args.bias:
    bias = {'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': float(args.bias.split('_')[1])}
th, kb, mi, balance, val_base = args.threshold, args.k_best, 100, 'ratio', True
if th>1:
    th=int(th)
if args.dataset == 'mnist':
    #dataset = {'name': args.dataset, 'args': {}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.35,
    #           'test': 0.25, 'unlabeled': 0.4, 'val': 0.2, 'runs': 30}
    dataset = {'name': args.dataset, 'args': {}, 'order': 'train_test_unlabeled_bias_validation', 'train': 0.30, 'test': 0.2, 'unlabeled': 0.70, 'val': 0.2, 'runs': 30}
    #th=85
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

none_hierarchy(model_dict, out_loc)