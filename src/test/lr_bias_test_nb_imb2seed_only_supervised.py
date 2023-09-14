R_STATE = 123
import os
os.environ['PYTHONHASHSEED']=str(0)
import sys

path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower() == 'diversepsuedolabeling':
        project_path = '/'.join(path2this[:i + 1])
sys.path.insert(0, project_path)

import numpy as np
np.random.seed(R_STATE)

import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, confusion_matrix, ConfusionMatrixDisplay, log_loss, roc_auc_score, accuracy_score
from sklearn import preprocessing
from src import load_dataset as ld
from src import bias_techniques as bt
import lightgbm as lgb
from src.lib.sutils import *
from src import config
from src.lib import visual_fnc
import random
import optuna
from optuna.integration import LightGBMPruningCallback
import traceback
from libtlda.rba import RobustBiasAwareClassifier
from sklearn.linear_model import LogisticRegression
from libtlda.iw import ImportanceWeightedClassifier
from libtlda.flda import FeatureLevelDomainAdaptiveClassifier
from libtlda.tcpr import TargetContrastivePessimisticClassifier
from libtlda.suba import SubspaceAlignedClassifier

import argparse

parser = argparse.ArgumentParser(description='RFE arguments')
parser.add_argument('--bias', '-b', metavar='the-bias', dest='bias', type=str,
                    help='Choose bias', default='cluster')
parser.add_argument('--dataset', '-d', metavar='the-dataset', dest='dataset', type=str,
                    help='Choose dataset', default='breast_cancer')
parser.add_argument('--bias_size', '-bs', metavar='the-bias-size', dest='bias_size', type=int,
                    help='Choose the bias size per class', default=50)
parser.add_argument('--early_stop', '-es', metavar='the-early-stop', dest='early_stop', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()
print(f'Running args:{args}')

#R_STATE = 123

param_grid = {
    # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    "n_estimators": 10000,
    "learning_rate": [0.01, 0.3],
    "num_leaves": np.arange(20, 3000, 20),
    "max_depth": [3, 12],
    "min_data_in_leaf": np.arange(200, 10000, 100),
    "lambda_l1": np.arange(5, 100, 5),
    "lambda_l2": np.arange(5, 100, 5),
    "min_gain_to_split": np.arange(0, 15),
    "bagging_fraction": np.arange(0.2, 0.95, 0.1),
    "bagging_freq": 1,
    "feature_fraction": np.arange(0.2, 0.95, 0.1),
    "class_weight": 'balanced',
    "random_state": R_STATE,
}

param_grid = {
    # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    "n_estimators": 10000,
    "learning_rate": [0.01, 0.3],
    "num_leaves": np.arange(20, 3000, 20),
    "max_depth": [3, 12],
    "min_data_in_leaf": np.arange(200, 10000, 100),
    "lambda_l1": np.arange(5, 100, 5),
    "lambda_l2": np.arange(5, 100, 5),
    "min_gain_to_split": np.arange(0, 15),
    "bagging_fraction": np.arange(0.2, 0.95, 0.1),
    "bagging_freq": 1,
    "feature_fraction": np.arange(0.2, 0.95, 0.1),
    "class_weight": 'balanced',
    "random_state": R_STATE,
}

params = {
    # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    "subsample": 0.9, "subsample_freq": 1,
    #"min_child_weight":0.01,
    "reg_lambda": 5,
    "class_weight": 'balanced',
    "random_state": R_STATE, "verbose": -1, "n_jobs":-1, 
}


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

@timeit
def selection_bias_trial(model_dict, res_folder, out_loc):
    ds_folder_name_list = [model_dict["params"]["dataset"]["name"]]
    [ds_folder_name_list.append(str(val)) for val in model_dict["params"]["dataset"]["args"].values()]
    ds_folder_name = '_'.join(ds_folder_name_list)
    res_folder = config.ROOT_DIR / res_folder / f'{model_dict["params"]["bias"]["name"]}' / ds_folder_name
    res_loc = res_folder / f'{out_loc}.csv'
    config.ensure_dir(res_loc)
    res_dict_loc = res_folder / f'{out_loc}.pkl'
    config.ensure_dir(res_dict_loc)
    png_bar_loss_loc = res_folder / 'images_bar' / f'{out_loc}_lossy.png'
    config.ensure_dir(png_bar_loss_loc)
    png_bp_loss_loc = res_folder / 'images_bp' / f'{out_loc}_lossy.png'
    config.ensure_dir(png_bp_loss_loc)
    png_bar_auprc_loc = res_folder / 'images_bar' / f'{out_loc}_auprcy.png'
    png_bp_auprc_loc = res_folder / 'images_bp' / f'{out_loc}_auprcy.png'
    png_bar_auroc_loc = res_folder / 'images_bar' / f'{out_loc}_aurocy.png'
    png_bp_auroc_loc = res_folder / 'images_bp' / f'{out_loc}_aurocy.png'
    png_bar_acc_loc = res_folder / 'images_bar' / f'{out_loc}_accy.png'
    png_bp_acc_loc = res_folder / 'images_bp' / f'{out_loc}_accy.png'
    res_df = None
    if os.path.exists(res_loc):
        res_df = pd.read_csv(res_loc)
    elif os.path.exists(res_dict_loc):
        res_dict_loc = config.load_pickle(res_dict_loc)
        res = []
        for fold, fold_res in res_dict_loc['models'].items():
            for seed_run, seed_res in fold_res.items():
                for model_name, model_res in seed_res.items():
                    try:
                        res.append([f'{model_name}', seed_res['SU']['sample_size'], seed_res['end_sample_size'],
                                    fold, seed_run, model_res['loss'], model_res['auroc'], model_res['auprc'], model_res['acc']])
                    except:
                        res.append([f'{model_name}', seed_res['SU']['sample_size'], seed_res['SU']['sample_size'],
                                    fold, seed_run, model_res['loss'], model_res['auroc'], model_res['auprc'], model_res['acc']])
        res_df = pd.DataFrame(res, columns=['Method', 'Start_size', 'EndSize', 'Iteration', 'Seed Run', 'LogLoss', 'AUROC', 'AUPRC', 'Accuracy'])
        res_df = pd.DataFrame(res, columns=['Method', 'Start_size', 'EndSize', 'Iteration', 'Seed Run', 'LogLoss', 'AUROC', 'AUPRC', 'Accuracy'])
        res_df.to_csv(res_loc, index=False)
    else:
        X, y, X_main_test, y_main_test = ld.load_dataset(model_dict['params']['dataset']['name'], **model_dict['params']['dataset']['args'], test=True)
        X, X_main_test = np.nan_to_num(X), np.nan_to_num(X_main_test)
        all_X, all_y = np.concatenate((X, X_main_test)), np.concatenate((y, y_main_test))
        #X, y = ld.load_dataset(model_dict['params']['dataset']['name'], **model_dict['params']['dataset']['args'])
        res = []
        for fold in range(model_dict['params']['dataset']['runs']):
            try:
                X_train, X_test, y_train, y_test = split_dataset(all_X, all_y, train_ratio=0.8, r_seed=R_STATE + fold)
                model_dict['models'][f'{fold}'] = {}
                if model_dict['params']['bias']["name"] is not None:
                    bias_params = {key: val for key, val in model_dict['params']['bias'].items()
                                   if ('name' not in key) and ('y' != key)}
                    if 'y' in model_dict['params']['bias'] and model_dict['params']['bias']['y']:
                        selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X_train, y=y_train,
                                                   **bias_params).astype(int)
                    else:
                        selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X=X_train, **bias_params).astype(int)
                    mask = np.ones_like(y_train, bool)
                    mask[selected_ids] = False
                    X_b_train, y_b_train = X_train[selected_ids, :], y_train[selected_ids]
                else:
                    X_b_train, y_b_train = X_train.copy(), y_train.copy()
                if model_dict['params']['dataset']['name'] in ['ppi', 'cora', 'citeseer', 'webkb']:
                    X_b_train = X_b_train[:,2:]
                print(f'{sum(y_b_train == 1)} pos and {sum(y_b_train == 0)} neg samples for run {fold}')
                log(f'{sum(y_b_train == 1)} pos and {sum(y_b_train == 0)} neg samples for run {fold}')
                x_b_train, x_b_val, y_b_train, y_b_val = split_dataset(X_b_train, y_b_train,
                                                                       train_ratio=1 - model_dict['params']['dataset'][
                                                                           'val'], r_seed=R_STATE + fold)
                y_b_train_val = np.concatenate((y_b_train, y_b_val))
                if model_dict['params']['model']['val_base']:
                    zv_cols = np.var(x_b_train, axis=0)==0 #zero variance columns
                else:
                    zv_cols = np.var(X_b_train, axis=0)==0
                #zv_cols = np.var(x_b_train, axis=0)==0 #zero variance columns
                x_b_train, x_b_val, X_b_train, X_test = x_b_train[:, ~zv_cols], x_b_val[:, ~zv_cols], X_b_train[:, ~zv_cols], X_test[:, ~zv_cols]
                
                x_b_train_val = np.concatenate((x_b_train, x_b_val))
            
                if model_dict['params']['model']['val_base']:
                    scaler = preprocessing.StandardScaler().fit(x_b_train)
                else:
                    scaler = preprocessing.StandardScaler().fit(x_b_train_val)
                #scaler = MinMaxScaler().fit(x_b_train)
                x_b_train_val = scaler.transform(x_b_train_val)
                x_b_train = scaler.transform(x_b_train)
                x_b_val = scaler.transform(x_b_val)
                X_test = scaler.transform(X_test)
                print(f'train: {x_b_train.shape}, test: {X_test.shape}')
                print(f'Number of labeled/unlabeled samples at the beginning of run {fold}:{len(y_b_train)}')
                log(f'Number of labeled/unlabeled samples at the beginning of run {fold}:{len(y_b_train)}')
                y_orig_test = y_test.copy()
                y_test = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).toarray()
                
                for seed_run in range(10):
                    model_dict['models'][f'{fold}'][f'{seed_run}'] = {}
                    base_m_params = {key: val for key, val in model_dict['params']['base_model'].items() if 'name' not in key}
                    np.random.seed(R_STATE+fold+seed_run)
                    su_clf = LogisticRegression(multi_class='auto', penalty='l2', max_iter=1000, class_weight='balanced', random_state=R_STATE+fold+seed_run, C=5)

                    if model_dict['params']['model']['val_base']:
                        su_clf.fit(x_b_train.copy(), y_b_train.copy())
                    else:
                        su_clf.fit(x_b_train_val.copy(), y_b_train_val.copy())

                    probs_su_test = su_clf.predict_proba(X_test.copy())
                    preds_su_test = su_clf.predict(X_test.copy())
                    auroc_su = roc_auc_score(y_test, probs_su_test)#[:, 1])
                    auprc_su = average_precision_score(y_test, probs_su_test)#[:, 1])
                    acc_su = accuracy_score(y_orig_test, preds_su_test)
                    if acc_su<0.5:
                        warnings.warn(f"Sorry, the Accuracy score for start is too low: {acc_su}", UserWarning)
                        continue
                    loss_su = log_loss(y_test, probs_su_test)
                    model_dict['models'][f'{fold}'][f'{seed_run}'][f'SU'] = {'model': su_clf, 'loss': loss_su, 'auroc': auroc_su, 'auprc': auprc_su, 'acc': acc_su,
                                                              'sample_size': len(y_b_train)}
                    res.append(['SU', len(y_b_train), len(y_b_train), fold, seed_run, loss_su, auroc_su, auprc_su, acc_su])

            except Exception as e1:
                print(f"Sorry, this fold couldn't run because of: {e1}")
                log(f"Sorry, this fold couldn't run because of: {e1}")
        if len(res)==0:
            raise Exception(f"Sorry, the AUROC score for start is too low: {auroc_su}")
            log(f'Sorry, all the AUROC score for start is too low. Last one: {auroc_su}')
        config.save_pickle(res_dict_loc, model_dict)
        res_df = pd.DataFrame(res, columns=['Method', 'Start_size', 'EndSize', 'Iteration', 'Seed Run', 'LogLoss', 'AUROC', 'AUPRC', 'Accuracy'])
        res_df.to_csv(res_loc, index=False)

    visual_fnc.call_plot('boxplot', png_bp_loss_loc, res_df, model_dict, metric='LogLoss')
    visual_fnc.call_plot('bar', png_bar_loss_loc, res_df, model_dict, metric='LogLoss')
    visual_fnc.call_plot('boxplot', png_bp_auroc_loc, res_df, model_dict, metric='AUROC')
    visual_fnc.call_plot('bar', png_bar_auroc_loc, res_df, model_dict, metric='AUROC')
    visual_fnc.call_plot('boxplot', png_bp_auprc_loc, res_df, model_dict, metric='AUPRC')
    visual_fnc.call_plot('bar', png_bar_auprc_loc, res_df, model_dict, metric='AUPRC')
    visual_fnc.call_plot('boxplot', png_bp_acc_loc, res_df, model_dict, metric='Accuracy')
    visual_fnc.call_plot('bar', png_bar_acc_loc, res_df, model_dict, metric='Accuracy')


kb_list = [6]#, 10]#[3, 6, 10]
bias_per_class = args.bias_size
unique_class=2
if args.dataset in ['cifar', 'mnist']:
    unique_class=10
    kb_list = [30]#[20, 30, 40]
if args.dataset in ['wine_uci']:
    unique_class=3
    kb_list = [9, 18, 30]

if args.dataset == 'cifar':
    dataset = {'name': args.dataset, 'args':{'class_size':10}, 'order': 'test_train_unlabeled_bias_validation',
                                                       'train': 0.8, 'test': 0.2, 'val': 0.2, 'runs': 30}
elif args.dataset == 'mnist':
    dataset = {'name': args.dataset, 'args':{}, 'order': 'test_train_unlabeled_bias_validation',
                                                       'train': 0.8, 'test': 0.2, 'val': 0.2, 'runs': 30}
elif 'ppi_' in args.dataset:
    dataset = {'name': 'ppi','args':{'id': int(args.dataset.split('_')[1])}, 'order': 'test_train_unlabeled_bias_validation',
                                                       'train': 0.8, 'test': 0.2, 'val': 0.2, 'runs': 30}
elif args.dataset in ['cora', 'citeseer']:
    dataset = {'name': args.dataset, 'args':{'chosen_class': 'Neural_Networks'}, 'order': 'test_train_unlabeled_bias_validation',
                                                       'train': 0.8, 'test': 0.2, 'val': 0.2, 'runs': 30}
elif 'drug_' in args.dataset:
    dataset = {'name': 'drug','args':{'drug': args.dataset.split('_')[1]}, 'order': 'train_test_unlabeled_bias_validation',
                                                       'train': 0.3, 'test': 0.2, 'unlabeled': 0.7, 'val': 0.2, 'runs': 30}
elif 'webkb_' in args.dataset:
    if len(args.dataset.split('_')) == 2:
        dataset = {'name': 'webkb','args':{'source': args.dataset.split('_')[1], 'chosen_class': 'all', 'pca':None}, 'order': 'test_train_unlabeled_bias_validation',
                                                       'train': 0.8, 'test': 0.2, 'val': 0.2, 'runs': 30}
    if len(args.dataset.split('_')) == 3:
        dataset = {'name': 'webkb','args':{'source': args.dataset.split('_')[1], 'chosen_class': 'all', 'pca':int(args.dataset.split('_')[2])}, 'order': 'test_train_unlabeled_bias_validation',
                                                       'train': 0.8, 'test': 0.2, 'val': 0.2, 'runs': 30} 
else:
    dataset = {'name': args.dataset,'args':{}, 'order': 'test_train_unlabeled_bias_validation',
                                                       'train': 0.3, 'test': 0.2, 'val': 0.2, 'runs': 30}
bias_list = []
if args.bias == 'cluster':
    bias_list.append({'name': 'cluster', 'y': True, 'k': 3, 'n_k': 1})
if args.bias == 'hierarchy':
    bias_list.append({'name': 'hierarchy', 'y': True, 'max_size': bias_per_class, 'prob': 0.9})
if args.bias == 'hierarchyy7':
    bias_list.append({'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.7})
if args.bias == 'hierarchyy8':
    bias_list.append({'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.8})
if args.bias == 'hierarchyy9':
    bias_list.append({'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': 0.9})
if 'hierarchyy_' in args.bias:
    bias_list.append({'name': 'hierarchyy', 'y': True, 'max_size': bias_per_class, 'prob': float(args.bias.split('_')[1])})
if args.bias == 'entity':
    bias_list.append({'name': 'entity', 'y': True, 'chosen_entity_size': 4, 'max_size': bias_per_class, 'prob': 0.8, 'dominant_class':1}) 
if args.bias == 'none':
    bias_list.append({'name': None})
if args.bias == 'random':
    bias_list.append({'name': 'random', 'y': True, 'size':bias_per_class})
if args.bias == 'joint':
    bias_list.append({'name': 'joint'})
if args.bias == 'dirichlet':
    bias_list.append({'name': 'dirichlet', 'n':bias_per_class*unique_class})
    
for bias in bias_list:
    for th in [0.9]:#, 85, 0.9]:
        for balance in ['ratio']:
            for kb in kb_list:#[3, 6, 10, 20]:#[3,6,10]: 
                for mi in [100]:
                    for val_base in [False]:#, True]:
                        model_dict = {'models': {}, 'params': {}}
                        model_dict['params']['bias'] = bias
                        model_dict['params']['dataset'] = dataset
                        model_dict['params']['base_model'] = params
                        model_dict['params']['base_model']['name'] = 'BRRF1'
                        model_dict['params']['model'] = {'name': 'DST', 'threshold': th, 'k_best': kb, 'max_iter': mi, 'balance': balance, 'val_base': val_base}
                        model_dict['params']['model']['full_name'] = f'{model_dict["params"]["model"]["name"]}' \
                                                                     f'-{model_dict["params"]["base_model"]["name"]}'
                        bias_str = f'{model_dict["params"]["bias"]["name"]}' \
                                   f'({"|".join([str(val) for key, val in model_dict["params"]["bias"].items() if "name" not in key])})'

                        ds_str = f'{model_dict["params"]["dataset"]["name"]}' \
                                 f'({model_dict["params"]["dataset"]["train"]}' \
                                 f'|{model_dict["params"]["dataset"]["val"]}' \
                                 f'|{model_dict["params"]["dataset"]["test"]})'
                        model_str = f'supervised' \
                                    f'|vb={model_dict["params"]["model"]["val_base"]})'
                        out_loc = f'{bias_str}_{ds_str}_{model_str}_{model_dict["params"]["dataset"]["runs"]}_es={args.early_stop}'
                        try:
                            ds_folder_name_list = [model_dict["params"]["dataset"]["name"]]
                            [ds_folder_name_list.append(str(val)) for val in model_dict["params"]["dataset"]["args"].values()]
                            ds_folder_name = '_'.join(ds_folder_name_list)
                            res_fold = 'results_lr_fsd_test_nb_imb_ss8'
                            log_loc = res_folder = config.ROOT_DIR / res_fold / f'{model_dict["params"]["bias"]["name"]}' / ds_folder_name / f'{out_loc}.log'
                            change_log_path(log_loc)
                            selection_bias_trial(model_dict, res_fold, out_loc)
                        except Exception as e:
                            traceback.print_exc()
                            print(f'Not successful:{e} for {out_loc}')
                            log(f'Not successful:{e} for {out_loc}')
