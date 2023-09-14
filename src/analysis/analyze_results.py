import os
import sys

path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower() == 'diversepsuedolabeling':
        project_path = '/'.join(path2this[:i + 1])
sys.path.insert(0, project_path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from src.models.self_training import StandardSelfTraining
from src.models.free_self_training import FreeSelfTraining
from src.models.free_self_diverse_training import FreeDiverseSelfTraining
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, confusion_matrix, ConfusionMatrixDisplay, log_loss, roc_auc_score
from src import load_dataset as ld
from src import bias_techniques as bt
import lightgbm as lgb
from src import config
from src.lib import visual_fnc
import random
import optuna
from optuna.integration import LightGBMPruningCallback
import umap

import argparse

parser = argparse.ArgumentParser(description='RFE arguments')
parser.add_argument('--bias', '-b', metavar='the-bias', dest='bias', type=str,
                    help='Choose bias', default='cluster')
parser.add_argument('--dataset', '-d', metavar='the-dataset', dest='dataset', type=str,
                    help='Choose dataset', default='breast_cancer')
parser.add_argument('--task', '-t', metavar='the-task', dest='task', type=str,
                    help='Choose task', default='get_start_size')
args = parser.parse_args()
print(f'Running args:{args}')

R_STATE = 123

params = {
    # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    "subsample": 0.9, "subsample_freq": 1,
    # "min_child_weight":0.01,
    "reg_lambda": 5,
    "class_weight": 'balanced',
    "random_state": R_STATE, "verbose": -1
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


def get_start_size(model_dict, out_loc):
    X, y = ld.load_dataset(model_dict['params']['dataset']['name'], **model_dict['params']['dataset']['args'])
    unique_classes, class_sizes = np.unique(y, return_counts=True)
    start_sizes = dict(zip(unique_classes, class_sizes))
    data_types = ['train', 'test', 'unk', 'biased', 'trainb', 'valb']
    cols = ['fold_id', 'data_type', 'class', 'size']
    # [cols.append(f'all_{unique_class}') for unique_class in unique_classes]
    # for data_type in data_types:
    #    [cols.append(f'{data_type}_{unique_class}') for unique_class in unique_classes]
    res = []
    # start_res = [start_sizes[unique_class] for unique_class in cols]
    # res.append(start_res)
    for fold in range(model_dict['params']['dataset']['runs']):
        p_data = split_dataset(X, y, model_dict['params']['dataset']['train'],
                               model_dict['params']['dataset']['test'],
                               model_dict['params']['dataset']['unlabeled'], r_seed=R_STATE + fold)
        X_train, y_train = p_data['x_train'], p_data['y_train']
        X_test, y_test = p_data['x_test'], p_data['y_test']
        X_unk, y_unk = p_data['x_unk'], p_data['y_unk']
        if model_dict['params']['bias']["name"] is not None:
            print(f'Bias-{model_dict["params"]["bias"]["name"]} started')
            bias_params = {key: val for key, val in model_dict['params']['bias'].items()
                           if ('name' not in key) and ('y' != key)}
            if 'y' in model_dict['params']['bias'] and model_dict['params']['bias']['y']:
                selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X_train, y=y_train,
                                           **bias_params).astype(int)
            else:
                selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X=X_train, **bias_params).astype(int)
            print(f'Bias-{model_dict["params"]["bias"]["name"]} ended')
            X_train = X_train[:,2:]
            mask = np.ones_like(y_train, bool)
            mask[selected_ids] = False
            X_biased, y_biased = X_train[selected_ids, :], y_train[selected_ids]
        else:
            X_biased, y_biased = X_train.copy(), y_train.copy()
        p_data['y_biased'] = y_biased
        x_b_train, x_b_val, y_b_train, y_b_val = split_dataset(X_biased, y_biased,
                                                               train_ratio=1 - model_dict['params']['dataset']['val'],
                                                               r_seed=R_STATE + fold)
        p_data['y_trainb'] = y_b_train
        p_data['y_valb'] = y_b_val
        # p_data['y_run'] = fold

        for data_type in data_types:
            for unique_class in unique_classes:
                res.append([fold, data_type, unique_class, sum(p_data[f'y_{data_type}'] == unique_class)])
        # fold_res = [sum(p_data[f'y_{col.split("_")[0]}'] == int(col.split('_')[-1])) for col in cols]
        # res.append(fold_res)
    df = pd.DataFrame(res, columns=cols)
    png_loc = config.RESULT_DIR / 'sample_analysis' / f'{out_loc}.png'
    config.ensure_dir(png_loc)
    visual_fnc.save_stacked_bar(df, out_loc=png_loc, title=out_loc)
    print()


def get_bias_umap(model_dict, out_loc):
    X, y = ld.load_dataset(model_dict['params']['dataset']['name'], **model_dict['params']['dataset']['args'])
    for fold in range(model_dict['params']['dataset']['runs']):
        print(f'Fold {fold} started!')
        p_data = split_dataset(X, y, model_dict['params']['dataset']['train'],
                               model_dict['params']['dataset']['test'],
                               model_dict['params']['dataset']['unlabeled'], r_seed=R_STATE + fold)
        X_train, y_train = p_data['x_train'], p_data['y_train']
        X_tr_umap = umap.UMAP(random_state=R_STATE).fit_transform(X_train)
        class_size = len(np.unique(y_train))
        bias_params = {key: val for key, val in model_dict['params']['bias'].items()
                       if ('name' not in key) and ('y' != key)}
        if 'y' in model_dict['params']['bias'] and model_dict['params']['bias']['y']:
            selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X_train, y=y_train,
                                       **bias_params).astype(int)
        else:
            selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X=X_train, **bias_params).astype(int)

        png_loc = config.RESULT_DIR / 'sample_analysis' / 'umap' / out_loc / f'{fold}.png'
        suptitle = f'{out_loc}_{fold}'
        config.ensure_dir(png_loc)
        visual_fnc.save_selected_umap(X_tr_umap, y_train, selected_ids, out_loc=png_loc, suptitle=suptitle)
        print(f'Fold {fold} ended!')
    print()


def get_bias_pca(model_dict, out_loc):
    X, y = ld.load_dataset(model_dict['params']['dataset']['name'], **model_dict['params']['dataset']['args'])
    for fold in range(model_dict['params']['dataset']['runs']):
        p_data = split_dataset(X, y, model_dict['params']['dataset']['train'],
                               model_dict['params']['dataset']['test'],
                               model_dict['params']['dataset']['unlabeled'], r_seed=R_STATE + fold)
        X_train, y_train = p_data['x_train'], p_data['y_train']
        X_tr_umap = X_train[:, :2]  # umap.UMAP(random_state=R_STATE).fit_transform(X_train)
        class_size = len(np.unique(y_train))
        bias_params = {key: val for key, val in model_dict['params']['bias'].items()
                       if ('name' not in key) and ('y' != key)}
        if 'y' in model_dict['params']['bias'] and model_dict['params']['bias']['y']:
            selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X_train, y=y_train,
                                       **bias_params).astype(int)
        else:
            selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X=X_train, **bias_params).astype(int)

        png_loc = config.RESULT_DIR / 'sample_analysis' / 'pca' / out_loc / f'{fold}.png'
        suptitle = f'{out_loc}_{fold}'
        config.ensure_dir(png_loc)
        visual_fnc.save_selected_umap(X_tr_umap, y_train, selected_ids, out_loc=png_loc, suptitle=suptitle)
    print()


def selection_bias_trial(model_dict, out_loc):
    res_loc = config.RESULT_DIR / f'{model_dict["params"]["bias"]["name"]}' \
              / f'{model_dict["params"]["dataset"]["name"]}' / f'{out_loc}.csv'
    config.ensure_dir(res_loc)
    res_dict_loc = config.RESULT_DIR / f'{model_dict["params"]["bias"]["name"]}' \
                   / f'{model_dict["params"]["dataset"]["name"]}' / f'{out_loc}.pkl'
    config.ensure_dir(res_dict_loc)
    png_bar_loc = config.RESULT_DIR / f'{model_dict["params"]["bias"]["name"]}' \
                  / f'{model_dict["params"]["dataset"]["name"]}' / 'images_bar' / f'{out_loc}.png'
    config.ensure_dir(png_bar_loc)
    png_bp_loc = config.RESULT_DIR / f'{model_dict["params"]["bias"]["name"]}' \
                 / f'{model_dict["params"]["dataset"]["name"]}' / 'images_bp' / f'{out_loc}.png'
    config.ensure_dir(png_bp_loc)
    res_df = None
    if os.path.exists(res_loc):
        res_df = pd.read_csv(res_loc)
    elif os.path.exists(res_dict_loc):
        res_dict_loc = config.load_pickle(res_dict_loc)
        res = []
        for fold, fold_res in res_dict_loc['models'].items():
            for model_name, model_res in fold_res.items():
                res.append([f'{model_name}', fold_res['SU']['sample_size'], model_res['model'].labeled_sample_size,
                            fold, model_res['model']['loss'], model_res['model']['auprc']])
        res_df = pd.DataFrame(res, columns=['Method', 'Start_size', 'EndSize', 'Iteration', 'LogLoss', 'AUPRC'])
        res_df.to_csv(res_loc, index=False)
    else:
        X, y = ld.load_dataset(model_dict['params']['dataset']['name'], **model_dict['params']['dataset']['args'])
        res = []
        for fold in range(model_dict['params']['dataset']['runs']):
            model_dict['models'][f'{fold}'] = {}
            p_data = split_dataset(X, y, model_dict['params']['dataset']['train'],
                                   model_dict['params']['dataset']['test'],
                                   model_dict['params']['dataset']['unlabeled'], r_seed=R_STATE + fold)
            X_train, y_train = p_data['x_train'], p_data['y_train']
            X_test, y_test = p_data['x_test'], p_data['y_test']
            X_unk, y_unk = p_data['x_unk'], p_data['y_unk']
            if model_dict['params']['bias']["name"] is not None:
                bias_params = {key: val for key, val in model_dict['params']['bias'].items()
                               if ('name' not in key) and ('y' != key)}
                if 'y' in model_dict['params']['bias'] and model_dict['params']['bias']['y']:
                    selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X_train, y=y_train,
                                               **bias_params).astype(int)
                else:
                    selected_ids = bt.get_bias(model_dict['params']['bias']['name'], X=X_train, **bias_params).astype(
                        int)
                mask = np.ones_like(y_train, bool)
                mask[selected_ids] = False
                X_b_train, y_b_train = X_train[selected_ids, :], y_train[selected_ids]
            else:
                X_b_train, y_b_train = X_train.copy(), y_train.copy()
            print(f'{sum(y_b_train == 1)} pos and {sum(y_b_train == 0)} neg samples for run {fold}')
            x_b_train, x_b_val, y_b_train, y_b_val = split_dataset(X_b_train, y_b_train,
                                                                   train_ratio=1 - model_dict['params']['dataset'][
                                                                       'val'], r_seed=R_STATE + fold)
            x_b_tr_unk = np.concatenate((x_b_train, X_unk))
            y_b_tr_unk = np.concatenate((y_b_train, np.ones_like(y_unk) * -1))
            print(f'Number of labeled/unlabeled samples at the beginning of run {fold}:{len(y_b_train)}/{len(y_unk)}')

            base_m_params = {key: val for key, val in model_dict['params']['base_model'].items() if 'name' not in key}
            su_clf = lgb.LGBMClassifier(boosting_type="rf", **base_m_params)
            su_clf.fit(x_b_train.copy(), y_b_train.copy(),
                       eval_set=[(x_b_val.copy(), y_b_val.copy())], eval_metric="binary_logloss",
                       callbacks=[lgb.early_stopping(100, verbose=False)])
            probs_su_test = su_clf.predict_proba(X_test)
            y_test = OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray()
            auprc_su = average_precision_score(y_test, probs_su_test)  # [:, 1])
            loss_su = log_loss(y_test, probs_su_test)
            model_dict['models'][f'{fold}'][f'SU'] = {'model': su_clf, 'loss': loss_su, 'auprc': auprc_su,
                                                      'sample_size': len(y_b_train)}
            res.append(['SU', len(y_b_train), len(y_b_train), fold, loss_su, auprc_su])

            model_params = {key: val for key, val in model_dict['params']['model'].items() if 'name' not in key}
            for diversity in [3, 5, 10, None]:
                base_clf = lgb.LGBMClassifier(boosting_type="rf", **base_m_params)
                st_clf = FreeDiverseSelfTraining(base_clf, diverse=diversity, verbose=False, **model_params)
                st_clf.fit(x_b_tr_unk.copy(), y_b_tr_unk.copy(), x_b_val.copy(), y_b_val.copy())
                print(f'Number of samples at the end for d{diversity}:{st_clf.labeled_sample_size}')
                probs_st_test = st_clf.predict_proba(X_test)
                auprc_st = average_precision_score(y_test, probs_st_test)  # [:, 1])
                loss_st = log_loss(y_test, probs_st_test)
                model_dict['models'][f'{fold}'][f'ST_{diversity}'] = {'model': st_clf, 'loss': loss_st,
                                                                      'auprc': auprc_st}
                res.append([f'DST-{diversity}', len(y_b_train), st_clf.labeled_sample_size, fold, loss_st, auprc_st])
            '''
            val_loss_dictx = model_dict['models'][f'{fold}']['ST_5']['model'].val_loss_dict_
            test_loss_dictx = {}
            for iter, model in model_dict['models'][f'{fold}']['ST_5']['model'].estimator_list.items():
                test_probx = model.predict_proba(X_test)
                test_lossx = log_loss(y_test, test_probx)
                test_loss_dictx[iter] = test_lossx
            plt.clf()
            plt.rcParams["figure.figsize"] = (8, 6)
            plt.plot(val_loss_dictx.keys(), val_loss_dictx.values(), label='val')
            plt.plot(test_loss_dictx.keys(), test_loss_dictx.values(), label='test')
            plt.legend()
            plt.show()
            '''
        config.save_pickle(res_dict_loc, model_dict)
        res_df = pd.DataFrame(res, columns=['Method', 'Start_size', 'EndSize', 'Iteration', 'LogLoss', 'AUPRC'])
        res_df.to_csv(res_loc, index=False)

    visual_fnc.call_plot('boxplot', png_bp_loc, res_df, model_dict)
    visual_fnc.call_plot('bar', png_bar_loc, res_df, model_dict)


'''
model_dict = {'models': {}, 'params': {}}
model_dict['params']['bias'] = {'name': 'cluster', 'y': True, 'k': 3, 'n_k': 1}
model_dict['params']['dataset'] = {'name': 'breast_cancer', 'order': 'train_test_unlabeled_bias_validation',
                                   'train': 0.2, 'test': 0.25, 'unlabeled': 0.55, 'val': 0.1, 'runs': 30}
model_dict['params']['base_model'] = params
model_dict['params']['base_model']['name'] = 'BRRF1'
model_dict['params']['model'] = {'name': 'DST', 'threshold': 81, 'k_best': 3, 'max_iter': 100, 'balance': 'free'}
model_dict['params']['model']['full_name'] = f'{model_dict["params"]["model"]["name"]}' \
                                             f'-{model_dict["params"]["base_model"]["name"]}'
bias_str = f'{model_dict["params"]["bias"]["name"]}' \
           f'({"|".join([str(val) for key, val in model_dict["params"]["bias"].items() if "name" not in key])})'

ds_str = f'{model_dict["params"]["dataset"]["name"]}' \
         f'({model_dict["params"]["dataset"]["train"]}' \
         f'|{model_dict["params"]["dataset"]["val"]}' \
         f'|{model_dict["params"]["dataset"]["test"]}' \
         f'|{model_dict["params"]["dataset"]["unlabeled"]})'
model_str = f'{model_dict["params"]["model"]["full_name"]}' \
            f'(th={model_dict["params"]["model"]["threshold"]}' \
            f'|kb={model_dict["params"]["model"]["k_best"]}' \
            f'|mi={model_dict["params"]["model"]["max_iter"]}' \
            f'|b={model_dict["params"]["model"]["balance"]})'
out_loc = f'{bias_str}_{ds_str}_{model_str}'
selection_bias_trial(model_dict, out_loc)
'''

for dataset_name in ['ppi_10116', 'ppi_10090', 'ppi_4932', 'ppi_9823', 'ppi_9606']:  # ['wine_uci2','mushroom', 'mnist', 'credit', 'breast_cancer']:
    for bias_name in ['entity']:#, 'hierarchy1', 'hierarchy2']:  # , 'dirichlet', 'cluster']:
        for bias_per_class in [25]:  # , 50, 100]:
            # dataset='wine_uci'
            # bias_name='hierarchy'
            # bias_per_class = 25
            unique_class = 2
            if dataset_name in ['cifar', 'mnist']:
                unique_class = 10
            if dataset_name in ['wine_uci']:
                unique_class = 3

            if 'ppi' in dataset_name:
                dataset = {'name': dataset_name.split('_')[0], 'args': {'id': int(dataset_name.split('_')[1])},
                           'order': 'train_test_unlabeled_bias_validation', 'train': 0.2, 'test': 0.25,
                           'unlabeled': 0.55, 'val': 0.1, 'runs': 30}
            elif dataset_name == 'cifar':
                dataset = {'name': dataset_name, 'args': {'class_size': 10},
                           'order': 'train_test_unlabeled_bias_validation',
                           'train': 0.2, 'test': 0.25, 'unlabeled': 0.55, 'val': 0.1, 'runs': 30}
            elif dataset_name == 'mnist':
                dataset = {'name': dataset_name, 'args': {}, 'order': 'train_test_unlabeled_bias_validation',
                           'train': 0.35, 'test': 0.25, 'unlabeled': 0.40, 'val': 0.1, 'runs': 30}
            else:
                dataset = {'name': dataset_name, 'args': {}, 'order': 'train_test_unlabeled_bias_validation',
                           'train': 0.2, 'test': 0.25, 'unlabeled': 0.55, 'val': 0.1, 'runs': 30}

            if bias_name == 'cluster':
                bias = {'name': 'cluster', 'y': True, 'k': 3, 'n_k': 1}
            if bias_name == 'hierarchy':
                bias = {'name': 'hierarchy', 'y': True, 'max_size': bias_per_class, 'prob': 0.95}
            if bias_name == 'hierarchy1':
                bias = {'name': 'hierarchy', 'y': True, 'max_size': bias_per_class, 'prob': 0.9}
            if bias_name == 'hierarchy2':
                bias = {'name': 'hierarchy', 'y': True, 'max_size': bias_per_class}
            if bias_name == 'entity':
                bias = {'name': 'entity', 'y': True, 'chosen_entity_size': 1, 'max_size': bias_per_class, 'prob': 0.9}
            if bias_name == 'none':
                bias = {'name': None}
            if bias_name == 'joint':
                bias = {'name': 'joint'}
            if bias_name == 'dirichlet':
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
            out_loc = f'{ds_str}_{bias_str}'
            try:
                # get_bias_pca(model_dict, out_loc)
                get_bias_umap(model_dict, out_loc)
                #get_start_size(model_dict, out_loc)
            except Exception as e:
                print(f'{e} for {dataset_name}-{bias_name}-{bias_per_class}')
'''
kb_list = [3, 6, 10, 20]
bias_per_class = 25
unique_class = 2
if args.dataset in ['cifar', 'mnist']:
    unique_class = 10
    kb_list = [10, 20, 30, 40]
if args.dataset in ['wine_uci']:
    unique_class = 3
    kb_list = [3, 9, 18, 30]

if args.dataset == 'cifar':
    dataset = {'name': args.dataset, 'args': {'class_size': 10}, 'order': 'train_test_unlabeled_bias_validation',
               'train': 0.2, 'test': 0.25, 'unlabeled': 0.55, 'val': 0.1, 'runs': 30}
elif args.dataset == 'mnist':
    dataset = {'name': args.dataset, 'args': {}, 'order': 'train_test_unlabeled_bias_validation',
               'train': 0.35, 'test': 0.25, 'unlabeled': 0.40, 'val': 0.1, 'runs': 30}
else:
    dataset = {'name': args.dataset, 'args': {}, 'order': 'train_test_unlabeled_bias_validation',
               'train': 0.2, 'test': 0.25, 'unlabeled': 0.55, 'val': 0.1, 'runs': 30}
bias_list = []
if args.bias == 'cluster':
    bias_list.append({'name': 'cluster', 'y': True, 'k': 3, 'n_k': 1})
if args.bias == 'hierarchy':
    bias_list.append({'name': 'hierarchy', 'y': True, 'max_size': bias_per_class})
if args.bias == 'none':
    bias_list.append({'name': None})
if args.bias == 'joint':
    bias_list.append({'name': 'joint'})
if args.bias == 'dirichlet':
    bias_list.append({'name': 'dirichlet', 'n': bias_per_class * unique_class})

for bias in bias_list:
    for th in [80, 85, 90, 93, 95, 0.7, 0.8, 0.9]:
        for balance in ['ratio', 'free', 'equal']:
            for kb in kb_list:  # [3, 6, 10, 20]:#[3,6,10]:
                for mi in [10, 20, 50, 100]:
                    model_dict = {'models': {}, 'params': {}}
                    model_dict['params']['bias'] = bias
                    model_dict['params']['dataset'] = dataset
                    model_dict['params']['base_model'] = params
                    model_dict['params']['base_model']['name'] = 'BRRF1'
                    model_dict['params']['model'] = {'name': 'DST', 'threshold': th, 'k_best': kb, 'max_iter': mi,
                                                     'balance': balance}
                    model_dict['params']['model']['full_name'] = f'{model_dict["params"]["model"]["name"]}' \
                                                                 f'-{model_dict["params"]["base_model"]["name"]}'
                    bias_str = f'{model_dict["params"]["bias"]["name"]}' \
                               f'({"|".join([str(val) for key, val in model_dict["params"]["bias"].items() if "name" not in key])})'

                    ds_str = f'{model_dict["params"]["dataset"]["name"]}' \
                             f'({model_dict["params"]["dataset"]["train"]}' \
                             f'|{model_dict["params"]["dataset"]["val"]}' \
                             f'|{model_dict["params"]["dataset"]["test"]}' \
                             f'|{model_dict["params"]["dataset"]["unlabeled"]})'
                    model_str = f'{model_dict["params"]["model"]["full_name"]}' \
                                f'(th={model_dict["params"]["model"]["threshold"]}' \
                                f'|kb={model_dict["params"]["model"]["k_best"]}' \
                                f'|mi={model_dict["params"]["model"]["max_iter"]}' \
                                f'|b={model_dict["params"]["model"]["balance"]})'
                    out_loc = f'{bias_str}_{ds_str}_{model_str}'
                    selection_bias_trial(model_dict, out_loc)
'''
