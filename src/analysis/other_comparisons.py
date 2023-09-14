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

import argparse

parser = argparse.ArgumentParser(description='RFE arguments')
parser.add_argument('--bias', '-b', metavar='the-bias', dest='bias', type=str, help='Choose bias', default='hierarchyy9')
parser.add_argument('--bias_size', '-bs', metavar='the-bias-size', dest='bias_size', type=int, help='Choose the bias size per class', default=30)
parser.add_argument('--folder', '-f', metavar='the-result-folder', dest='res_fold', type=str, help='Choose the main result folder', default='results_test_nb_imb')#results_nn_test_nb_imb_fin
parser.add_argument('--thold', '-t', metavar='the-threshold', dest='threshold', type=float, help='Choose the main threshold', default=97)
parser.add_argument('--valbase', '-v', metavar='the-val-base', dest='val_base', default=False, action=argparse.BooleanOptionalAction)
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

def get_bias_dict():
    bias_per_class = args.bias_size
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
        bias = {'name': 'dirichlet', 'n': bias_per_class}
    return bias
    
def none_hierarchy_multi_datasets(datasets, metric="LogLoss"):
    main_bias = get_bias_dict()
    bias = main_bias.copy()
    res_folder = config.ROOT_DIR / 'comparisons' / f'{bias["name"]}'
    all_datasets_lst = []
    ds_name_fix = {'breast_cancer':"Breast Cancer", 'wine_uci2': "Wine", "mushroom": "Mushroom", "mnist": "MNIST", "drug_CX-5461": "Drug", 'spam': 'Spam', 
                   'adult': 'Adult', 'rice': 'Rice', 'fire': 'Fire', 'pumpkin': 'Pumpkin', 'pistachio': 'Pistachio', 'raisin': 'Raisin'}
    for dataset in datasets:
        lst_to_concat =[]
        kb=6
        u_classes = 2
        th_rf=97
        ds_str = f'{dataset}({0.3}|{0.2}|{0.2}|{0.7})'
        if 'drug_' in dataset:
            ds_str = f'drug({0.3}|{0.2}|{0.2}|{0.7})'
        if 'mnist' in dataset:
            th_rf=85
            kb=30
            u_classes=10
        if 'dirichlet' in bias['name']:
            bias['n']=main_bias['n']
            bias['n']=bias['n']*u_classes
        
        bias_str = f'{bias["name"]}({"|".join([str(val) for key, val in bias.items() if "name" not in key])})'
        #Main RF
        try:
            model_rf_str = f'DST-BRRF1(th={th_rf}|kb={kb}|mi=100|vb=False|b=ratio)'
            out_rf_loc = f'{bias_str}_{ds_str}_{model_rf_str}'
            rf_df = get_res('results_test_nb_imb', f'{bias["name"]}', dataset, out_rf_loc)
            rf_df = rf_df[~np.isin(rf_df['Method'],['DST-50', 'DST-200'])]
            dictre = {"SU": 'Biased-RF', "DST-None": 'ST-RF', 'DST-10':'DST-RF-10', 'DST-100':'DST-RF-100'}#, "DST-10": 'H', "DST-100": 'P'}
            rf_df = rf_df.replace({"Method": dictre})
            lst_to_concat.append(rf_df)
        except Exception as e1:
            print(f"Sorry, main rf coulnd't open: {e1}")
        #KMM RF
        try:
            model_kmm_rf_str = f'KMM-BRRF1(th=rbf|vb=False'
            out_kmm_rf_loc = f'{bias_str}_{ds_str}_{model_kmm_rf_str}'
            kmm_rf_df = get_res('results_kmm_rf_test_nb_imb', f'{bias["name"]}', dataset, out_kmm_rf_loc)
            kmm_rf_df = kmm_rf_df[~np.isin(kmm_rf_df['Method'],['SU'])]
            dictre = {"KMM": 'KMM-RF'}#, "DST-10": 'H', "DST-100": 'P'}
            kmm_rf_df = kmm_rf_df.replace({"Method": dictre})
            lst_to_concat.append(kmm_rf_df)
        except Exception as e1:
            print(f"Sorry, main kmm_rf coulnd't open: {e1}")
        #Main NN
        try:
            model_nn_str = f'DST-BRRF1(th=0.9|kb={kb}|mi=100|vb=False|b=ratio)'
            out_nn_loc = f'{bias_str}_{ds_str}_{model_nn_str}'
            nn_df = get_res('results_nn_test_nb_imb_fin_cw3', f'{bias["name"]}', dataset, out_nn_loc)
            nn_df = nn_df[~np.isin(nn_df['Method'],['DST-50', 'DST-200'])]
            dictre = {"SU": 'Biased-NN', "DST-None": 'ST-NN', 'DST-10':'DST-NN-10', 'DST-100':'DST-NN-100'}#, "DST-10": 'H', "DST-100": 'P'}
            nn_df = nn_df.replace({"Method": dictre})
            lst_to_concat.append(nn_df)
        except Exception as e1:
            print(f"Sorry, main nn coulnd't open: {e1}")
        #KMM NN
        try:
            model_kmm_nn_str = f'KMM-BRRF1(th=rbf|vb=False'
            out_kmm_nn_loc = f'{bias_str}_{ds_str}_{model_kmm_nn_str}'
            kmm_nn_df = get_res('results_kmm_nn_test_nb_imb', f'{bias["name"]}', dataset, out_kmm_nn_loc)
            kmm_nn_df = kmm_nn_df[~np.isin(kmm_nn_df['Method'],['SU'])]
            dictre = {"KMM": 'KMM-NN'}#, "DST-10": 'H', "DST-100": 'P'}
            kmm_nn_df = kmm_nn_df.replace({"Method": dictre})
            lst_to_concat.append(kmm_nn_df)
        except Exception as e1:
            print(f"Sorry, main kmm_nn coulnd't open: {e1}")
        #LR - KMM and RBA
        try:
            model_lr_str = f'DST-BRRF1(th=97|kb={kb}|mi=100|vb=False|b=ratio)y'
            out_lr_loc = f'{bias_str}_{ds_str}_{model_lr_str}'
            lr_df = get_res('results_lr_test_nb_imb', f'{bias["name"]}', dataset, out_lr_loc)
            dictre = {"KMM": 'KMM-LR', "RBA": 'RBA-LR', "SU": 'Biased-LR'}#, "DST-10": 'H', "DST-100": 'P'}
            lr_df = lr_df.replace({"Method": dictre})
            lst_to_concat.append(lr_df) 
        except Exception as e1:
            print(f"Sorry, main lr coulnd't open: {e1}")
        
        a = pd.concat(lst_to_concat)
        a['Dataset'] = ds_name_fix[dataset]
        all_datasets_lst.append(a)
        #print(a.head())
    all_ds = pd.concat(all_datasets_lst)
    print(all_ds.head())
    #print(all_ds.head())
    #print(all_ds.shape)
    out_loc = res_folder / f'Multids_{bias["name"]}_{metric}_{args.val_base}.png'
    out_loc_pdf = res_folder / f'Multids_{bias["name"]}_{metric}_{args.val_base}.pdf'
    config.ensure_dir(out_loc)
    PROPS = {
        'boxprops': {'edgecolor': 'black', 'linewidth':0.4, 'alpha':0.85},
        'medianprops': {'color': 'black', 'linewidth':0.6},
        'whiskerprops': {'color': 'black', 'linewidth':0.4 },
        'capprops': {'color': 'black', 'linewidth':0.4 },
        'flierprops': {'marker':'o', 'markerfacecolor':'None', 'markersize':2, 'linestyle':'none', 'markeredgewidth':0.5}
    }
    rc = {'xtick.bottom': True, 'xtick.left': True}
    fig = plt.figure(figsize=(7, 10))
    #plt.rcParams["figure.figsize"] = (5,6)
    sns.axes_style(style="white", rc=rc) #font='Arial'
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "Arial"
    sbox = sns.boxplot(y="Dataset", x=metric, hue="Method", data=all_ds, showfliers=True, **PROPS)
    grouped_summary = all_ds.groupby(['Dataset', 'Method']).agg({'Accuracy':['mean', 'median', 'std', 'min', 'max'], 
                                                                 'LogLoss':['mean', 'median', 'std', 'min', 'max'],
                                                                 'AUROC':['mean', 'median', 'std', 'min', 'max'],
                                                                 'AUPRC':['mean', 'median', 'std', 'min', 'max']})
    print(grouped_summary) 

    
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
    plt.savefig(out_loc, dpi=300, bbox_inches='tight')
    plt.savefig(out_loc_pdf, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def none_hierarchy_multi_datasets_facet(datasets, metric="LogLoss"):
    last_ds = datasets[-1]
    main_bias = get_bias_dict()
    bias = main_bias.copy()
    res_folder = config.ROOT_DIR / 'comparisons' / f'{bias["name"]}'
    all_datasets_lst = []
    ds_name_fix = {'breast_cancer':"Breast Cancer", 'wine_uci2': "Wine", "mushroom": "Mushroom", "mnist": "MNIST", "drug_CX-5461": "Drug", 'spam': 'Spam', 
                   'adult': 'Adult', 'rice': 'Rice', 'fire': 'Fire', 'pumpkin': 'Pumpkin', 'pistachio': 'Pistachio', 'raisin': 'Raisin'}
    for dataset in datasets:
        lst_to_concat =[]
        kb=6
        u_classes = 2
        th_rf=97
        ds_str = f'{dataset}({0.3}|{0.2}|{0.2}|{0.7})'
        if 'drug_' in dataset:
            ds_str = f'drug({0.3}|{0.2}|{0.2}|{0.7})'
        if 'mnist' in dataset:
            th_rf=85
            kb=30
            u_classes=10
        if 'dirichlet' in bias['name']:
            bias['n']=main_bias['n']
            bias['n']=bias['n']*u_classes
        
        bias_str = f'{bias["name"]}({"|".join([str(val) for key, val in bias.items() if "name" not in key])})'
        #Main RF
        try:
            model_rf_str = f'DST-BRRF1(th={th_rf}|kb={kb}|mi=100|vb=False|b=ratio)'
            out_rf_loc = f'{bias_str}_{ds_str}_{model_rf_str}'
            rf_df = get_res('results_test_nb_imb', f'{bias["name"]}', dataset, out_rf_loc)
            rf_df = rf_df[~np.isin(rf_df['Method'],['DST-50', 'DST-200'])]
            dictre = {"SU": 'Biased-RF', "DST-None": 'ST-RF', 'DST-10':'DST-RF-10', 'DST-100':'DST-RF-100'}#, "DST-10": 'H', "DST-100": 'P'}
            rf_df = rf_df.replace({"Method": dictre})
            rf_df['Main Model'] = 'RF'
            lst_to_concat.append(rf_df)
        except Exception as e1:
            print(f"Sorry, main rf coulnd't open: {e1}")
        #KMM RF
        try:
            model_kmm_rf_str = f'KMM-BRRF1(th=rbf|vb=False'
            out_kmm_rf_loc = f'{bias_str}_{ds_str}_{model_kmm_rf_str}'
            kmm_rf_df = get_res('results_kmm_rf_test_nb_imb', f'{bias["name"]}', dataset, out_kmm_rf_loc)
            kmm_rf_df = kmm_rf_df[~np.isin(kmm_rf_df['Method'],['SU'])]
            dictre = {"KMM": 'KMM-RF'}#, "DST-10": 'H', "DST-100": 'P'}
            kmm_rf_df = kmm_rf_df.replace({"Method": dictre})
            kmm_rf_df['Main Model'] = 'RF'
            lst_to_concat.append(kmm_rf_df)
        except Exception as e1:
            print(f"Sorry, main kmm_rf coulnd't open: {e1}")
            tmp_dct = {'Method':'KMM-RF', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'RF'}
            df_dictionary = pd.DataFrame([tmp_dct])
            rf_df = pd.concat([rf_df, df_dictionary])
        #Main NN
        try:
            model_nn_str = f'DST-BRRF1(th=0.9|kb={kb}|mi=100|vb=False|b=ratio)'
            out_nn_loc = f'{bias_str}_{ds_str}_{model_nn_str}'
            nn_df = get_res('results_nn_test_nb_imb_fin_cw3', f'{bias["name"]}', dataset, out_nn_loc)
            nn_df = nn_df[~np.isin(nn_df['Method'],['DST-50', 'DST-200'])]
            dictre = {"SU": 'Biased-NN', "DST-None": 'ST-NN', 'DST-10':'DST-NN-10', 'DST-100':'DST-NN-100'}#, "DST-10": 'H', "DST-100": 'P'}
            nn_df = nn_df.replace({"Method": dictre})
            nn_df['Main Model'] = 'NN'
            lst_to_concat.append(nn_df)
        except Exception as e1:
            print(f"Sorry, main nn coulnd't open: {e1}")
        #KMM NN
        try:
            model_kmm_nn_str = f'KMM-BRRF1(th=rbf|vb=False'
            out_kmm_nn_loc = f'{bias_str}_{ds_str}_{model_kmm_nn_str}'
            kmm_nn_df = get_res('results_kmm_nn_test_nb_imb', f'{bias["name"]}', dataset, out_kmm_nn_loc)
            kmm_nn_df = kmm_nn_df[~np.isin(kmm_nn_df['Method'],['SU'])]
            dictre = {"KMM": 'KMM-NN'}#, "DST-10": 'H', "DST-100": 'P'}
            kmm_nn_df = kmm_nn_df.replace({"Method": dictre})
            kmm_nn_df['Main Model'] = 'NN'
            lst_to_concat.append(kmm_nn_df)
        except Exception as e1:
            print(f"Sorry, main kmm_nn coulnd't open: {e1}")
            tmp_dct = {'Method':'KMM-NN', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'NN'}
            df_dictionary = pd.DataFrame([tmp_dct])
            nn_df = pd.concat([nn_df, df_dictionary])
        #LR - KMM and RBA
        try:
            model_lr_str = f'DST-BRRF1(th=97|kb={kb}|mi=100|vb=False|b=ratio)'
            out_lr_loc = f'{bias_str}_{ds_str}_{model_lr_str}'
            lr_df = get_res('results_lr_test_nb_imb', f'{bias["name"]}', dataset, out_lr_loc)
            dictre = {"KMM": 'KMM-LR', "RBA": 'RBA-LR', "KDE": 'KDE-LR', "FLDA": 'FLDA-LR', "TCPR": 'TCPR-LR', "SUBA": 'SUBA-LR', "SU": 'Biased-LR'}#, "DST-10": 'H', "DST-100": 'P'}
            lr_df = lr_df.replace({"Method": dictre})
            lr_df['Main Model'] = 'LR'
            if True:#last_ds == dataset:# and len(lr_df['Method'].unique())<3:
                drug_lr_mets = lr_df['Method'].unique()
                if 'KMM-LR' not in drug_lr_mets:
                    tmp_dct = {'Method':'KMM-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                    df_dictionary = pd.DataFrame([tmp_dct])
                    lr_df = pd.concat([lr_df, df_dictionary])
                if 'KDE-LR' not in drug_lr_mets:
                    tmp_dct = {'Method':'KDE-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                    df_dictionary = pd.DataFrame([tmp_dct])
                    lr_df = pd.concat([lr_df, df_dictionary])
                if 'RBA-LR' not in drug_lr_mets:
                    tmp_dct = {'Method':'RBA-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                    df_dictionary = pd.DataFrame([tmp_dct])
                    lr_df = pd.concat([lr_df, df_dictionary])
                if 'FLDA-LR' not in drug_lr_mets:
                    tmp_dct = {'Method':'FLDA-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                    df_dictionary = pd.DataFrame([tmp_dct])
                    lr_df = pd.concat([lr_df, df_dictionary])
                if 'TCPR-LR' not in drug_lr_mets:
                    tmp_dct = {'Method':'TCPR-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                    df_dictionary = pd.DataFrame([tmp_dct])
                    lr_df = pd.concat([lr_df, df_dictionary])
                if 'SUBA-LR' not in drug_lr_mets:
                    tmp_dct = {'Method':'SUBA-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                    df_dictionary = pd.DataFrame([tmp_dct])
                    lr_df = pd.concat([lr_df, df_dictionary])
            lst_to_concat.append(lr_df) 
        except Exception as e1:
            print(f"Sorry, main lr coulnd't open: {e1}")
            if True:#last_ds == dataset:# and len(lr_df['Method'].unique())<3:
                lr_df = pd.DataFrame()
                tmp_dct = {'Method':'Biased-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                df_dictionary = pd.DataFrame([tmp_dct])
                lr_df = pd.concat([lr_df, df_dictionary])
                tmp_dct = {'Method':'KMM-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                df_dictionary = pd.DataFrame([tmp_dct])
                lr_df = pd.concat([lr_df, df_dictionary])
                tmp_dct = {'Method':'KDE-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                df_dictionary = pd.DataFrame([tmp_dct])
                lr_df = pd.concat([lr_df, df_dictionary])
                tmp_dct = {'Method':'RBA-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                df_dictionary = pd.DataFrame([tmp_dct])
                lr_df = pd.concat([lr_df, df_dictionary])
                tmp_dct = {'Method':'FLDA-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                df_dictionary = pd.DataFrame([tmp_dct])
                lr_df = pd.concat([lr_df, df_dictionary])
                tmp_dct = {'Method':'TCPR-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                df_dictionary = pd.DataFrame([tmp_dct])
                lr_df = pd.concat([lr_df, df_dictionary])
                tmp_dct = {'Method':'SUBA-LR', 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                df_dictionary = pd.DataFrame([tmp_dct])
                lr_df = pd.concat([lr_df, df_dictionary])
                lst_to_concat.append(lr_df) 
                
        
        a = pd.concat(lst_to_concat)
        print(lst_to_concat[0].head())
        a['Dataset'] = ds_name_fix[dataset]
        all_datasets_lst.append(a)
        #print(a.head())
    accepted_methods = ['Biased-RF', 'ST-RF', 'DST-RF-10', 'DST-RF-100', 'KMM-RF', 'Biased-NN', 'ST-NN', 'DST-NN-10', 'DST-NN-100', 'KMM-NN', 'Biased-LR', 'KMM-LR', 'KDE-LR', 'RBA-LR', 'FLDA-LR', 'TCPR-LR', 'SUBA-LR']
    met_sort = {'Biased-RF':0, 'ST-RF':1, 'DST-RF-10':2, 'DST-RF-100':3, 'KMM-RF':4,
                'Biased-NN':5, 'ST-NN':6, 'DST-NN-10':7, 'DST-NN-100':8, 'KMM-NN':9,
                'Biased-LR':10, 'KMM-LR':11, 'KDE-LR':12, 'RBA-LR':13, 'FLDA-LR':14, 'TCPR-LR':15, 'SUBA-LR':16}
    color_dict = {'Biased-RF':'b', 'ST-RF':'r', 'DST-RF-10':'r', 'DST-RF-100':'r', 'KMM-RF':'g',
                'Biased-NN':'b', 'ST-NN':'r', 'DST-NN-10':'r', 'DST-NN-100':'r', 'KMM-NN':'g',
                'Biased-LR':'b', 'KMM-LR':'g', 'KDE-LR':'g', 'RBA-LR':'y', 'FLDA-LR':'y', 'TCPR-LR':'y', 'SUBA-LR':'y'}
    met_hue = {'Biased-RF':'Biased', 'ST-RF':'SS', 'DST-RF-10':'SS', 'DST-RF-100':'SS', 'KMM-RF':'KMM',
                'Biased-NN':'Biased', 'ST-NN':'SS', 'DST-NN-10':'SS', 'DST-NN-100':'SS', 'KMM-NN':'KMM',
                'Biased-LR':'Biased', 'KMM-LR':'KMM', 'RBA-LR':'RBA', 'KDE-LR':'KDE', 'FLDA-LR':'FLDA', 'TCPR-LR':'TCPR', 'SUBA-LR':'SUBA'}
    #ds_sort = {"Breast Cancer":0, "Wine":1, "Mushroom":2, "MNIST":3, "Drug":4, 'Adult':5}#'spam', 'adult', 'fire', 'rice', 'raisin', 'pistachio', 'pumpkin'
    ds_sort = {"Breast Cancer":0, "Wine":1, "Mushroom":2, "MNIST":3, "Drug":4, 'Adult':5, 
              "Spam":6, "Adult":7, "Fire":8, "Rice":9, "Raisin":10, 'Pistachio':11, 'Pumpkin':12}
    all_ds = pd.concat(all_datasets_lst)
    all_ds = all_ds[all_ds['Method'].isin(accepted_methods)]
    all_ds['met_sort'] = all_ds['Method'].map(met_sort)
    all_ds['Category'] = all_ds['Method'].map(met_hue)
    all_ds['ds_sort'] = all_ds['Dataset'].map(ds_sort)
    all_ds = all_ds.sort_values(['ds_sort', 'met_sort', 'Iteration'], ascending=[True, True, True])
    print(all_ds.head())
    #print(all_ds.head())
    #print(all_ds.shape)
    out_loc = res_folder / f'Multids_{bias["name"]}_{metric}_{args.val_base}_facetx.png'
    out_loc_pdf = res_folder / f'Multids_{bias["name"]}_{metric}_{args.val_base}_facetx.pdf'
    config.ensure_dir(out_loc)
    PROPS = {
        'boxprops': {'edgecolor': 'black', 'linewidth':0.4},
        'medianprops': {'color': 'black', 'linewidth':0.4},
        'whiskerprops': {'color': 'black', 'linewidth':0.4 },
        'capprops': {'color': 'black', 'linewidth':0.4 },
        'flierprops': {'marker':'o', 'markerfacecolor':'None', 'markersize':2, 'linestyle':'none', 'markeredgewidth':0.5}
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
    gf = sns.FacetGrid(all_ds, col='Main Model', row='Dataset', height=2, aspect=1.35, sharex='col', sharey='row', gridspec_kws={"wspace":0.08, 'hspace':0.15})
    gmapped = gf.map_dataframe(fixed_boxplot, "Method", metric, showfliers=False, width=0.6, palette=color_dict, **PROPS)
    for ax in gf.axes.flat:
        for idd, patch in enumerate(ax.artists):
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .1))
    gmapped2 = gf.map_dataframe(sns.swarmplot, "Method", metric, palette=color_dict, size=1)
    grouped_summary = all_ds.groupby(['Dataset', 'Method']).agg({'Accuracy':['mean', 'median', 'std', 'min', 'max'], 
                                                                 'LogLoss':['mean', 'median', 'std', 'min', 'max'],
                                                                 'AUROC':['mean', 'median', 'std', 'min', 'max'],
                                                                 'AUPRC':['mean', 'median', 'std', 'min', 'max']})
    print(grouped_summary) 

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
    for ax in gf.axes.flat:
        ax.set_xlabel(ax.get_xlabel(), family='Arial', fontdict={'weight':'bold', 'size':6.5})
        ax.set_ylabel(ax.get_ylabel(), family='Arial', fontdict={'weight':'bold', 'size':6.5})
        ax.set_xticklabels(ax.get_xticklabels(), family='Arial', fontdict={ 'size':5.4})
        plt.setp(ax.get_xticklabels(), rotation=30)
        plt.setp(ax.get_yticklabels(), size=5.4)
        #ax.tick_params(width=0.3)
        #if ax.get_yticklabels():
        #ax.set_yticklabels(ax.get_yticklabels(), family='Arial', fontdict={ 'size':6})
        #print(ax.get_ylabel())
        if not ax.get_ylabel():
            ax.spines['left'].set_linewidth(0.05)
            ax.tick_params(axis='y', width=0.1)
        else:
            ax.spines['left'].set_linewidth(0.2)
            ax.tick_params(width=0.3)
        
        if not ax.get_xlabel():
            ax.spines['bottom'].set_linewidth(0.05)
            ax.tick_params(axis='x', width=0.1)
        else:
            ax.spines['bottom'].set_linewidth(0.2)
            ax.tick_params(width=0.3)
        #ax.spines['bottom'].set_linewidth(0.2)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.title.set_position([.5, 0.9])
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .95))
    
    gf.set_titles(col_template="Base Model = {col_name}", row_template="{row_name}", family='Arial', size=6.5, pad=-0.8)
    #plt.tight_layout()#(pad=0.7)
    plt.savefig(out_loc, dpi=300, bbox_inches='tight')
    plt.savefig(out_loc_pdf, dpi=300, bbox_inches='tight')
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

datasets = ['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'drug_CX-5461']#, 'spam']
datasets = ['spam', 'adult', 'fire', 'rice', 'raisin', 'pistachio', 'pumpkin']
datasets = ['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'spam', 'fire', 'pistachio', 'raisin']
#'spam', 'adult', 'rice', 'fire', 'pumpkin', 'pistachio', 'raisin'
none_hierarchy_multi_datasets_facet(datasets, metric='LogLoss')#model_dict, out_loc)
none_hierarchy_multi_datasets_facet(datasets, metric='Accuracy')#model_dict, out_loc)
none_hierarchy_multi_datasets_facet(datasets, metric='AUROC')#model_dict, out_loc)
none_hierarchy_multi_datasets_facet(datasets, metric='AUPRC')#model_dict, out_loc)