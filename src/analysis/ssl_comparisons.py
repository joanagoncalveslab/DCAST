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
            dictre = {"SU": 'RF', "DST-None": 'ST-RF', 'DST-10':'DST-10-RF', 'DST-100':'DST-100-RF'}#, "DST-10": 'H', "DST-100": 'P'}
            rf_df = rf_df.replace({"Method": dictre})
            lst_to_concat.append(rf_df)
        except Exception as e1:
            print(f"Sorry, main rf coulnd't open: {e1}")
        #Vanilla RF
        try:
            vanilla_rf_df = get_res('results_extra_test_nb_imb_ss', f'{bias["name"]}', dataset, out_rf_loc)
            vanilla_rf_df = kmm_rf_df[~np.isin(kmm_rf_df['Method'],['SU'])]
            dictre = {"KMM": 'KMM-RF'}#, "DST-10": 'H', "DST-100": 'P'}
            vanilla_rf_df = vanilla_rf_df.replace({"Method": dictre})
            lst_to_concat.append(kmm_rf_df)
        except Exception as e1:
            print(f"Sorry, main kmm_rf coulnd't open: {e1}")
        #Main NN
        try:
            model_nn_str = f'DST-BRRF1(th=0.9|kb={kb}|mi=100|vb=False|b=ratio)'
            out_nn_loc = f'{bias_str}_{ds_str}_{model_nn_str}'
            nn_df = get_res('results_nn_test_nb_imb_fin_cw3', f'{bias["name"]}', dataset, out_nn_loc)
            nn_df = nn_df[~np.isin(nn_df['Method'],['DST-50', 'DST-200'])]
            dictre = {"SU": 'NN', "DST-None": 'ST-NN', 'DST-10':'DST-NN-10', 'DST-100':'DST-NN-100'}#, "DST-10": 'H', "DST-100": 'P'}
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
    
    
def none_hierarchy_multi_datasets_facet(datasets, metric="LogLoss", swarmed=True):
    colsize = 4
    rowsize = int(len(datasets)/colsize)
    mutids_name = '|'.join(datasets)
    plt.clf()
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    last_ds = datasets[-1]
    main_bias = get_bias_dict()
    bias = main_bias.copy()
    res_folder = config.ROOT_DIR / 'comparisons_SSL' / f'{bias["name"]}'
    all_datasets_lst = []
    ds_name_fix = {'breast_cancer':"Breast Cancer", 'wine_uci2': "Wine", "mushroom": "Mushroom", "mnist": "MNIST", "drug_CX-5461": "Drug", 'spam': 'Spam', 
                   'adult': 'Adult', 'rice': 'Rice', 'fire': 'Fire', 'pumpkin': 'Pumpkin', 'pistachio': 'Pistachio', 'raisin': 'Raisin'}
    for dataset in datasets:
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
            th_lr=0.9
            kb=30
            u_classes=10
        if 'dirichlet' in bias['name']:
            bias['n']=main_bias['n']
            bias['n']=bias['n']*u_classes
        
        bias_str = f'{bias["name"]}({"|".join([str(val) for key, val in bias.items() if "name" not in key])})'
        #Main RF
        try:
            model_rf_str = f'DST-BRRF1(th={th_rf}|kb={kb}|mi=100|vb=False|b=ratio)_es=True'
            out_rf_loc = f'{bias_str}_{ds_str}_{model_rf_str}'
            rf_df = get_res('results_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_rf_loc)
            rf_df = rf_df[~np.isin(rf_df['Method'],['DST-50', 'DST-200', 'ST_50', 'ST_200'])]
            dictre_rf = {"SU": 'RF', "DST-None": 'BST-RF', 'DST-10':'DBST-10-RF', 'DST-100':'DBST-100-RF', 'ST_None': 'BST-RF', 'ST_10':'DBST-10-RF', 'ST_100':'DBST-100-RF'}#, "DST-10": 'H', "DST-100": 'P'}
            rf_df = rf_df.replace({"Method": dictre_rf})
            rf_df['Main Model'] = 'RF'
            lst_to_concat.append(rf_df)
        except Exception as e1:
            print(f"Sorry, main rf coulnd't open: {e1}")
            
        #Vanilla RF
        dictre_vrf = {"SU": 'RF', "ST_th": 'ST_th-RF', 'ST_kb':'ST_kb-RF'}
        try:
            model_vrf_str = f'DST-BRRF1(th={0.9}|kb={kb}|mi=100|vb=False|b=ratio)'
            out_vrf_loc = f'{bias_str}_{ds_str}_{model_vrf_str}'
            vrf_df = get_res('results_extra_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_vrf_loc)
            vrf_df = vrf_df[~np.isin(vrf_df['Method'],['SU','DST-50', 'DST-200'])]
            #, "DST-10": 'H', "DST-100": 'P'}
            vrf_df = vrf_df.replace({"Method": dictre_vrf})
            vrf_df['Main Model'] = 'RF'
            for newmetname in dictre_vrf.values():
                if newmetname not in vrf_df['Method'].unique():
                    tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'RF'}
                    df_dictionary = pd.DataFrame([tmp_dct]).copy()
                    vrf_df = pd.concat([vrf_df, df_dictionary])
            lst_to_concat.append(vrf_df)
        except Exception as e1:
            print(f"Sorry, vanilla rf coulnd't open: {e1}")
            vrf_df = pd.DataFrame()
            for newmetname in dictre_vrf.values():
                tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'RF'}
                df_dictionary = pd.DataFrame([tmp_dct]).copy()
                vrf_df = pd.concat([vrf_df, df_dictionary])
            lst_to_concat.append(vrf_df)

        #FSD RF
        dictre_fsd_rf = {'DST-10':'FSD-BST-10-RF', 'DST-100':'FSD-BST-100-RF'}#, "DST-10": 'H', "DST-100": 'P'}
        try:
            try:
                fsd_rf_df = get_res('results_fsd_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_rf_loc)
            except:
                model_rf_dbst_str = f'DST-BRRF1(th={th_rf}|kb={kb}|mi=100|vb=False|b=ratio)'
                out_rf_dbst_loc = f'{bias_str}_{ds_str}_{model_rf_dbst_str}'
                fsd_rf_df = get_res('results_fsd_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_rf_dbst_loc)
            
            if 'ST_None' in fsd_rf_df['Method'].unique():
                dictre_fsd_rf = {'ST_10':'FSD-BST-10-RF', 'ST_100':'FSD-BST-100-RF'}
                fsd_rf_df = fsd_rf_df[~np.isin(fsd_rf_df['Method'],['SU', 'ST_None', 'ST_50', 'ST_200'])]
            fsd_rf_df = fsd_rf_df[~np.isin(fsd_rf_df['Method'],['SU', 'DST-None', 'DST-50', 'DST-200'])]
            fsd_rf_df = fsd_rf_df.replace({"Method": dictre_fsd_rf})
            fsd_rf_df['Main Model'] = 'RF'
            for newmetname in dictre_fsd_rf.values():
                if newmetname not in fsd_rf_df['Method'].unique():
                    tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'RF'}
                    df_dictionary = pd.DataFrame([tmp_dct]).copy()
                    fsd_rf_df = pd.concat([fsd_rf_df, df_dictionary])
            lst_to_concat.append(fsd_rf_df)
        except Exception as e1:
            print(f"Sorry, fsd-st rf coulnd't open: {e1}")
            fsd_rf_df = pd.DataFrame()
            for newmetname in dictre_fsd_rf.values():
                tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'RF'}
                df_dictionary = pd.DataFrame([tmp_dct]).copy()
                fsd_rf_df = pd.concat([fsd_rf_df, df_dictionary])
            lst_to_concat.append(fsd_rf_df)
            

        #Main NN
        try:
            model_nn_str = f'DST-BRRF1(th=0.9|kb={kb}|mi=100|vb=False|b=ratio)_es=True'
            out_nn_loc = f'{bias_str}_{ds_str}_{model_nn_str}'
            nn_df = get_res('results_nn_test_nb_imb_fin_cw3_ss8', f'{bias["name"]}', dataset, out_nn_loc)
            nn_df = nn_df[~np.isin(nn_df['Method'],['DST-50', 'DST-200'])]
            dictre_nn = {"SU": 'NN', "DST-None": 'BST-NN', 'DST-10':'DBST-10-NN', 'DST-100':'DBST-100-NN'}#, "DST-10": 'H', "DST-100": 'P'}
            nn_df = nn_df.replace({"Method": dictre_nn})
            nn_df['Main Model'] = 'NN'
            lst_to_concat.append(nn_df)
            print(nn_df['Method'].unique())
        except Exception as e1:
            print(f"Sorry, main nn coulnd't open: {e1}")
            
        #Vanilla NN
        dictre_vnn = {"SU": 'NN', "ST_th": 'ST_th-NN', 'ST_kb':'ST_kb-NN'}
        try:
            model_vnn_str = f'DST-BRRF1(th=0.9|kb={kb}|mi=100|vb=False|b=ratio)'
            out_vnn_loc = f'{bias_str}_{ds_str}_{model_vnn_str}'
            vnn_df = get_res('results_nn_extra_test_nb_imb_fin_cw3_ss8', f'{bias["name"]}', dataset, out_vnn_loc)
            vnn_df = vnn_df[~np.isin(vnn_df['Method'],['SU','DST-50', 'DST-200'])]
            #dictre_vnn = {"SU": 'NN', "ST_th": 'ST_th-NN', 'ST_kb':'ST_kb-NN'}#, "DST-10": 'H', "DST-100": 'P'}
            vnn_df = vnn_df.replace({"Method": dictre_vnn})
            vnn_df['Main Model'] = 'NN'
            for newmetname in dictre_vnn.values():
                if newmetname not in vnn_df['Method'].unique():
                    tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'NN'}
                    df_dictionary = pd.DataFrame([tmp_dct]).copy()
                    vnn_df = pd.concat([vnn_df, df_dictionary])
            lst_to_concat.append(vnn_df)
        except Exception as e1:
            print(f"Sorry, vanilla nn coulnd't open: {e1}")
            vnn_df = pd.DataFrame()
            for newmetname in dictre_vnn.values():
                tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'NN'}
                df_dictionary = pd.DataFrame([tmp_dct]).copy()
                vnn_df = pd.concat([vnn_df, df_dictionary])
            lst_to_concat.append(vnn_df)

        #FSD NN
        dictre_fsd_nn = {'DST-10':'FSD-BST-10-NN', 'DST-100':'FSD-BST-100-NN'}#, "DST-10": 'H', "DST-100": 'P'}
        try:
            try:
                fsd_nn_df = get_res('results_fsdnn_test_nb_imb_fin_cw3_ss8', f'{bias["name"]}', dataset, out_nn_loc)
            except:
                model_nn_fsd_str = f'DST-BRRF1(th=0.9|kb={kb}|mi=100|vb=False|b=ratio)'
                out_nn_fsd_loc = f'{bias_str}_{ds_str}_{model_nn_fsd_str}'
                fsd_nn_df = get_res('results_fsdnn_test_nb_imb_fin_cw3_ss8', f'{bias["name"]}', dataset, out_nn_fsd_loc)
            fsd_nn_df = fsd_nn_df[~np.isin(fsd_nn_df['Method'],['SU', 'DST-None', 'DST-50', 'DST-200'])]
            fsd_nn_df = fsd_nn_df.replace({"Method": dictre_fsd_nn})
            fsd_nn_df['Main Model'] = 'NN'
            for newmetname in dictre_fsd_nn.values():
                if newmetname not in fsd_nn_df['Method'].unique():
                    tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'NN'}
                    df_dictionary = pd.DataFrame([tmp_dct]).copy()
                    fsd_nn_df = pd.concat([fsd_nn_df, df_dictionary])
            lst_to_concat.append(fsd_nn_df)
        except Exception as e1:
            print(f"Sorry, fsd-st nn coulnd't open: {e1}")
            fsd_nn_df = pd.DataFrame()
            for newmetname in dictre_fsd_nn.values():
                tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'NN'}
                df_dictionary = pd.DataFrame([tmp_dct]).copy()
                fsd_nn_df = pd.concat([fsd_nn_df, df_dictionary])
            lst_to_concat.append(fsd_nn_df)
            print(fsd_nn_df.head())

            
        #Vanilla LR
        dictre_vlr = {"SU": 'LR', "ST_th": 'ST_th-LR', 'ST_kb':'ST_kb-LR'}
        try:
            model_vlr_str = f'DST-BRRF1(th={0.9}|kb={kb}|mi=100|vb=False|b=ratio)_es=False'
            out_vlr_loc = f'{bias_str}_{ds_str}_{model_vlr_str}'
            vlr_df = get_res('results_lr_extra_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_vlr_loc)
            vlr_df = vlr_df[~np.isin(vlr_df['Method'],['SU','DST-50', 'DST-200'])]
            #, "DST-10": 'H', "DST-100": 'P'}
            vlr_df = vlr_df.replace({"Method": dictre_vlr})
            vlr_df['Main Model'] = 'LR'
            for newmetname in dictre_vlr.values():
                if newmetname not in vlr_df['Method'].unique():
                    tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                    df_dictionary = pd.DataFrame([tmp_dct]).copy()
                    vlr_df = pd.concat([vlr_df, df_dictionary])
            lst_to_concat.append(vlr_df)
        except Exception as e1:
            print(f"Sorry, vanilla lr coulnd't open: {e1}")
            vlr_df = pd.DataFrame()
            for newmetname in dictre_vlr.values():
                tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                df_dictionary = pd.DataFrame([tmp_dct]).copy()
                vlr_df = pd.concat([vlr_df, df_dictionary])
            lst_to_concat.append(vlr_df)
        
        dictre_fsd_lr = {'SU': 'LR', "DST-None": 'BST-LR', 'DST-10':'FSD-BST-10-LR', 'DST-100':'FSD-BST-100-LR'}#, "DST-10": 'H', "DST-100": 'P'}
        try:
            model_lr_dbst_str = f'DST-BRRF1(th={th_lr}|kb={kb}|mi=100|vb=False|b=ratio)'
            try:
                out_lr_dbst_loc = f'{bias_str}_{ds_str}_{model_lr_dbst_str}_es=True'
                fsd_lr_df = get_res('results_lr_fsd_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_lr_dbst_loc)
            except Exception as e_weird:
                print(e_weird)
                out_lr_dbst_loc = f'{bias_str}_{ds_str}_{model_lr_dbst_str}'
                fsd_lr_df = get_res('results_lr_fsd_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_lr_dbst_loc)
            
            if 'ST_None' in fsd_lr_df['Method'].unique():
                dictre_fsd_lr = {'SU': 'LR', "ST_None": 'BST-LR', 'ST_10':'FSD-BST-10-LR', 'ST_100':'FSD-BST-100-LR'}
                fsd_lr_df = fsd_lr_df[~np.isin(fsd_lr_df['Method'],['ST_50', 'ST_200'])]
            fsd_lr_df = fsd_lr_df[~np.isin(fsd_lr_df['Method'],['DST-50', 'DST-200'])]
            fsd_lr_df = fsd_lr_df.replace({"Method": dictre_fsd_lr})
            fsd_lr_df['Main Model'] = 'LR'
            for newmetname in dictre_fsd_lr.values():
                if newmetname not in fsd_lr_df['Method'].unique():
                    tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                    df_dictionary = pd.DataFrame([tmp_dct]).copy()
                    fsd_lr_df = pd.concat([fsd_lr_df, df_dictionary])
            lst_to_concat.append(fsd_lr_df)
        except Exception as e1:
            print(f"Sorry, fsd-st lr coulnd't open: {e1}")
            fsd_lr_df = pd.DataFrame()
            for newmetname in dictre_fsd_lr.values():
                tmp_dct = {'Method':newmetname, 'Start_size': 1,  'EndSize': 1, 'Iteration': 0, 'LogLoss': 1, 'AUROC': 0.5, 'AUPRC': 0.5, 'Accuracy': 0.5, 'Main Model': 'LR'}
                df_dictionary = pd.DataFrame([tmp_dct]).copy()
                fsd_lr_df = pd.concat([fsd_lr_df, df_dictionary])
            lst_to_concat.append(fsd_lr_df)
                
        
        a = pd.concat(lst_to_concat)
        print(lst_to_concat[0].head())
        a['Dataset'] = ds_name_fix[dataset]
        print(a['Method'].unique())
        all_datasets_lst.append(a)
        #print(a.head())
    #accepted_methods = ['RF', 'ST-RF', 'DST-10-RF', 'DST-100-RF', 'KMM-RF', 'NN', 'ST-NN', 'DST-NN-10', 'DST-NN-100', 'KMM-NN', 'Biased-LR', 'KMM-LR', 'KDE-LR', 'RBA-LR', 'FLDA-LR', 'TCPR-LR', 'SUBA-LR']
    accepted_methods = ['RF', 'ST_th-RF', 'ST_kb-RF', 'BST-RF', 'FSD-BST-10-RF', 'FSD-BST-100-RF', 'DBST-10-RF', 'DBST-100-RF', 'NN', 'ST_th-NN', 'ST_kb-NN', 'BST-NN', 'FSD-BST-10-NN', 'FSD-BST-100-NN', 'DBST-10-NN', 'DBST-100-NN','LR',  'ST_th-LR', 'ST_kb-LR', 'BST-LR', 'FSD-BST-10-LR', 'FSD-BST-100-LR']
    color_dict_pre = {'RF':'#000000', 'ST_th-RF':"#DDAA33", 'ST_kb-RF':"#DDAA33", 'BST-RF':"#08519c", 'FSD-BST-10-RF':"#08519c", 'FSD-BST-100-RF':"#08519c", 'DBST-10-RF':"#08519c", 'DBST-100-RF':"#08519c", 
                  'NN':"#000000", 'ST_th-NN':"#DDAA33", 'ST_kb-NN':"#DDAA33", 'BST-NN':"#238b45", 'FSD-BST-10-NN':"#238b45", 'FSD-BST-100-NN':"#238b45", 'DBST-10-NN':"#238b45", 'DBST-100-NN':"#238b45",
                     'LR':'#000000', 'ST_th-LR':"#DDAA33", 'ST_kb-LR':"#DDAA33", 'BST-LR':"#cb181d", 'FSD-BST-10-LR':"#cb181d", 'FSD-BST-100-LR':"#cb181d", }

    bp_color_dict_pre = {'RF':'#FFFFFF', 'ST_th-RF':"#DDAA33", 'ST_kb-RF':"#DDAA33", 'BST-RF':"#6baed6", 'FSD-BST-10-RF':"#6baed6", 'FSD-BST-100-RF':"#6baed6", 'DBST-10-RF':"#6baed6", 'DBST-100-RF':"#6baed6", 
              'NN':"#FFFFFF", 'ST_th-NN':"#DDAA33", 'ST_kb-NN':"#DDAA33", 'BST-NN':"#41ab5d", 'FSD-BST-10-NN':"#41ab5d", 'FSD-BST-100-NN':"#41ab5d", 'DBST-10-NN':"#41ab5d", 'DBST-100-NN':"#41ab5d",
                        'LR':'#FFFFFF', 'ST_th-LR':"#DDAA33", 'ST_kb-LR':"#DDAA33", 'BST-LR':"#ef3b2c", 'FSD-BST-10-LR':"#ef3b2c", 'FSD-BST-100-LR':"#ef3b2c",}

    met_hue = {'RF':'Biased', 'ST_th-RF':"ST", 'ST_kb-RF':"ST", 'BST-RF':"BST", 'FSD-BST-10-RF':"FSD-BST", 'FSD-BST-100-RF':"FSD-BST", 'DBST-10-RF':"DBST", 'DBST-100-RF':"DBST", 
              'NN':"Biased", 'ST_th-NN':"ST", 'ST_kb-NN':"ST", 'BST-NN':"BST", 'FSD-BST-10-NN':"FSD-BST", 'FSD-BST-100-NN':"FSD-BST", 'DBST-10-NN':"DBST", 'DBST-100-NN':"DBST"}
    
    '''
    color_dict = {'RF':'#DDAA33', 'ST-RF':'#bdd7e7', 'DST-10-RF':'#6baed6', 'DST-100-RF':'#3182bd', 'KMM-RF':'#08519c',
                'NN':'#DDAA33', 'ST-NN':'#bae4b3', 'DST-NN-10':'#74c476', 'DST-NN-100':'#31a354', 'KMM-NN':'#006d2c',
                'Biased-LR':'#DDAA33', 'KMM-LR':'#fcbba1', 'KDE-LR':'#fc9272', 'RBA-LR':'#fb6a4a', 'FLDA-LR':'#ef3b2c', 'TCPR-LR':'#cb181d', 'SUBA-LR':'#99000d'}
    
    bp_color_dict = {'RF':'#DDAA33', 'ST-RF':'#08519c', 'DST-10-RF':'#08519c', 'DST-100-RF':'#08519c', 'KMM-RF':'#08519c',
                'NN':'#DDAA33', 'ST-NN':'#006d2c', 'DST-NN-10':'#006d2c', 'DST-NN-100':'#006d2c', 'KMM-NN':'#006d2c',
                'Biased-LR':'#DDAA33', 'KMM-LR':'#99000d', 'KDE-LR':'#99000d', 'RBA-LR':'#99000d', 'FLDA-LR':'#99000d', 'TCPR-LR':'#99000d', 'SUBA-LR':'#99000d'}
    
    met_hue = {'RF':'Biased', 'ST-RF':'SS', 'DST-10-RF':'SS', 'DST-100-RF':'SS', 'KMM-RF':'KMM',
                'NN':'Biased', 'ST-NN':'SS', 'DST-NN-10':'SS', 'DST-NN-100':'SS', 'KMM-NN':'KMM',
                'Biased-LR':'Biased', 'KMM-LR':'KMM', 'RBA-LR':'RBA', 'KDE-LR':'KDE', 'FLDA-LR':'FLDA', 'TCPR-LR':'TCPR', 'SUBA-LR':'SUBA'}
    '''
    
    met_last_names = {'RF':'Biased-RF', 'ST_th-RF': 'ST(0.9)-RF', 'ST_kb-RF': 'ST(6)-RF', 'BST-RF': 'BaST-RF', 'FSD-BST-10-RF':'FDBaST-10-RF', 'FSD-BST-100-RF':'FDBaST-100-RF', 'DBST-10-RF':'DBaST-10-RF', 'DBST-100-RF':'DBaST-100-RF', 
                'NN':'Biased-NN', 'ST_th-NN': 'ST(0.9)-NN', 'ST_kb-NN': 'ST(6)-NN', 'BST-NN': 'BaST-NN', 'FSD-BST-10-NN':'FDBaST-10-NN', 'FSD-BST-100-NN':'FDBaST-100-NN', 'DBST-10-NN':'DBaST-10-NN', 'DBST-100-NN':'DBaST-100-NN',
                     'LR':'Biased-LR', 'ST_th-LR': 'ST(0.9)-LR', 'ST_kb-LR': 'ST(6)-LR', 'BST-LR': 'BaST-LR', 'FSD-BST-10-LR':'FDBaST-10-LR', 'FSD-BST-100-LR':'FDBaST-100-LR'}
    color_dict = {metnewname: color_dict_pre[metname] for metname, metnewname in met_last_names.items()}
    bp_color_dict = {metnewname: bp_color_dict_pre[metname] for metname, metnewname in met_last_names.items()}

    #ds_sort = {"Breast Cancer":0, "Wine":1, "Mushroom":2, "MNIST":3, "Drug":4, 'Adult':5}#'spam', 'adult', 'fire', 'rice', 'raisin', 'pistachio', 'pumpkin'
    #ds_sort = {"Breast Cancer":0, "Wine":1, "Mushroom":2, "MNIST":3, "Drug":4, 'Adult':5, 
    #          "Spam":6, "Adult":7, "Fire":8, "Rice":9, "Raisin":10, 'Pistachio':11, 'Pumpkin':12}
    ds_sort = {ds_name_fix[dssname]:idds for idds, dssname in enumerate(
        datasets)}
    row_no = {key: int(val/colsize) for key,val in ds_sort.items()}
    col_no = {key: int(val%colsize) for key,val in ds_sort.items()}
    met_sort = {met:metenum for metenum, met in enumerate(accepted_methods)}
    #met_sort = {'RF':0, 'ST-RF':1, 'DST-10-RF':2, 'DST-100-RF':3, 'KMM-RF':4,
    #            'NN':5, 'ST-NN':6, 'DST-NN-10':7, 'DST-NN-100':8, 'KMM-NN':9,
    #            'Biased-LR':10, 'KMM-LR':11, 'KDE-LR':12, 'RBA-LR':13, 'FLDA-LR':14, 'TCPR-LR':15, 'SUBA-LR':16}
    
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
    all_ds = all_ds.groupby(["Method", "Dataset", "Iteration"])[['Accuracy', 'LogLoss', 'AUROC', 'AUPRC']].median().reset_index()
    all_ds = all_ds[all_ds['Method'].isin(accepted_methods)]
    all_ds['met_sort'] = all_ds['Method'].map(met_sort)
    all_ds['Category'] = all_ds['Method'].map(met_hue)
    all_ds['ds_sort'] = all_ds['Dataset'].map(ds_sort)
    all_ds['row_no'] = all_ds['Dataset'].map(row_no)
    all_ds['col_no'] = all_ds['Dataset'].map(col_no)
    all_ds['Method'] = all_ds['Method'].map(met_last_names)
    all_ds = all_ds.sort_values(['ds_sort', 'met_sort', 'Iteration'], ascending=[True, True, True])
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
    gf = sns.FacetGrid(all_ds, col='col_no', row='row_no', height=2.4, aspect=0.85, sharex=True, sharey='row', gridspec_kws={"wspace":0.08, 'hspace':0.11})
    gmapped = gf.map_dataframe(fixed_boxplot, metric, "Method", showfliers=not swarmed, width=0.6, palette=bp_color_dict, **PROPS)
    for ax in gf.axes.flat:
        for idd, patch in enumerate(ax.artists):
            r, g, b, a = patch.get_facecolor()
            if swarmed:
                patch.set_facecolor((r, g, b, .1))
            else:
                patch.set_facecolor((r, g, b, .95))
    if swarmed:
        gmapped2 = gf.map_dataframe(sns.swarmplot, metric, "Method", palette=color_dict, size=1, zorder=5)
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
        ax.set_xlabel(ax.get_xlabel(), family='Arial', fontdict={'weight':'bold', 'size':6.5}, labelpad=1.6)
        ax.set_ylabel(ax.get_ylabel(), family='Arial', fontdict={'weight':'bold', 'size':6.5}, labelpad=1.6)
        #ax.set_xticklabels(ax.get_xticklabels(), family='Arial', fontdict={ 'size':5.4})
        #ax.set_yticklabels(ax.get_yticklabels(), family='Arial', fontdict={ 'size':5.4})
        plt.setp(ax.get_xticklabels(), size=5.4)
        plt.setp(ax.get_yticklabels(), size=5.4)
        #ax.tick_params(width=0.3)
        #if ax.get_yticklabels():
        #ax.set_yticklabels(ax.get_yticklabels(), family='Arial', fontdict={ 'size':6})
        #print(ax.get_ylabel())
        if not ax.get_ylabel():
            ax.spines['left'].set_linewidth(0.05)
            ax.tick_params(axis='y', width=0.1, length=1)
        else:
            ax.spines['left'].set_linewidth(0.2)
            ax.tick_params(axis='y', width=0.3, length=1.2, pad=2)
        
        if not ax.get_xlabel():
            ax.spines['bottom'].set_linewidth(0.05)
            ax.tick_params(axis='x', width=0.1, length=1)
        else:
            ax.spines['bottom'].set_linewidth(0.2)
            ax.tick_params(axis='x', width=0.3, length=1.2, pad=2)
        #ax.spines['bottom'].set_linewidth(0.2)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        #ax.title.set_position([.5, 0.9])
        print(ax.get_title())
        row_id = int(ax.get_title()[9])
        col_id = int(ax.get_title()[-1])
        
        try:
            ax.set_title(row_col_dict[row_id][col_id], family='Arial', size=6.5, pad=-0.6)
            y_axis_ticks = ax.get_yaxis().get_majorticklocs()
            ax.axhline(y=7.5, c='black', linewidth=0.2)#, ls='--')
            ax.axhline(y=15.5, c='black', linewidth=0.2)#, ls='--')
        except:
            print('This facet is empty')
            ax.set_title('')
            ax.set_axis_off()
        #for patch in ax.artists:
        #    r, g, b, a = patch.get_facecolor()
        #    patch.set_facecolor((r, g, b, .3))
    
    #gf.set_titles(col_template="{col_name}", family='Arial', size=6.5, pad=-0.6)
    #plt.tight_layout()#(pad=0.7)
    
    out_loc = res_folder / f'SSL97_Multids_{bias["name"]}_{metric}_{args.val_base}_facet_h{mutids_name}_s={swarmed}_ss8_v6.png'
    out_loc_pdf = res_folder / f'SSL97_Multids_{bias["name"]}_{metric}_{args.val_base}_facet_h{mutids_name}_s={swarmed}_ss8_v6.pdf' 
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
if args.d_group==0:
    datasets = ['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'rice', 'fire']
elif args.d_group==1:
    datasets = ['adult', 'spam', 'raisin', 'pistachio', 'pumpkin', 'fire']
elif args.d_group==2:
    datasets = ['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'rice', 'fire', 'spam', 'adult', 'raisin', 'pistachio', 'pumpkin']
none_hierarchy_multi_datasets_facet(datasets, metric='LogLoss', swarmed=True)#model_dict, out_loc)
none_hierarchy_multi_datasets_facet(datasets, metric='Accuracy', swarmed=True)#model_dict, out_loc)
#none_hierarchy_multi_datasets_facet(datasets, metric='LogLoss', swarmed=False)#model_dict, out_loc)
#none_hierarchy_multi_datasets_facet(datasets, metric='Accuracy', swarmed=False)#model_dict, out_loc)
#none_hierarchy_multi_datasets_facet(datasets, metric='AUROC')#model_dict, out_loc)
#none_hierarchy_multi_datasets_facet(datasets, metric='AUPRC')#model_dict, out_loc)