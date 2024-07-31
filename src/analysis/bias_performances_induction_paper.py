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
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "Arial"

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
    #print(f'{ds_folder_name}/{out_loc} found')
    #except:
    #    if args.res_fold == 'results_test_nb_imb':
    #        res_loc = config.ROOT_DIR / 'resultsxxx' / f'{bias_name}' / ds_folder_name / f'{out_loc}.csv'
    #        res_df = pd.read_csv(res_loc)
    return res_df


def create_model_dict(ds_name):
    bias_per_class = args.bias_size
    dataset = {'name': ds_name,'args':{}, 'order': 'train_test_bias_validation', 'train': 0.8, 'test': 0.2, 'val': 0.2, 'runs': 30}
    if 'drug_' in ds_name:
        dataset = {'name': 'drug','args':{'drug': ds_name.split('_')[1]}, 'order': 'train_test_bias_validation',
                                                           'train': 0.8, 'test': 0.2, 'val': 0.2, 'runs': 30}
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
        dataset = {'name': ds_name, 'args': {}, 'order': 'train_test_bias_validation', 'train': 0.8, 'test': 0.2, 'val': 0.2, 'runs': 30}
        
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
             f'|{model_dict["params"]["dataset"]["test"]})'
    model_str = f'supervised|vb={model_dict["params"]["model"]["val_base"]})'
    out_loc = f'{bias_str}_{ds_str}_{model_str}_es=True'
    
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
    
    
def none_vs_all_multi_datasets_facet(datasets, base_m, bias_list, metric="LogLoss"):
    mutids_name = '|'.join(datasets)
    width_bp = 0.8
    alpha=0.95
    seed_run_size = 10
    if base_m == 'RF':
        long_base_m = 'Random Forest'
    elif base_m == 'NN':
        long_base_m = 'Neural Network'
    print(f'{metric} started')
    plt.clf()
    #bias_per_class = args.bias_size
    #main_bias = get_bias_dict()
    #bias = main_bias.copy()
    res_folder = config.ROOT_DIR / 'bias_supervised_comparisons_only_supervised' 
    all_datasets_lst = []
    ds_name_fix = {'breast_cancer':"Breast Cancer", 'wine_uci2': "Wine", "mushroom": "Mushroom", "mnist": "MNIST", "drug_CX-5461": "Drug", 'spam': 'Spam', 
                   'adult': 'Adult', 'rice': 'Rice', 'fire': 'Fire', 'pumpkin': 'Pumpkin', 'pistachio': 'Pistachio', 'raisin': 'Raisin'}
    for dataset in datasets:
        print(f'Dataset {dataset} started!')
        lst_to_concat =[]
        kb=6
        u_classes = 2
        th_rf=97
        ds_str = f'{dataset}({0.8}|{0.2}|{0.2})'
        if 'drug_' in dataset:
            ds_str = f'drug({0.8}|{0.2}|{0.2})'
        if 'mnist' in dataset:
            th_rf=85
            kb=30
            u_classes=10
        
        #Main RF
        for bias_name in bias_list:# 0.6, 0.7, 0.8, 0.9]: # hierarchyy(True|30|0.5)_adult(0.8|0.2|0.2)_supervised|vb=False)_es=True_(0, 30)
            bias = get_bias_dict(bias_name)
            model_rf_str = f'supervised|vb=False)_es=True'
            if dataset=='mnist':
                model_rf_str = f'supervised|vb=False)_es=True'
            if bias_name =='dirichlet':
                bias['n']=args.bias_size*2
                if dataset=='mnist':
                    bias['n']=args.bias_size*10
            bias_str = f'{bias["name"]}({"|".join([str(val) for key, val in bias.items() if "name" not in key])})'
            out_rf_loc = f'{bias_str}_{ds_str}_{model_rf_str}'
            rf_df = get_res('results_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_rf_loc)
            rf_df = rf_df[rf_df['Method']=='SU']
            dictre = {"SU": 'Biased-RF'}#, "DST-10": 'H', "DST-100": 'P'}
            rf_df = rf_df.replace({"Method": dictre})
            rf_df['Bias'] = bias_name
            rf_df['Main Model'] = 'RF'
            lst_to_concat.append(rf_df)
        
        #Main NN
        for bias_name in bias_list:# 0.6, 0.7, 0.8, 0.9]:
            bias = get_bias_dict(bias_name)
            if bias_name =='dirichlet':
                bias['n']=args.bias_size*2
                if dataset=='mnist':
                    bias['n']=args.bias_size*10
            model_nn_str = f'supervised|vb=False)_es=True_(0, 30)'
            bias_str = f'{bias["name"]}({"|".join([str(val) for key, val in bias.items() if "name" not in key])})'
            out_nn_loc = f'{bias_str}_{ds_str}_{model_nn_str}'
            nn_df = get_res('results_nn_test_nb_imb_fin_cw3_ss8', f'{bias["name"]}', dataset, out_nn_loc)
            nn_df = nn_df[nn_df['Method']=='SU']
            dictre = {"SU": 'Biased-NN', "DST-None": 'ST-NN', 'DST-10':'DST-NN-10', 'DST-100':'DST-NN-100'}#, "DST-10": 'H', "DST-100": 'P'}
            nn_df = nn_df.replace({"Method": dictre})
            nn_df['Bias'] = bias_name
            nn_df['Main Model'] = 'NN'
            lst_to_concat.append(nn_df)
         
        #Main LR
        for bias_name in bias_list:# 0.6, 0.7, 0.8, 0.9]:
            bias = get_bias_dict(bias_name)
            if bias_name =='dirichlet':
                bias['n']=args.bias_size*2
                if dataset=='mnist':
                    bias['n']=args.bias_size*10
            model_lr_str = f'supervised|vb=False)_30_es=False'
            lr_ds_str = f'{dataset}({0.3}|{0.2}|{0.2})'
            bias_str = f'{bias["name"]}({"|".join([str(val) for key, val in bias.items() if "name" not in key])})'
            out_lr_loc = f'{bias_str}_{lr_ds_str}_{model_lr_str}'
            lr_df = get_res('results_lr_fsd_test_nb_imb_ss8', f'{bias["name"]}', dataset, out_lr_loc)
            lr_df = lr_df[lr_df['Method']=='SU']
            dictre = {"SU": 'Biased-LR', "DST-None": 'ST-LR', 'DST-10':'DST-LR-10', 'DST-100':'DST-LR-100'}#, "DST-10": 'H', "DST-100": 'P'}
            lr_df = lr_df.replace({"Method": dictre})
            lr_df['Bias'] = bias_name
            lr_df['Main Model'] = 'LR'
            lst_to_concat.append(lr_df)
                
        a = pd.concat(lst_to_concat)
        print(lst_to_concat[0].head())
        a['Dataset'] = ds_name_fix[dataset]
        all_datasets_lst.append(a)
        #print(a.head())
    met_sort = {'No Bias':0, 'Biased-RF':1, 'ST-RF':2, 'DST-RF-10':3, 'DST-RF-100':4, 
                'Unbiased-NN':5, 'Biased-NN':6, 'ST-NN':7, 'DST-NN-10':8, 'DST-NN-100':9,
                'Unbiased-LR':10, 'Biased-LR':11, 'ST-LR':12, 'DST-LR-10':13, 'DST-LR-100':14}
    bias_last_names = {'none': 'None', 'dirichlet': 'Dirichlet', 'joint': 'Joint', 'random': 'Random','hierarchyy9': 'Hierarchy'}
    for strength in np.arange(0.1, 1.0, 0.1).round(1):
        bias_last_names[f'hierarchyy_{strength}'] = f'Hierarchy ({strength})'
    print(bias_last_names)
    bias_sort = {biasn:biasnum for biasnum, biasn in enumerate(bias_list)}
    color_dict = {'Unbiased-RF':'b', 'Biased-RF':'r', 'ST-RF':'g', 'DST-RF-10':'g', 'DST-RF-100':'g',
                'Unbiased-NN':'b', 'Biased-NN':'r', 'ST-NN':'g', 'DST-NN-10':'g', 'DST-NN-100':'g'}
    met_hue = {'Unbiased-RF':'Unbiased', 'Biased-RF':'Biased', 'ST-RF':'SS', 'DST-RF-10':'SS', 'DST-RF-100':'SS', 'KMM-RF':'KMM',
                'Unbiased-NN':'Unbiased', 'Biased-NN':'Biased', 'ST-NN':'SS', 'DST-NN-10':'SS', 'DST-NN-100':'SS', 'KMM-NN':'KMM',
                'Biased-LR':'Biased', 'KMM-LR':'KMM', 'RBA-LR':'RBA'}
    met_last_names = {'Unbiased-RF':'RF, No Bias', 'Biased-RF':'RF, Bias', 'ST-RF': 'BaST, Bias', 'DST-RF-10':'DBaST-10, Bias', 'DST-RF-100':'DBaST-100, Bias', 
                      'Unbiased-NN':'NN, No Bias', 'Biased-NN':'NN, Bias', 'ST-NN':'BaST, Bias', 'DST-NN-10':'DBaST-10, Bias', 'DST-NN-100':'DBaST-100, Bias' }
    last_color_dict = {cval: color_dict[ckey] for ckey, cval in met_last_names.items()}
    ds_sort = {ds_name_fix[datasets[dsi]]:dsi for dsi in range(len(datasets))}#{"Breast Cancer":0, "Wine":1, "Mushroom":2, "MNIST":3, "Drug":4, "Adult":5, 'Spam':6}
    all_ds = pd.concat(all_datasets_lst)
    all_ds = all_ds.groupby(["Method", "Dataset", "Iteration", "Bias", "Main Model"])[['Accuracy', 'LogLoss', 'AUROC', 'AUPRC']].median().reset_index()
    all_ds['met_sort'] = all_ds['Method'].map(met_sort)
    all_ds['bias_sort'] = all_ds['Bias'].map(bias_sort)
    all_ds['Category'] = all_ds['Method'].map(met_hue)
    all_ds['ds_sort'] = all_ds['Dataset'].map(ds_sort)
    all_ds['Method'] = all_ds['Method'].map(met_last_names)
    all_ds['Bias'] = all_ds['Bias'].map(bias_last_names)
    all_ds = all_ds.sort_values(['bias_sort','ds_sort', 'Iteration'], ascending=[True, True, True])
    print(all_ds.head())
    print(all_ds['Bias'].unique())
    #print(all_ds.head())
    #print(all_ds.shape)
    #out_loc = res_folder / f'Multids_{base_m}_{bias["name"]}_{metric}_{args.val_base}_facet8.png'
    #out_loc_pdf = res_folder / f'Multids_{base_m}_{bias["name"]}_{metric}_{args.val_base}_facet8.pdf'
    #config.ensure_dir(out_loc)
    PROPS = {
        'boxprops': {'edgecolor': 'black', 'linewidth':0.2},
        'medianprops': {'color': 'black', 'linewidth':0.3},
        'whiskerprops': {'color': 'black', 'linewidth':0.2 },
        'capprops': {'color': 'black', 'linewidth':0.2 },
        'flierprops': {'marker':'o', 'markerfacecolor':'black', 'markeredgecolor':'black', 'markersize':0.6, 'linestyle':'none', 'markeredgewidth':0.2}
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
    gf = sns.FacetGrid(all_ds, col='Dataset', row='Main Model', height=0.95, aspect=0.81, sharex='col', sharey='row', gridspec_kws={"wspace":0.08, 'hspace':0.25})
    color_lst = ['#ffffff', '#004488', '#004488','#004488']
    gmapped = gf.map_dataframe(fixed_boxplot, 'Bias', metric, showfliers=True, width=0.6, color='white', **PROPS)#, palette=last_color_dict, **PROPS)
    #gmapped2 = gf.map_dataframe(sns.swarmplot, metric, 'Bias', size =1)# palette=last_color_dict, size=1)

    
    for ax in gf.axes.flat:
        ax.set(ylim=(0.5, 1.0))
        ax.set_xlabel('', family='Arial', fontdict={'weight':'bold', 'size':5.5}, labelpad=1.6)
        ax.set_ylabel(ax.get_ylabel(), family='Arial', fontdict={'weight':'bold', 'size':5.5}, labelpad=1.6)
        ax.set_xticklabels(ax.get_xticklabels(), family='Arial', fontdict={ 'size':5})
        plt.setp(ax.get_xticklabels(), rotation=85)
        plt.setp(ax.get_yticklabels(), size=5)
        #ax.tick_params(width=0.3)
        #if ax.get_yticklabels():
        #ax.set_yticklabels(ax.get_yticklabels(), family='Arial', fontdict={ 'size':6})
        if not ax.get_ylabel():
            ax.spines['left'].set_linewidth(0.1)
            ax.tick_params(axis='y', width=0.2, length=1, pad=2)
        else:
            ax.spines['left'].set_linewidth(0.2)
            ax.tick_params(axis='y', width=0.2, length=1, pad=2)
        
        if not ax.get_xlabel():
            ax.spines['bottom'].set_linewidth(0.1)
            ax.tick_params(axis='x', width=0.2, length=1, pad=2)
        else:
            ax.spines['bottom'].set_linewidth(0.2)
            ax.tick_params(axis='x', width=0.2, length=1, pad=2)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.title.set_position([.5, 0.8])
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 1.0))
    
    gf.set_titles(template = '{col_name}', family='Arial', size=5.2, pad=-0.2) 
    #plt.tight_layout()#(pad=0.7) 
    
    out_loc = res_folder / f'Multids_{base_m}_{bias["name"]}_{metric}_{args.val_base}_facet{mutids_name}_ss8_strength.png'
    out_loc_pdf = res_folder / f'Multids_{base_m}_{bias["name"]}_{metric}_{args.val_base}_facet{mutids_name}_ss8_strength.pdf'
    config.ensure_dir(out_loc)
    plt.savefig(out_loc, dpi=300, bbox_inches='tight', pad_inches = 0.01)
    plt.savefig(out_loc_pdf, dpi=300, bbox_inches='tight', pad_inches = 0.01)
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
#bias_list = ['none', 'random', 'dirichlet', 'joint', 'hierarchyy9']
bias_list = ['none', 'hierarchyy_0.5', 'hierarchyy_0.6', 'hierarchyy_0.7', 'hierarchyy_0.8', 'hierarchyy_0.9']
if args.d_group==0:
    datasets = ['breast_cancer', 'wine_uci2', 'mushroom', 'mnist', 'rice', 'fire']
elif args.d_group==1:
    datasets = ['adult', 'spam', 'raisin', 'pistachio', 'pumpkin', 'fire']
elif args.d_group==2:
    datasets = ['adult', 'breast_cancer', 'fire', 'mnist', 'mushroom', 'pistachio', 'pumpkin', 'raisin', 'rice', 'spam', 'wine_uci2']
    #datasets = ['breast_cancer', 'wine_uci2', 'mushroom', 'rice', 'fire', 'adult', 'spam', 'raisin', 'pistachio', 'pumpkin']
for base_m in ['all']:#['nn', 'rf']:
    #none_vs_all_multi_datasets_facet(datasets, base_m=base_m, metric='LogLoss')#model_dict, out_loc)
    none_vs_all_multi_datasets_facet(datasets, base_m=base_m, bias_list= bias_list, metric='Accuracy')#model_dict, out_loc)
    #none_hierarchy_multi_datasets_facet(datasets, base_m=base_m, metric='AUROC')#model_dict, out_loc)
    #none_hierarchy_multi_datasets_facet(datasets, base_m=base_m, metric='AUPRC')#model_dict, out_loc)