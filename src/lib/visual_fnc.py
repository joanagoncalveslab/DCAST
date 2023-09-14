import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

def save_boxplot_old(res_df, model_dict, is_show=False, metric='LogLoss'):
    plt.clf()
    plt.rcParams["figure.figsize"] = (8, 6) #'SU', 'DST-None', 'DST-5', 'DST-10', 'DST-20', 'DST-50'
    bp_df = pd.DataFrame(0, columns=['SU', 'DST-None', 'DST-10', 'DST-50', 'DST-100', 'DST-200'], index=np.unique(res_df.Iteration))
    for col in bp_df.columns:
        bp_df[col] = res_df[res_df['Method'] == col].set_index('Iteration').loc[bp_df.index][metric].values
    bp_df.plot(kind='box',
             color=dict(boxes='r', whiskers='r', medians='r', caps='r'),
             boxprops=dict(linestyle='-', linewidth=1.5),
             flierprops=dict(linestyle='-', linewidth=1.5),
             medianprops=dict(linestyle='-', linewidth=1),
             whiskerprops=dict(linestyle='-', linewidth=1.5),
             capprops=dict(linestyle='-', linewidth=1.5),
             showfliers=False, grid=False)
 

def save_boxplot(res_df, metric='LogLoss'):   
    PROPS = {
        'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
        'medianprops': {'color': 'orange', 'linewidth':0.8},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }
    rc = {'xtick.bottom': True, 'xtick.left': True}
    sns.axes_style(style="white", rc=rc) #font='Arial'

    ax1 = sns.boxplot(x="Method", y=metric, data=res_df, showfliers=False, width=0.6, **PROPS)
    ax2 = sns.swarmplot(x="Method", y=metric, data=res_df, size=3, color="#2596be")
    ax1.set_ylabel(ax1.get_ylabel(), rotation=0)
    ax1.yaxis.set_label_coords(-.075, 1.05)
    sns.despine()
    

def save_boxplot_sign(res_df, metric='LogLoss'):
    PROPS = {
        'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
        'medianprops': {'color': 'orange', 'linewidth':0.8},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }
    rc = {'xtick.bottom': True, 'xtick.left': True}
    sns.axes_style(style="white", rc=rc) #font='Arial'

    ax1 = sns.boxplot(x="Method", y=metric, data=res_df, showfliers=False, width=0.6, **PROPS)
    ax2 = sns.swarmplot(x="Method", y=metric, data=res_df, size=3, color="#2596be")
    ax1.set_ylabel(ax1.get_ylabel(), rotation=0)
    ax1.yaxis.set_label_coords(-.075, 1.05)
    rand_res = res_df[res_df['Method'] == 'Biased']
    su_res = res_df[res_df['Method'] == 'DST-100']
    common_iters = np.intersect1d(rand_res['Iteration'].values, su_res['Iteration'].values)
    rand_vals = rand_res.set_index('Iteration').loc[common_iters,metric].values
    su_vals = su_res.set_index('Iteration').loc[common_iters,metric].values

    w, p = wilcoxon(rand_vals, su_vals)
    label_txt = f'Pval: {p:.2e}'
    plt.text((1.0 + 4.0) / 2.0, min(min(rand_vals),min(su_vals))-0.04, label_txt, horizontalalignment='center',
             verticalalignment='top', size='medium', color='black', weight='semibold')
    plt.hlines(y=min(min(rand_vals),min(su_vals))-0.03, xmin=1, xmax=4, color='red')
    plt.vlines(x=1, ymin=min(min(rand_vals),min(su_vals))-0.01, ymax=min(min(rand_vals),min(su_vals))-0.03, color='red')
    plt.vlines(x=4, ymin=min(min(rand_vals),min(su_vals))-0.01, ymax=min(min(rand_vals),min(su_vals))-0.03, color='red')
    sns.despine()


def save_boxplot_match(res_df, metric='LogLoss'):
    PROPS = {
        'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
        'medianprops': {'color': 'orange', 'linewidth': 0.8},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }
    rc = {'xtick.bottom': True, 'xtick.left': True}
    sns.axes_style(style="white", rc=rc)  # font='Arial'
    my_pal = {"No\nBias": "", "setosa": "b", "virginica": "m"}
    ax1 = sns.boxplot(x="Method", y=metric, data=res_df, showfliers=False, width=0.6, **PROPS)
    ax2 = sns.swarmplot(x="Method", y=metric, data=res_df, size=3, color="#2596be")
    try:
        ax2.collections[3].remove()
        ax2.collections[2].remove()
        ax1.patches[3].remove()
        ax1.patches[2].remove()
        [line.remove() for line in ax1.lines[10:20]]
    except:
        pass

    ax1.set_ylabel(ax1.get_ylabel(), rotation=0)
    ax1.yaxis.set_label_coords(-.075, 1.05)

    su_df = res_df[res_df['Method'] == 'Biased'].set_index('Iteration')
    dst100_df = res_df[res_df['Method'] == 'DST-100'].set_index('Iteration')
    for fold in np.intersect1d(su_df.index,dst100_df.index):
        strt = su_df.loc[fold, metric]
        end = dst100_df.loc[fold, metric]
        plt.plot((1, 4), (strt, end))

    rand_res = res_df[res_df['Method'] == 'Biased']
    su_res = res_df[res_df['Method'] == 'DST-100']
    common_iters = np.intersect1d(rand_res['Iteration'].values, su_res['Iteration'].values)
    rand_vals = rand_res.set_index('Iteration').loc[common_iters,metric].values
    su_vals = su_res.set_index('Iteration').loc[common_iters,metric].values

    w, p = wilcoxon(rand_vals, su_vals)
    label_txt = f'Pval: {p:.2e}'
    plt.text((1.0 + 4.0) / 2.0, min(min(rand_vals),min(su_vals))-0.04, label_txt, horizontalalignment='center',
             verticalalignment='top', size='medium', color='black', weight='semibold')
    plt.hlines(y=min(min(rand_vals),min(su_vals))-0.03, xmin=1, xmax=4, color='red')
    plt.vlines(x=1, ymin=min(min(rand_vals),min(su_vals))-0.01, ymax=min(min(rand_vals),min(su_vals))-0.03, color='red')
    plt.vlines(x=4, ymin=min(min(rand_vals),min(su_vals))-0.01, ymax=min(min(rand_vals),min(su_vals))-0.03, color='red')
    sns.despine()



def save_bar_old(res_df, model_dict, is_show=False, metric='LogLoss'):
    qual = res_df.groupby(["Method"])[metric].agg([np.mean, np.std]).reset_index()
    qual = qual.set_index('Method').loc[['SU', 'DST-None', 'DST-10', 'DST-50', 'DST-100', 'DST-200']].reset_index()
    qual.plot('Method', 'mean', yerr='std', kind='bar')

    
def save_bar(res_df, metric='LogLoss'):
    rc = {'xtick.bottom': True, 'xtick.left': True}
    sns.axes_style(style="white", rc=rc) #font='Arial'

    ax1 = sns.barplot(x="Method", y=metric, data=res_df)#, **PROPS)
    #ax2 = sns.swarmplot(x="Method", y=metric, data=res_df, size=3, color="#2596be")
    ax1.set_ylabel(ax1.get_ylabel(), rotation=0)
    ax1.yaxis.set_label_coords(-.08, 1.05)
    sns.despine()

def save_stacked_bar(res_df, out_loc, title='', x_ticks='data_type', stacks='class'):
    df = res_df.groupby([x_ticks, stacks])['size'].agg([np.mean, np.std]).reset_index()
    #df = res_df.drop(columns=['fold_id'])
    plt.clf()
    fig, ax = plt.subplots()
    df = df.pivot(index=x_ticks, columns=stacks).fillna(0).stack().reset_index()

    colors = ['green', 'red', 'blue']
    positions = [-0.5, 0.5, 1.5]

    for group, color, pos in zip(df.groupby(stacks), colors, positions):
        key, group = group
        group = group.set_index(x_ticks).loc[['unk', 'test', 'train', 'biased', 'trainb', 'valb']].reset_index()
        #print(group)
        group.plot(x_ticks, 'mean', yerr='std', kind='bar', width=0.2, label=key,
                   position=pos, color=color, alpha=0.5, ax=ax)

    plt.ylabel('Number of Samples')
    plt.xlabel('Data Splits')
    plt.title(title)
    ax.set_xlim(-1, 6)
    plt.savefig(out_loc, type='png', dpi=300, bbox_inches='tight')


def save_selected_umap(X, y, sel_ids, out_loc, max_class_size=3, suptitle=''):
    non_sel_ids = np.array([ elem for elem in np.arange(len(y)) if elem not in sel_ids])
    plt.clf()
    fig, ax = plt.subplots()
    marker_s = float(len(y))/50.0
    marker_s = max(math.sqrt(len(y))/3.0,10)
    title = ''
    faded_colors = ["#EE99AA", "#6699CC", '#9bd4a4'] # Faded red, Faded blue,
    real_colors = ["#994455", "#004488", '#25612f'] # Real red, Real Blue
    classes = np.unique(y)[:max_class_size]
    for class_idx, class_name in enumerate(classes):
        ax.scatter(X[non_sel_ids[y[non_sel_ids] == class_name], 0], X[non_sel_ids[y[non_sel_ids] == class_name], 1], c=faded_colors[class_idx],
                       s=marker_s, label=f'{class_name}')
    for class_idx, class_name in enumerate(classes):
        class_sel_ids = sel_ids[y[sel_ids] == class_name]
        title = f'{title}Class_{class_name}: {len(class_sel_ids)}/{sum(y==class_name)} '
        ax.scatter(X[class_sel_ids, 0], X[class_sel_ids, 1], c=real_colors[class_idx],
                       s=marker_s, label=f'{class_name}_selected')

    fig.suptitle(suptitle, y=1.01)
    plt.title(title, y=-0.01)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(title)
    plt.legend()
    plt.savefig(out_loc, type='png', dpi=300, bbox_inches='tight')
    #plt.show()

    
def call_plot(plot_type, save_loc, res_df, model_dict, is_show=False, metric='LogLoss'):
    if os.path.exists(save_loc):
        pass
    plt.clf()
    plt.rcParams["figure.figsize"] = (8, 6)
    if plot_type == 'bar':
        save_bar(res_df, metric=metric)
    if plot_type == 'boxplot':
        save_boxplot(res_df, metric=metric)
    if plot_type == 'boxplot_sign':
        save_boxplot_sign(res_df, metric=metric)
    if plot_type=='boxplot_match':
        save_boxplot_match(res_df, metric=metric)

    model_str = f'{model_dict["params"]["model"]["full_name"]}' \
                f'(th={model_dict["params"]["model"]["threshold"]}' \
                f'|kb={model_dict["params"]["model"]["k_best"]}' \
                f'|mi={model_dict["params"]["model"]["max_iter"]}' \
                f'|b={model_dict["params"]["model"]["balance"]})'
    title = f'{model_dict["params"]["dataset"]["name"]} - {model_str}'
    print(res_df.head())
    max_val_tmp = res_df[res_df['Method'] == 'SU'][metric].max()+0.1
    if metric == 'Accuracy':
        min_val_tmp = res_df[res_df['Method'] == 'SU'][metric].min()-0.1
    else:
        min_val_tmp = res_df[res_df['Method'] == 'SU'][metric].min()-0.15
    
    if 'boxplot' in plot_type:
        max_val_tmp = res_df[metric].max() + 0.1
    if metric == 'LogLoss':
        min_val_tmp = res_df[metric].min() - 0.15
    elif metric == 'Accuracy':
        min_val_tmp = res_df[metric].min() - 0.1
    min_val_tmp = max(0.0000001,min_val_tmp)
    
    val_lst = np.arange(0,10,0.1)
    min_val = val_lst[np.searchsorted(val_lst,min_val_tmp)-1]
    print(min_val_tmp)
    print(max_val_tmp)
    max_val = val_lst[np.searchsorted(val_lst,max_val_tmp)]
    if metric == 'Accuracy':
        max_val = min(1, max_val)

    plt.ylim((min_val, max_val))
    ticks =np.arange(min_val, max_val+0.01, 0.1)
    tick_labels = ["%.2f" % number for number in ticks]
    plt.yticks(ticks, labels=tick_labels)
    #plt.ylabel(metric)
    #plt.xlabel('Methods')
    #plt.title(title)
    plt.savefig(save_loc, dpi=300, bbox_inches='tight')
