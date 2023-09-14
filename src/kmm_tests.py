import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.semi_supervised import SelfTrainingClassifier
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
from src.models import kmm
import random

R_STATE=123

def selection_bias_by_feature_trial(dataset_name, n_iter=30,
                                    threshold=0.95, k_best=3, max_iter=100, balance='free', diverse=None):
    auprc_dict = {}
    res= []
    test_percentage = 0.70
    val_percentage = 0.1
    res_loc = f'kmm_featbias_{dataset_name}({test_percentage}_{val_percentage})_DST-RF_th={threshold}|kb={k_best}|mi={max_iter}|b={balance}|d={diverse}.csv'
    if os.path.exists(res_loc):
        res_df = pd.read_csv(res_loc)
    else:
        X, y = ld.load_dataset(dataset_name)
        #threshold, k_best, max_iter, balance, diverse = 0.95, 3, 100, 'free', None
        for feature_id in range(3):#range(X.shape[1]):
            auprc_st_res = []
            logloss_st_res = []
            auprc_su_res = []
            logloss_su_res = []
            auprc_kmm_res = []
            logloss_kmm_res = []
            for fold in range(n_iter):
                X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, test_size=test_percentage,
                                                     random_state=R_STATE+fold)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, stratify=y_train, test_size=val_percentage,
                                                     random_state=R_STATE+fold)
                selected_ids = bt.bias_select_by_feature(X_train, feature_id=feature_id)
                mask = np.ones_like(y_train, bool)
                mask[selected_ids] = False
                y_st_train = y_train.copy()
                y_st_train[mask] = -1
                y_su_train = y_train[selected_ids]
                X_su_train = X_train[selected_ids,:]

                print(f'Number of samples at the beginning for feature {feature_id} & run {fold}:{len(y_su_train)}')

                # KMM Approach
                #weights = kmm.getBeta(X_su_train, X_test, X_su_train.shape[1])
                weights = kmm.kernel_mean_matching(X_test, X_su_train, kern='rbf' )[:,0]
                #svc_kmm = RandomForestClassifier(class_weight=weights, random_state=R_STATE)
                #svc_kmm.fit(X_su_train, y_su_train)
                svc_kmm = SVC(kernel='rbf', gamma=0.1, probability=True)
                svc_kmm.fit(X_su_train, y_su_train, sample_weight=weights)
                probs_kmm_test = svc_kmm.predict_proba(X_test)
                auprc_kmm = average_precision_score(y_test, probs_kmm_test[:,1])
                loss_kmm = log_loss(y_test, probs_kmm_test)
                auprc_kmm_res.append(auprc_kmm)
                logloss_kmm_res.append(loss_kmm)

                #svc = SVC(probability=True, gamma=0.001, random_state=R_STATE, class_weight='balanced')
                svc = SVC(kernel='rbf', gamma=0.1, probability=True)
                svc.fit(X_su_train, y_su_train)
                #svc = RandomForestClassifier(class_weight=None, random_state=R_STATE)
                #svc.fit(X_su_train, y_su_train)
                probs_su_test = svc.predict_proba(X_test)
                auprc_su = average_precision_score(y_test, probs_su_test[:,1])
                loss_su = log_loss(y_test, probs_su_test)
                auprc_su_res.append(auprc_su)
                logloss_su_res.append(loss_su)

                #base_classifier = SVC(probability=True, gamma=0.001, random_state=R_STATE)
                #base_classifier = RandomForestClassifier(class_weight=None, random_state=R_STATE)
                #self_training_clf = SelfTrainingClassifier(base_classifier, criterion='k_best', k_best=10, max_iter=10, verbose=False)
                #self_training_clf = SelfTrainingClassifier(base_classifier, criterion='threshold',threshold=0.95, max_iter=50,
                #                                           verbose=False)
                #self_training_clf = StandardSelfTraining(base_classifier, threshold=0.95, k_best=5, max_iter=30, verbose=False)
                #self_training_clf = FreeSelfTraining(base_classifier, threshold=0.90, k_best=10, max_iter=30, balance='equal', verbose=False)
                #self_training_clf = FreeDiverseSelfTraining(base_classifier, threshold=threshold, k_best=k_best, max_iter=max_iter, balance=balance,
                #                                     diverse=diverse, verbose=False)
                #self_training_clf.fit(X_train, y_st_train, X_val, y_val)
                #for clff in self_training_clf.estimator_list:
                #    probs_ff = clff.predict_proba(X_test)
                #    auprc_ff = average_precision_score(y_test, probs_ff[:, 1])
                #    loss_ff = log_loss(y_test, probs_ff)
                #    print(loss_ff)

                #print(f'Number of samples at the end:{self_training_clf.labeled_sample_size}')
                #probs_st_test = self_training_clf.predict_proba(X_test)
                #auprc_st = average_precision_score(y_test, probs_st_test[:,1])
                #loss_st = log_loss(y_test, probs_st_test)
                #auprc_st_res.append(auprc_st)
                #logloss_st_res.append(loss_st)

                #print(f'Self-training Fold {fold} AUPRC score: {auprc_st}')
                print(f'Supervised Fold {fold} AUPRC score: {auprc_su}\n')
                #print(f'Self-training Fold {fold} Log loss: {loss_st}')
                print(f'Supervised Fold {fold} Log loss: {loss_su}\n')
                #res.append(['DST', feature_id, fold, loss_st, auprc_st])
                res.append(['SU', feature_id, fold, loss_su, auprc_su])
                res.append(['KMM', feature_id, fold, loss_kmm, auprc_kmm])
            print(f'Mean AUPRC score of self-training: {np.average(auprc_st_res)}')
            print(f'Mean AUPRC score of supervised: {np.average(auprc_su_res)}')

            auprc_dict[feature_id] = {'st_auprc':auprc_st_res, 'su_auprc':auprc_su_res,
                                      'st_logloss':logloss_st_res, 'su_logloss':logloss_su_res}

        res_df = pd.DataFrame(res, columns=['Method', 'Feature', 'Iteration', 'LogLoss', 'AUPRC'])
        res_df.to_csv(res_loc, index=None)
    #qual = res_df.groupby(["Feature", "Method"]).agg({'LogLoss': [np.mean, np.std], 'AUPRC': [np.mean, np.std]})
    qual_loss = res_df.groupby(["Feature", "Method"])['LogLoss'].agg([np.mean, np.std]).reset_index()
    qual_auprc = res_df.groupby(["Feature", "Method"])['AUPRC'].agg([np.mean, np.std]).reset_index()

    plt.clf()
    plt.rcParams["figure.figsize"] = (8, 6)
    #plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    fig, ax = plt.subplots()
    colors = ['green', 'red', 'blue']
    positions = [0, 1, 2]

    for group, color, pos in zip(qual_loss.groupby('Method'), colors, positions):
        key, group = group
        print(group)
        group.plot('Feature', 'mean', yerr='std', kind='bar', width=0.4, label=key,
                   position=pos, color=color, alpha=0.5, ax=ax)

    ax.set_xlim(-1, 3)
    #ax.set_ylim(0, )
    title = f'{dataset_name} - DST-RF(th={threshold}|kb={k_best}|mi={max_iter}|b={balance}|d={diverse})'
    plt.title(title)
    png_loc = f'kmm_featbias_{dataset_name}({test_percentage}_{val_percentage})_DST-RF_th={threshold}|kb={k_best}|mi={max_iter}|b={balance}|d={diverse}.png'
    plt.savefig(png_loc, type='png', dpi=300, bbox_inches='tight')

    plt.show()
    print(auprc_dict)

selection_bias_by_feature_trial('breast_cancer', n_iter=30,
                                threshold=0.95, k_best=3, max_iter=200, balance='free', diverse=None)