import os
import sys
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower()=='diversepsuedolabeling':
        project_path = '/'.join(path2this[:i+1])
sys.path.insert(0,project_path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import MinMaxScaler
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
import random
import optuna
from optuna.integration import LightGBMPruningCallback

R_STATE=123

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
    "random_state" : R_STATE,
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
    "random_state" : R_STATE,
    }

params = {
    # "device_type": trial.suggest_categorical("device_type", ['gpu']),
    "subsample":0.9, "subsample_freq":1,
    "reg_lambda": 5,
    "class_weight": 'balanced',
    "random_state" : R_STATE,"verbose":-1
    }


def iris_dataset_trial():
    rng = np.random.RandomState(R_STATE)
    iris = ld.load_dataset('iris')
    random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.3
    iris.target[random_unlabeled_points] = -1
    svc = SVC(probability=True, gamma="auto", random_state=R_STATE)
    self_training_model = SelfTrainingClassifier(svc, criterion='k_best', k_best=3, max_iter=10, verbose=True)
    self_training_model.fit(iris.data, iris.target)
    print()


def KFoldTrial(dataset_name):
    X, y = ld.load_dataset(dataset_name)
    skf = StratifiedKFold(n_splits=10, random_state=R_STATE, shuffle=True)
    auprc_st_res = []
    auprc_su_res = []
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_st_train = y_train.copy()
        random_unlabeled_points = np.random.RandomState(R_STATE).rand(len(y_st_train)) < 0.9
        y_st_train[random_unlabeled_points] = -1
        y_su_train = y_train[~random_unlabeled_points]
        print(f'Fold {fold}, Number of samples at start:{len(y_su_train)}')
        X_su_train = X_train[~random_unlabeled_points,:]

        #svc = SVC(probability=True, gamma=0.001, random_state=R_STATE, class_weight='balanced')
        svc = RandomForestClassifier(class_weight='balanced', random_state=R_STATE)
        svc.fit(X_su_train, y_su_train)
        probs_su_test = svc.predict_proba(X_test)[:,1]
        auprc_su = average_precision_score(y_test, probs_su_test)
        auprc_su_res.append(auprc_su)

        #base_classifier = SVC(probability=True, gamma=0.001, random_state=R_STATE)
        base_classifier = RandomForestClassifier(class_weight='balanced', random_state=R_STATE)
        #self_training_clf = SelfTrainingClassifier(base_classifier, criterion='k_best', k_best=10, max_iter=10, verbose=False)
        #self_training_clf = StandardSelfTraining(base_classifier, threshold=0.95, k_best=5, max_iter=30, verbose=False)
        #self_training_clf = FreeSelfTraining(base_classifier, threshold=0.90, k_best=10, max_iter=30, balance='equal', verbose=False)
        self_training_clf = FreeDiverseSelfTraining(base_classifier, threshold=0.90, k_best=3, max_iter=30, balance='free',
                                             diverse=5, verbose=False)
        self_training_clf.fit(X_train, y_st_train)
        for clff in self_training_clf.estimator_list:
            probs_ff = clff.predict_proba(X_test)[:, 1]
            auprc_ff = average_precision_score(y_test, probs_ff)
            #print(auprc_ff)

        print(f'Number of samples at the end:{self_training_clf.labeled_sample_size}')
        probs_st_test = self_training_clf.predict_proba(X_test)[:,1]
        auprc_st = average_precision_score(y_test, probs_st_test)
        auprc_st_res.append(auprc_st)

        print(f'Self-training Fold {fold} AUPRC score: {auprc_st}')
        print(f'Supervised Fold {fold} AUPRC score: {auprc_su}\n')
    print(f'Mean AUPRC score of self-training: {np.average(auprc_st_res)}')
    print(f'Mean AUPRC score of supervised: {np.average(auprc_su_res)}')


def selection_bias_by_dirichlet(dataset_name, n_iter=30,
                                    threshold=0.95, k_best=3, max_iter=100, balance='free'):
    auprc_dict = {}
    res= []
    test_percentage = 0.40
    val_percentage = 0.1
    res_loc = f'../results/dirichletbias5_{dataset_name}({test_percentage}_{val_percentage})_DST-RRF_th={threshold}|kb={k_best}|mi={max_iter}|b={balance}.csv'
    if os.path.exists(res_loc):
        res_df = pd.read_csv(res_loc)
    else:
        X, y = ld.load_dataset(dataset_name)
        #threshold, k_best, max_iter, balance, diverse = 0.95, 3, 100, 'free', None
        auprc_st_res = []
        logloss_st_res = []
        auprc_su_res = []
        logloss_su_res = []
        for fold in range(n_iter):
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, test_size=test_percentage,
                                                 random_state=R_STATE+fold)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, stratify=y_train, test_size=val_percentage,
                                                 random_state=R_STATE+fold)
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_val = scaler.transform(X_val)
            #selected_ids = bt.bias_select_by_feature(X_train, feature_id=feature_id)
            selected_ids = bt.bias_by_dirichlet(X_train)
            mask = np.ones_like(y_train, bool)
            mask[selected_ids] = False
            y_st_train = y_train.copy()
            y_st_train[mask] = -1
            y_su_train = y_train[selected_ids]
            X_su_train = X_train[selected_ids,:]

            print(f'Number of samples at the beginning of run {fold}:{len(y_su_train)}')

            #svc = SVC(probability=True, gamma=0.001, random_state=R_STATE, class_weight='balanced')
            svc = lgb.LGBMClassifier(boosting_type="rf", **params)
            svc.fit(X_su_train, y_su_train,
                    eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100,verbose=False)])
            probs_su_test = svc.predict_proba(X_test)
            auprc_su = average_precision_score(y_test, probs_su_test[:,1])
            loss_su = log_loss(y_test, probs_su_test)
            auprc_su_res.append(auprc_su)
            logloss_su_res.append(loss_su)

            #base_classifier = SVC(probability=True, gamma=0.001, random_state=R_STATE)
            base_classifier1 = lgb.LGBMClassifier(boosting_type="rf", **params)
            base_classifier2 = lgb.LGBMClassifier(boosting_type="rf", **params)
            base_classifier3 = lgb.LGBMClassifier(boosting_type="rf", **params)
            base_classifier4 = lgb.LGBMClassifier(boosting_type="rf", **params)
            #self_training_clf = SelfTrainingClassifier(base_classifier, criterion='k_best', k_best=10, max_iter=10, verbose=False)
            #self_training_clf = SelfTrainingClassifier(base_classifier, criterion='threshold',threshold=0.95, max_iter=50,
            #                                           verbose=False)
            #self_training_clf = StandardSelfTraining(base_classifier, threshold=0.95, k_best=5, max_iter=30, verbose=False)
            #self_training_clf = FreeSelfTraining(base_classifier, threshold=0.90, k_best=10, max_iter=30, balance='equal', verbose=False)
            self_training_clf1 = FreeDiverseSelfTraining(base_classifier1, threshold=threshold, k_best=k_best, max_iter=max_iter, balance=balance,
                                                 diverse=3, verbose=False)
            self_training_clf2 = FreeDiverseSelfTraining(base_classifier2, threshold=threshold, k_best=k_best, max_iter=max_iter, balance=balance,
                                                 diverse=5, verbose=False)
            self_training_clf3 = FreeDiverseSelfTraining(base_classifier3, threshold=threshold, k_best=k_best, max_iter=max_iter, balance=balance,
                                                 diverse=10, verbose=False)
            self_training_clf_none = FreeDiverseSelfTraining(base_classifier2, threshold=threshold, k_best=k_best,
                                                        max_iter=max_iter, balance=balance,
                                                        diverse=None, verbose=False)
            self_training_clf1.fit(X_train, y_st_train, X_val, y_val)
            self_training_clf2.fit(X_train, y_st_train, X_val, y_val)
            self_training_clf3.fit(X_train, y_st_train, X_val, y_val)
            print(f'Number of samples at the end for d{3}:{self_training_clf1.labeled_sample_size}')
            self_training_clf_none.fit(X_train, y_st_train, X_val, y_val)
            print(f'Number of samples at the end for not d:{self_training_clf_none.labeled_sample_size}')
            #for clff in self_training_clf.estimator_list:
            #    probs_ff = clff.predict_proba(X_test)
            #    auprc_ff = average_precision_score(y_test, probs_ff[:, 1])
            #    loss_ff = log_loss(y_test, probs_ff)
            #    print(loss_ff)

            #print(f'Number of samples at the end:{self_training_clf.labeled_sample_size}')
            probs_st_test1 = self_training_clf1.predict_proba(X_test)
            auprc_st1 = average_precision_score(y_test, probs_st_test1[:,1])
            loss_st1 = log_loss(y_test, probs_st_test1)
            #auprc_st_res.append(auprc_st)
            #logloss_st_res.append(loss_st)

            probs_st_test2 = self_training_clf2.predict_proba(X_test)
            auprc_st2= average_precision_score(y_test, probs_st_test2[:,1])
            loss_st2 = log_loss(y_test, probs_st_test2)

            probs_st_test3 = self_training_clf3.predict_proba(X_test)
            auprc_st3= average_precision_score(y_test, probs_st_test3[:,1])
            loss_st3 = log_loss(y_test, probs_st_test3)

            probs_st_test_none = self_training_clf_none.predict_proba(X_test)
            auprc_st_none= average_precision_score(y_test, probs_st_test_none[:,1])
            loss_st_none = log_loss(y_test, probs_st_test_none)
            #auprc_st_none_res.append(auprc_st_none)
            #logloss_st_none_res.append(loss_st_none)

            #print(f'Self-training Fold {fold} AUPRC score: {auprc_st}')
            #print(f'Supervised Fold {fold} AUPRC score: {auprc_su}\n')
            #print(f'Self-training Fold {fold} Log loss: {loss_st}')
            #print(f'Supervised Fold {fold} Log loss: {loss_su}\n')
            res.append(['DST-None', len(y_su_train), self_training_clf_none.labeled_sample_size, fold, loss_st_none, auprc_st_none])
            res.append([f'DST-{3}', len(y_su_train), self_training_clf1.labeled_sample_size, fold, loss_st1, auprc_st1])
            res.append([f'DST-{5}', len(y_su_train), self_training_clf2.labeled_sample_size, fold, loss_st2, auprc_st2])
            res.append([f'DST-{10}', len(y_su_train), self_training_clf3.labeled_sample_size, fold, loss_st3, auprc_st3])
            res.append(['SU', len(y_su_train), len(y_su_train), fold, loss_su, auprc_su])
            print(f'Mean AUPRC score of self-training: {np.average(auprc_st_res)}')
        print(f'Mean AUPRC score of supervised: {np.average(auprc_su_res)}')

        #auprc_dict[feature_id] = {'st_auprc':auprc_st_res, 'su_auprc':auprc_su_res,
        #                          'st_logloss':logloss_st_res, 'su_logloss':logloss_su_res}

        res_df = pd.DataFrame(res, columns=['Method', 'Start_size','EndSize', 'Iteration', 'LogLoss', 'AUPRC'])
        res_df.to_csv(res_loc, index=None)
    #qual = res_df.groupby(["Feature", "Method"]).agg({'LogLoss': [np.mean, np.std], 'AUPRC': [np.mean, np.std]})
    qual_loss = res_df.groupby(["Method"])['LogLoss'].agg([np.mean, np.std]).reset_index()
    qual_auprc = res_df.groupby(["Method"])['AUPRC'].agg([np.mean, np.std]).reset_index()

    plt.clf()
    plt.rcParams["figure.figsize"] = (8, 6)
    #plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    fig, ax = plt.subplots()
    #colors = ['green', 'red', 'blue']
    #positions = [0, 1, 2]

    #for group, color, pos in zip(qual_loss.groupby('Method'), colors, positions):
    #    key, group = group
    #    print(group)
    #    group.plot('Feature', 'mean', yerr='std', kind='bar', width=0.2, label=key,
    #               position=pos, color=color, alpha=0.5, ax=ax)
    qual_loss = qual_loss.set_index('Method').loc[['SU', 'DST-None', 'DST-3', 'DST-5', 'DST-10']].reset_index()
    qual_loss.plot('Method', 'mean', yerr='std', kind='bar')

    #ax.set_xlim(-1, 1)
    #ax.set_ylim(0, 0.55)
    title = f'{dataset_name} - DST-RRF(th={threshold}|kb={k_best}|mi={max_iter}|b={balance})'
    plt.title(title)
    png_loc = f'../results/images/dirichletbias5_{dataset_name}({test_percentage}_{val_percentage})_DST-RRF_th={threshold}|kb={k_best}|mi={max_iter}|b={balance}.png'
    plt.savefig(png_loc, type='png', dpi=300, bbox_inches='tight')

    plt.show()
    print(auprc_dict)


def selection_bias_by_joint_feature(dataset_name, n_iter=30,
                                    threshold=0.95, k_best=3, max_iter=100, balance='free'):
    auprc_dict = {}
    res= []
    test_percentage = 0.70
    val_percentage = 0.1
    res_loc = f'../results/jointfeatbias5_{dataset_name}({test_percentage}_{val_percentage})_DST-RRF_th={threshold}|kb={k_best}|mi={max_iter}|b={balance}.csv'
    if os.path.exists(res_loc):
        res_df = pd.read_csv(res_loc)
    else:
        X, y = ld.load_dataset(dataset_name)
        #threshold, k_best, max_iter, balance, diverse = 0.95, 3, 100, 'free', None
        auprc_st_res = []
        logloss_st_res = []
        auprc_su_res = []
        logloss_su_res = []
        for fold in range(n_iter):
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, test_size=test_percentage,
                                                 random_state=R_STATE+fold)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, stratify=y_train, test_size=val_percentage,
                                                 random_state=R_STATE+fold)
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_val = scaler.transform(X_val)
            #selected_ids = bt.bias_select_by_feature(X_train, feature_id=feature_id)
            selected_ids = bt.bias_select_joint_feature(X_train*10)
            mask = np.ones_like(y_train, bool)
            mask[selected_ids] = False
            y_st_train = y_train.copy()
            y_st_train[mask] = -1
            y_su_train = y_train[selected_ids]
            X_su_train = X_train[selected_ids,:]

            print(f'Number of samples at the beginning of run {fold}:{len(y_su_train)}')

            #svc = SVC(probability=True, gamma=0.001, random_state=R_STATE, class_weight='balanced')
            svc = lgb.LGBMClassifier(boosting_type="rf", **params)
            svc.fit(X_su_train, y_su_train,
                    eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100,verbose=False)])
            probs_su_test = svc.predict_proba(X_test)
            auprc_su = average_precision_score(y_test, probs_su_test[:,1])
            loss_su = log_loss(y_test, probs_su_test)
            auprc_su_res.append(auprc_su)
            logloss_su_res.append(loss_su)

            #base_classifier = SVC(probability=True, gamma=0.001, random_state=R_STATE)
            base_classifier1 = lgb.LGBMClassifier(boosting_type="rf", **params)
            base_classifier2 = lgb.LGBMClassifier(boosting_type="rf", **params)
            base_classifier3 = lgb.LGBMClassifier(boosting_type="rf", **params)
            base_classifier4 = lgb.LGBMClassifier(boosting_type="rf", **params)
            #self_training_clf = SelfTrainingClassifier(base_classifier, criterion='k_best', k_best=10, max_iter=10, verbose=False)
            #self_training_clf = SelfTrainingClassifier(base_classifier, criterion='threshold',threshold=0.95, max_iter=50,
            #                                           verbose=False)
            #self_training_clf = StandardSelfTraining(base_classifier, threshold=0.95, k_best=5, max_iter=30, verbose=False)
            #self_training_clf = FreeSelfTraining(base_classifier, threshold=0.90, k_best=10, max_iter=30, balance='equal', verbose=False)
            self_training_clf1 = FreeDiverseSelfTraining(base_classifier1, threshold=threshold, k_best=k_best, max_iter=max_iter, balance=balance,
                                                 diverse=3, verbose=False)
            self_training_clf2 = FreeDiverseSelfTraining(base_classifier2, threshold=threshold, k_best=k_best, max_iter=max_iter, balance=balance,
                                                 diverse=5, verbose=False)
            self_training_clf3 = FreeDiverseSelfTraining(base_classifier3, threshold=threshold, k_best=k_best, max_iter=max_iter, balance=balance,
                                                 diverse=10, verbose=False)
            self_training_clf_none = FreeDiverseSelfTraining(base_classifier2, threshold=threshold, k_best=k_best,
                                                        max_iter=max_iter, balance=balance,
                                                        diverse=None, verbose=False)
            self_training_clf1.fit(X_train, y_st_train, X_val, y_val)
            self_training_clf2.fit(X_train, y_st_train, X_val, y_val)
            self_training_clf3.fit(X_train, y_st_train, X_val, y_val)
            print(f'Number of samples at the end for d{3}:{self_training_clf1.labeled_sample_size}')
            self_training_clf_none.fit(X_train, y_st_train, X_val, y_val)
            print(f'Number of samples at the end for not d:{self_training_clf_none.labeled_sample_size}')
            #for clff in self_training_clf.estimator_list:
            #    probs_ff = clff.predict_proba(X_test)
            #    auprc_ff = average_precision_score(y_test, probs_ff[:, 1])
            #    loss_ff = log_loss(y_test, probs_ff)
            #    print(loss_ff)

            #print(f'Number of samples at the end:{self_training_clf.labeled_sample_size}')
            probs_st_test1 = self_training_clf1.predict_proba(X_test)
            auprc_st1 = average_precision_score(y_test, probs_st_test1[:,1])
            loss_st1 = log_loss(y_test, probs_st_test1)
            #auprc_st_res.append(auprc_st)
            #logloss_st_res.append(loss_st)

            probs_st_test2 = self_training_clf2.predict_proba(X_test)
            auprc_st2= average_precision_score(y_test, probs_st_test2[:,1])
            loss_st2 = log_loss(y_test, probs_st_test2)

            probs_st_test3 = self_training_clf3.predict_proba(X_test)
            auprc_st3= average_precision_score(y_test, probs_st_test3[:,1])
            loss_st3 = log_loss(y_test, probs_st_test3)

            probs_st_test_none = self_training_clf_none.predict_proba(X_test)
            auprc_st_none= average_precision_score(y_test, probs_st_test_none[:,1])
            loss_st_none = log_loss(y_test, probs_st_test_none)
            #auprc_st_none_res.append(auprc_st_none)
            #logloss_st_none_res.append(loss_st_none)

            #print(f'Self-training Fold {fold} AUPRC score: {auprc_st}')
            #print(f'Supervised Fold {fold} AUPRC score: {auprc_su}\n')
            #print(f'Self-training Fold {fold} Log loss: {loss_st}')
            #print(f'Supervised Fold {fold} Log loss: {loss_su}\n')
            res.append(['DST-None', len(y_su_train), self_training_clf_none.labeled_sample_size, fold, loss_st_none, auprc_st_none])
            res.append([f'DST-{3}', len(y_su_train), self_training_clf1.labeled_sample_size, fold, loss_st1, auprc_st1])
            res.append([f'DST-{5}', len(y_su_train), self_training_clf2.labeled_sample_size, fold, loss_st2, auprc_st2])
            res.append([f'DST-{10}', len(y_su_train), self_training_clf3.labeled_sample_size, fold, loss_st3, auprc_st3])
            res.append(['SU', len(y_su_train), len(y_su_train), fold, loss_su, auprc_su])
        print(f'Mean AUPRC score of self-training: {np.average(auprc_st_res)}')
        print(f'Mean AUPRC score of supervised: {np.average(auprc_su_res)}')

        #auprc_dict[feature_id] = {'st_auprc':auprc_st_res, 'su_auprc':auprc_su_res,
        #                          'st_logloss':logloss_st_res, 'su_logloss':logloss_su_res}

        res_df = pd.DataFrame(res, columns=['Method', 'Start_size','EndSize', 'Iteration', 'LogLoss', 'AUPRC'])
        res_df.to_csv(res_loc, index=None)
    #qual = res_df.groupby(["Feature", "Method"]).agg({'LogLoss': [np.mean, np.std], 'AUPRC': [np.mean, np.std]})
    qual_loss = res_df.groupby(["Method"])['LogLoss'].agg([np.mean, np.std]).reset_index()
    qual_auprc = res_df.groupby(["Method"])['AUPRC'].agg([np.mean, np.std]).reset_index()

    plt.clf()
    plt.rcParams["figure.figsize"] = (8, 6)
    #plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    fig, ax = plt.subplots()
    #colors = ['green', 'red', 'blue']
    #positions = [0, 1, 2]

    #for group, color, pos in zip(qual_loss.groupby('Method'), colors, positions):
    #    key, group = group
    #    print(group)
    #    group.plot('Feature', 'mean', yerr='std', kind='bar', width=0.2, label=key,
    #               position=pos, color=color, alpha=0.5, ax=ax)
    qual_loss = qual_loss.set_index('Method').loc[['SU', 'DST-None', 'DST-3', 'DST-5', 'DST-10']].reset_index()
    qual_loss.plot('Method', 'mean', yerr='std', kind='bar')

    #ax.set_xlim(-1, 1)
    #ax.set_ylim(0, 0.55)
    title = f'{dataset_name} - DST-RRF(th={threshold}|kb={k_best}|mi={max_iter}|b={balance})'
    plt.title(title)
    png_loc = f'../results/images/jointfeatbias5_{dataset_name}({test_percentage}_{val_percentage})_DST-RRF_th={threshold}|kb={k_best}|mi={max_iter}|b={balance}.png'
    plt.savefig(png_loc, type='png', dpi=300, bbox_inches='tight')

    plt.show()
    print(auprc_dict)


def selection_bias_by_feature_trial(dataset_name, n_iter=30,
                                    threshold=0.95, k_best=3, max_iter=100, balance='free', diverse=None):

    auprc_dict = {}
    res= []
    test_percentage = 0.70
    val_percentage = 0.1
    res_loc = f'../results/featbias3_{dataset_name}({test_percentage}_{val_percentage})_DST-RRF_th={threshold}|kb={k_best}|mi={max_iter}|b={balance}|d={diverse}.csv'
    if False:#os.path.exists(res_loc):
        res_df = pd.read_csv(res_loc)
    else:
        X, y = ld.load_dataset(dataset_name)
        #threshold, k_best, max_iter, balance, diverse = 0.95, 3, 100, 'free', None
        for feature_id in range(3):#range(X.shape[1]):
            auprc_st_res = []
            logloss_st_res = []
            auprc_su_res = []
            logloss_su_res = []
            for fold in range(n_iter):
                X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, test_size=test_percentage,
                                                     random_state=R_STATE+fold)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, stratify=y_train, test_size=val_percentage,
                                                     random_state=R_STATE+fold)
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                X_val = scaler.transform(X_val)
                selected_ids = bt.bias_select_by_feature(X_train, feature_id=feature_id)
                mask = np.ones_like(y_train, bool)
                mask[selected_ids] = False
                y_st_train = y_train.copy()
                y_st_train[mask] = -1
                y_su_train = y_train[selected_ids]
                X_su_train = X_train[selected_ids,:]

                print(f'Number of samples at the beginning for feature {feature_id} & run {fold}:{len(y_su_train)}')

                #svc = SVC(probability=True, gamma=0.001, random_state=R_STATE, class_weight='balanced')
                svc = lgb.LGBMClassifier(boosting_type="rf", **params)
                svc.fit(X_su_train, y_su_train,
                        eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100,verbose=False)])
                probs_su_test = svc.predict_proba(X_test)
                auprc_su = average_precision_score(y_test, probs_su_test[:,1])
                loss_su = log_loss(y_test, probs_su_test)
                auprc_su_res.append(auprc_su)
                logloss_su_res.append(loss_su)

                #base_classifier = SVC(probability=True, gamma=0.001, random_state=R_STATE)
                base_classifier1 = lgb.LGBMClassifier(boosting_type="rf", **params)
                base_classifier2 = lgb.LGBMClassifier(boosting_type="rf", **params)
                #self_training_clf = SelfTrainingClassifier(base_classifier, criterion='k_best', k_best=10, max_iter=10, verbose=False)
                #self_training_clf = SelfTrainingClassifier(base_classifier, criterion='threshold',threshold=0.95, max_iter=50,
                #                                           verbose=False)
                #self_training_clf = StandardSelfTraining(base_classifier, threshold=0.95, k_best=5, max_iter=30, verbose=False)
                #self_training_clf = FreeSelfTraining(base_classifier, threshold=0.90, k_best=10, max_iter=30, balance='equal', verbose=False)
                self_training_clf = FreeDiverseSelfTraining(base_classifier1, threshold=threshold, k_best=k_best, max_iter=max_iter, balance=balance,
                                                     diverse=diverse, verbose=False)
                self_training_clf_none = FreeDiverseSelfTraining(base_classifier2, threshold=threshold, k_best=k_best,
                                                            max_iter=max_iter, balance=balance,
                                                            diverse=None, verbose=False)
                self_training_clf.fit(X_train, y_st_train, X_val, y_val)
                print(f'Number of samples at the end for d:{self_training_clf.labeled_sample_size}')
                self_training_clf_none.fit(X_train, y_st_train, X_val, y_val)
                print(f'Number of samples at the end for not d:{self_training_clf_none.labeled_sample_size}')
                #for clff in self_training_clf.estimator_list:
                #    probs_ff = clff.predict_proba(X_test)
                #    auprc_ff = average_precision_score(y_test, probs_ff[:, 1])
                #    loss_ff = log_loss(y_test, probs_ff)
                #    print(loss_ff)

                #print(f'Number of samples at the end:{self_training_clf.labeled_sample_size}')
                probs_st_test = self_training_clf.predict_proba(X_test)
                auprc_st = average_precision_score(y_test, probs_st_test[:,1])
                loss_st = log_loss(y_test, probs_st_test)
                auprc_st_res.append(auprc_st)
                logloss_st_res.append(loss_st)

                probs_st_test_none = self_training_clf_none.predict_proba(X_test)
                auprc_st_none= average_precision_score(y_test, probs_st_test_none[:,1])
                loss_st_none = log_loss(y_test, probs_st_test_none)
                #auprc_st_none_res.append(auprc_st_none)
                #logloss_st_none_res.append(loss_st_none)

                print(f'Self-training Fold {fold} AUPRC score: {auprc_st}')
                print(f'Supervised Fold {fold} AUPRC score: {auprc_su}\n')
                print(f'Self-training Fold {fold} Log loss: {loss_st}')
                print(f'Supervised Fold {fold} Log loss: {loss_su}\n')
                res.append(['DST-None', feature_id, fold, loss_st_none, auprc_st_none])
                res.append([f'DST-{diverse}', feature_id, fold, loss_st, auprc_st])
                res.append(['SU', feature_id, fold, loss_su, auprc_su])
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
        group.plot('Feature', 'mean', yerr='std', kind='bar', width=0.2, label=key,
                   position=pos, color=color, alpha=0.5, ax=ax)

    ax.set_xlim(-1, 2.5)
    ax.set_ylim(0, 0.55)
    title = f'{dataset_name} - DST-RRF(th={threshold}|kb={k_best}|mi={max_iter}|b={balance}|d={diverse})'
    plt.title(title)
    png_loc = f'../results/images/featbias3_{dataset_name}({test_percentage}_{val_percentage})_DST-RRF_th={threshold}|kb={k_best}|mi={max_iter}|b={balance}|d={diverse}.png'
    plt.savefig(png_loc, type='png', dpi=300, bbox_inches='tight')

    plt.show()
    print(auprc_dict)


def general_trial(dataset='breast', multi_class=False, labeled_points=400):
    (X_train, y_train), (X_test, y_test) = ld.load_dataset(dataset_name=dataset, )

    y_st_train = y_train.copy().astype(np.int32)
    if labeled_points<1:
        random_unlabeled_points = np.random.RandomState(R_STATE).rand(len(y_st_train)) < 0.9
        y_st_train[random_unlabeled_points] = -1
        y_su_train = y_train[~random_unlabeled_points]
        X_su_train = X_train[~random_unlabeled_points,:]
        print(f'Number of samples at the beginning:{len(y_su_train)}')
    else:
        chosen_labeled_dict = {}
        for i, label in enumerate(np.unique(y_train)):
            y_train_indices = np.arange(len(y_train))
            random.seed(R_STATE + i)
            chosen_labeled = random.sample(list(y_train_indices[y_train==label]), labeled_points)
            chosen_labeled_dict[label] = chosen_labeled
        all_chosen_labels = np.concatenate(tuple(chosen_labeled_dict.values()))
        mask = np.ones_like(y_st_train, bool)
        mask[all_chosen_labels] = False
        y_st_train[mask] = -1
        y_su_train = y_train[all_chosen_labels]
        X_su_train = X_train[all_chosen_labels,:]

    #svc = SVC(probability=True, gamma=0.001, random_state=R_STATE, class_weight='balanced')

    svc = RandomForestClassifier(class_weight='balanced', random_state=R_STATE)
    svc.fit(X_su_train, y_su_train)
    probs_su_test = svc.predict_proba(X_test)
    probs_su_max_test = np.max(probs_su_test, axis=1)
    preds_su_test = svc.predict(X_test)
    if multi_class:
        auroc_su = roc_auc_score(y_test, probs_su_test, multi_class='ovr')
        print(f'Supervised AUROC score: {auroc_su}')
        loss_su = log_loss(y_test, probs_su_test)
        print(f'Supervised Log Loss: {loss_su}')
        cm_su_test = confusion_matrix(y_test, preds_su_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_su_test, display_labels = svc.classes_)
        disp.plot()
        plt.show()
    else:
        auprc_su = average_precision_score(y_test, probs_su_test)

    #base_classifier = SVC(probability=True, gamma=0.001, random_state=R_STATE)

    base_classifier = RandomForestClassifier(class_weight='balanced', random_state=R_STATE)
    #self_training_clf = SelfTrainingClassifier(base_classifier, criterion='k_best', k_best=10, max_iter=10, verbose=False)
    #self_training_clf = StandardSelfTraining(base_classifier, threshold=0.95, k_best=5, max_iter=30, verbose=False)
    #self_training_clf = FreeSelfTraining(base_classifier, threshold=0.90, k_best=10, max_iter=30, balance='equal', verbose=False)
    self_training_clf = FreeDiverseSelfTraining(base_classifier, threshold=0, k_best=10, max_iter=50, balance='free',
                                         diverse=10, verbose=False)
    self_training_clf.fit(X_train, y_st_train)
    for it, clff in enumerate(self_training_clf.estimator_list):
        #probs_ff = clff.predict_proba(X_test)[:, 1]
        #auprc_ff = average_precision_score(y_test, probs_ff)
        probs_ff = clff.predict_proba(X_test)
        auroc_ff = roc_auc_score(y_test, probs_ff, multi_class='ovr')
        print(f'Self-training iteration {it} AUROC score: {auroc_ff}')
        #print(auprc_ff)

    print(f'Number of samples at the end:{self_training_clf.labeled_sample_size}')
    probs_st_test = self_training_clf.predict_proba(X_test)
    #auprc_st = average_precision_score(y_test, probs_st_test)
    probs_st_max_test = np.max(probs_st_test, axis=1)
    preds_st_test = self_training_clf.predict(X_test)
    if multi_class:
        auroc_st = roc_auc_score(y_test, probs_st_test, multi_class='ovr')
        print(f'Self-training AUROC score: {auroc_st}')
        loss_st = log_loss(y_test, probs_st_test)
        print(f'Self-training Log Loss: {loss_st}')
        plt.clf()
        cm_st_test = confusion_matrix(y_test, preds_st_test)
        disp_st = ConfusionMatrixDisplay(confusion_matrix=cm_st_test, display_labels = self_training_clf.base_estimator_.classes_)
        disp_st.plot()
        plt.show()
    else:
        auprc_su = average_precision_score(y_test, probs_su_test)

    #print(f'Self-training AUPRC score: {auprc_st}')
    #print(f'Supervised AUPRC score: {auprc_su}\n')

#KFoldTrial(dataset='breast_cancer')
#general_trial('cifar10', multi_class=True)
'''
selection_bias_by_feature_trial('breast_cancer', n_iter=30,
                                threshold=0.70, k_best=3, max_iter=200, balance='free', diverse=3)
selection_bias_by_feature_trial('breast_cancer', n_iter=30,
                                threshold=0.70, k_best=3, max_iter=200, balance='free', diverse=5)
selection_bias_by_feature_trial('breast_cancer', n_iter=30,
                                threshold=0.70, k_best=3, max_iter=200, balance='free', diverse=10)
selection_bias_by_feature_trial('breast_cancer', n_iter=30,
                                threshold=0.70, k_best=2, max_iter=200, balance='free', diverse=3)
selection_bias_by_feature_trial('breast_cancer', n_iter=30,
                                threshold=0.70, k_best=2, max_iter=200, balance='free', diverse=5)
selection_bias_by_feature_trial('breast_cancer', n_iter=30,
                                threshold=0.70, k_best=2, max_iter=200, balance='free', diverse=10)
'''
#selection_bias_by_joint_feature('breast_cancer', n_iter=30,
#                                threshold=0.70, k_best=3, max_iter=200, balance='free')
selection_bias_by_dirichlet('breast_cancer', n_iter=30,
                                threshold=0.70, k_best=3, max_iter=200, balance='free')
selection_bias_by_dirichlet('breast_cancer', n_iter=30,
                                threshold=0.70, k_best=5, max_iter=200, balance='free')
selection_bias_by_dirichlet('breast_cancer', n_iter=30,
                                threshold=0.70, k_best=10, max_iter=200, balance='free')