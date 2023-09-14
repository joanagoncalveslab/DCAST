from sklearn.preprocessing import MinMaxScaler
from src import load_dataset as ld
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_chunked
import numpy as np
import math
from operator import itemgetter
import pandas as pd
from src import config


R_STATE = 123


def recursive_search(data, row_id, step):
    id1, id2 = data[row_id][0], data[row_id][1]
    selected=[]
    if id1 < step and id2 < step:
        selected.append(id1)
        selected.append(id2)
    elif id1 < step:
        new_row = int(id2 - step)
        selected = recursive_search(data, new_row, step)
        selected.append(id1)
    elif id2 < step:
        new_row = int(id1 - step)
        selected = recursive_search(data, new_row, step)
        selected.append(id2)
    else:
        new_row1, new_row2 = int(id1 - step), int(id2 - step)
        selected1 = recursive_search(data, new_row1, step)
        selected2 = recursive_search(data, new_row2, step)
        selected = [*selected1, *selected2]
    return selected


# Our technique to choose which samples to label so that we have biased dataset.
def bias_by_cluster(X, y=None, k=3, n_k=1):
    """
    :param X: ndarray
        2D array containing data with float or int type.
    :param y: ndarray or list or None
        labels of samples if bias will be created separately for each class, else None. Default is None.
    :param k: int
        Number of clusters. Default is 3.
    :param n_k: int
        Number of clusters to select.
    :return: ndarray
        1D array of indices of selected samples.
    """
    X = MinMaxScaler().fit_transform(X)
    selected_indices = []
    if y is not None:
        class_indices = {class_id: np.where(y==class_id)[0] for class_id in np.unique(y)}
        for class_id, indices in class_indices.items():
            labels = KMeans(n_clusters=k, random_state=R_STATE).fit(X[indices,:]).labels_
            unique_labels = np.unique(labels, return_counts=True)
            selected_cluster = unique_labels[0][np.argmax(unique_labels[1])]
            #np.random.seed(R_STATE)
            #selected_cluster = np.random.choice(np.unique(labels), n_k, replace=False)
            selected_indices = np.concatenate((selected_indices, indices[np.isin(labels, selected_cluster)]), axis=None)
    else:
        labels = KMeans(n_clusters=k, random_state=R_STATE).fit(X).labels_
        np.random.seed(R_STATE)
        selected_cluster = np.random.choice(np.unique(labels), n_k, replace=False)
        selected_indices = np.where(np.isin(y, selected_cluster))[0]
    return np.sort(selected_indices)


# Our technique to choose which samples to label so that we have biased dataset.
def bias_by_random(X, y=None, size=25):
    """
    :param X: ndarray
        2D array containing data with float or int type.
    :param y: ndarray or list or None
        labels of samples if bias will be created separately for each class, else None. Default is None.
    :param max_size: int
        Number of samples to choose from each class.
    :param prob: int
        Ratio of biased samples in the max_size.
    :return: ndarray
        Sorted 1D array of indices of selected samples.
    """
    selected_indices = []
    if y is not None:
        class_indices = {class_id: np.where(y == class_id)[0] for class_id in np.unique(y)}
        for class_id, indices in class_indices.items():
            replacing=False
            if len(indices)<size:
                replacing = True
            np.random.seed(R_STATE+class_id)
            selected_ids_class = np.random.choice(len(indices), size, replace=replacing)
            selected_ids = indices[selected_ids_class]
            selected_indices = [*selected_indices, *selected_ids]
    else:
        np.random.seed(R_STATE)
        selected_indices = np.random.choice(X.shape[0], size)
    return np.sort(selected_indices)


# Our technique to choose which samples to label so that we have biased dataset.
def bias_by_hierarchy(X, y=None, max_size=25, prob = None):
    """
    :param X: ndarray
        2D array containing data with float or int type.
    :param y: ndarray or list or None
        labels of samples if bias will be created separately for each class, else None. Default is None.
    :param max_size: int
        Number of samples to choose from each class.
    :param prob: int
        Ratio of biased samples in the max_size.
    :return: ndarray
        Sorted 1D array of indices of selected samples.
    """
    X = MinMaxScaler().fit_transform(X)
    selected_indices = []
    if y is not None:
        class_indices = {class_id: np.where(y == class_id)[0] for class_id in np.unique(y)}
        for class_id, indices in class_indices.items():
            np.random.seed(R_STATE)
            gen = pairwise_distances_chunked(X[indices,:], n_jobs=-1, working_memory=4096)
            Z = np.concatenate(list(gen), axis=0)
            Z_cond = Z[np.triu_indices(Z.shape[0], k=1)]
            np.random.seed(R_STATE)
            linked = linkage(Z_cond, 'ward').astype(int)
            #linked = linkage(X[indices,:], 'ward').astype(int)
            min_selection_size = min(max_size, np.max(linked[:,3]))
            row = np.where(linked[:, 3] >= min_selection_size)[0][0]
            selected_ids_class= recursive_search(linked, row, len(y[indices]))
            if prob is not None:
                px = np.ones_like(indices)*(1-prob)
                px[selected_ids_class] = prob
                px = px / sum(px)
                np.random.seed(R_STATE)
                selected_ids_class = np.random.choice(len(indices), max_size, p=px, replace=False)
            selected_ids = indices[selected_ids_class]
            selected_indices = [*selected_indices, *selected_ids]
    else:
        linked = linkage(X, 'ward')
        min_selection_size = min(max_size, np.max(linked[:,3]))
        row = np.where(linked[:, 3] == min_selection_size)[0][0]
        step = len(y)
        selected_indices = recursive_search(linked, row, step)
    return np.sort(selected_indices)


# Our technique to choose which samples to label so that we have biased dataset.
def bias_by_hierarchyy(X, y=None, max_size=25, prob = None):
    """
    :param X: ndarray
        2D array containing data with float or int type.
    :param y: ndarray or list or None
        labels of samples if bias will be created separately for each class, else None. Default is None.
    :param max_size: int
        Number of samples to choose from each class.
    :param prob: int
        Ratio of biased samples in the max_size.
    :return: ndarray
        Sorted 1D array of indices of selected samples.
    """
    biased_size = int(np.round(float(max_size)*prob))
    unbiased_size = int(max_size-biased_size)
    X = MinMaxScaler().fit_transform(X)
    selected_indices = []
    if y is not None:
        class_indices = {class_id: np.where(y == class_id)[0] for class_id in np.unique(y)}
        for class_id, indices in class_indices.items():
            np.random.seed(R_STATE)
            gen = pairwise_distances_chunked(X[indices,:], n_jobs=-1, working_memory=4096)
            #print(gen)
            Z = np.concatenate(list(gen), axis=0)
            #print(Z.shape)
            Z_cond = Z[np.triu_indices(Z.shape[0], k=1)]
            np.random.seed(R_STATE)
            linked = linkage(Z_cond, 'ward').astype(int)
            #linked = linkage(X[indices,:], 'ward').astype(int)
            min_selection_size = min(max_size, np.max(linked[:,3]))
            row = np.where(linked[:, 3] >= min_selection_size)[0][0]
            selected_ids_class= recursive_search(linked, row, len(y[indices]))
            replacing = False
            if len(selected_ids_class) < biased_size:
                replacing = True
            np.random.seed(R_STATE)
            selected_biased_ids_class = np.random.choice(np.arange(len(indices))[selected_ids_class], biased_size,
                                                         replace=replacing)
            replacing = False
            if len(indices) - len(selected_ids_class) < unbiased_size:
                replacing = True
            if len(selected_ids_class)<len(indices):
                np.random.seed(R_STATE)
                selected_unbiased_ids_class = np.random.choice(np.delete(np.arange(len(indices)), selected_ids_class),
                                                           unbiased_size, replace=replacing)
                selected_ids_class = np.concatenate([selected_biased_ids_class, selected_unbiased_ids_class], axis=None)
            else:
                selected_ids_class = selected_biased_ids_class.copy()
            #selected_ids_class = np.random.choice(len(indices), max_size, p=px, replace=False)
            selected_ids = indices[selected_ids_class]
            selected_indices = [*selected_indices, *selected_ids]
    else:
        linked = linkage(X, 'ward')
        min_selection_size = min(max_size, np.max(linked[:,3]))
        row = np.where(linked[:, 3] == min_selection_size)[0][0]
        step = len(y)
        selected_indices = recursive_search(linked, row, step)
    return np.sort(selected_indices)


def bias_by_entity(X, y=None, chosen_entity_size=1, max_size=25, prob = None, dominant_class = 1):
    """
    :param X: ndarray
        2D array containing data with float or int type.
    :param y: ndarray or list or None
        labels of samples if bias will be created separately for each class, else None. Default is None.
    :param chosen_entity_size: int
        Number of entities will be used to create bias.
    :param max_size: int
        Number of samples to choose from each class.
    :param prob: int
        Ratio of biased samples in the max_size.
    :return: ndarray
        Sorted 1D array of indices of selected samples.
    """
    biased_size = int(np.round(float(max_size)*prob))
    unbiased_size = int(max_size-biased_size)
    entities = X[:,0:2]
    selected_indices = []
    if y is not None:
        class_indices = {class_id: np.where(y == class_id)[0] for class_id in np.unique(y)}
        if dominant_class is not None:
            dom_class_id, dom_indices = dominant_class, class_indices[dominant_class]
            dom_entities = np.ravel(entities[dom_indices])
            unique_dom_entities, unique_dom_counts = np.unique(dom_entities, return_counts=True)
            selected_dom_entity = np.ravel(unique_dom_entities[np.argpartition(unique_dom_counts, -chosen_entity_size)[-chosen_entity_size:]])
        for class_id, indices in class_indices.items():
            if dominant_class is not None:
                selected_entity = selected_dom_entity
            else:
                class_entities = np.ravel(entities[indices])
                unique_c_entities, unique_c_counts = np.unique(class_entities, return_counts=True)
                selected_entity = np.ravel(unique_c_entities[np.argpartition(unique_c_counts, -chosen_entity_size)[-chosen_entity_size:]])
            selected_ids_class = (np.isin(entities[indices][:, 0], selected_entity)) | (np.isin(entities[indices][:, 1], selected_entity))
            #selected_ids_class = (entities[indices][:,0]==selected_entity) | (entities[indices][:,1]==selected_entity)
            if prob is not None:
                #px_biased = np.zeros_like(indices)
                #px_biased[selected_ids_class] = 1
                #px_biased = px_biased / sum(px_biased)
                replacing=False
                if sum(selected_ids_class)<biased_size:
                    replacing = True
                np.random.seed(R_STATE)
                selected_biased_ids_class = np.random.choice(np.arange(len(indices))[selected_ids_class], biased_size, replace=replacing)
                #px_unbiased = np.ones_like(indices)
                #px_unbiased[selected_ids_class] = 0
                #px_unbiased = px_unbiased / sum(px_unbiased)
                replacing=False
                if len(selected_ids_class)-sum(selected_ids_class)<unbiased_size:
                    replacing = True
                np.random.seed(R_STATE)
                selected_unbiased_ids_class = np.random.choice(np.delete(np.arange(len(indices)), selected_ids_class), unbiased_size, replace=replacing)
                #selected_unbiased_ids_class = np.random.choice(len(indices), unbiased_size, p=px_unbiased, replace=False)
                selected_ids_class = np.concatenate([selected_biased_ids_class, selected_unbiased_ids_class], axis=None)
            selected_ids = indices[selected_ids_class]
            selected_indices = [*selected_indices, *selected_ids]
    return np.sort(selected_indices)


def bias_by_entity_elisl(X, y=None, chosen_entity_size=5, max_size=25, prob = None, cancer='BRCA'):
    bias_loc = config.DATA_DIR / 'graphs' / 'STRING' / 'from_ELISL' / f'{cancer}_genes.csv'
    entities_cancer = pd.read_csv(bias_loc, header=None).values[:,0]
    selected_entity = entities_cancer[:chosen_entity_size]

    biased_size = int(np.round(float(max_size) * prob))
    unbiased_size = int(max_size - biased_size)
    entities = X[:, 0:2]
    selected_indices = []
    if y is not None:
        class_indices = {class_id: np.where(y == class_id)[0] for class_id in np.unique(y)}
        for class_id, indices in class_indices.items():
            selected_ids_class = (np.isin(entities[indices][:, 0], selected_entity)) | (
                np.isin(entities[indices][:, 1], selected_entity))
            # selected_ids_class = (entities[indices][:,0]==selected_entity) | (entities[indices][:,1]==selected_entity)
            if prob is not None:
                # px_biased = np.zeros_like(indices)
                # px_biased[selected_ids_class] = 1
                # px_biased = px_biased / sum(px_biased)
                replacing = False
                if sum(selected_ids_class) < biased_size:
                    replacing = True
                np.random.seed(R_STATE)
                selected_biased_ids_class = np.random.choice(np.arange(len(indices))[selected_ids_class], biased_size,
                                                             replace=replacing)
                # px_unbiased = np.ones_like(indices)
                # px_unbiased[selected_ids_class] = 0
                # px_unbiased = px_unbiased / sum(px_unbiased)
                replacing = False
                if len(selected_ids_class) - sum(selected_ids_class) < unbiased_size:
                    replacing = True
                np.random.seed(R_STATE)
                selected_unbiased_ids_class = np.random.choice(np.delete(np.arange(len(indices)), selected_ids_class),
                                                               unbiased_size, replace=replacing)
                # selected_unbiased_ids_class = np.random.choice(len(indices), unbiased_size, p=px_unbiased, replace=False)
                selected_ids_class = np.concatenate([selected_biased_ids_class, selected_unbiased_ids_class], axis=None)
            selected_ids = indices[selected_ids_class]
            selected_indices = [*selected_indices, *selected_ids]
    return np.sort(selected_indices)


#https://proceedings.neurips.cc/paper/2014/file/d67d8ab4f4c10bf22aa353e27879133c-Paper.pdf
def bias_by_dirichlet(X, n=100):
    """
    :param X: ndarray
        2D array containing data with float or int type.
    :param n: int
        Number of samples to select.
    :return: ndarray
        1D array of indices of selected samples.
    """
    X = MinMaxScaler().fit_transform(X)
    np.random.seed(R_STATE)
    px_k = np.random.dirichlet(tuple([1] * X.shape[1])).transpose().reshape((-1, 1))
    px = X @ px_k
    px = px[:, 0] / sum(px[:, 0])
    np.random.seed(R_STATE)
    selected_ids = np.random.choice(X.shape[0], n, p=px)
    return selected_ids

#https://papers.nips.cc/paper/2014/file/d67d8ab4f4c10bf22aa353e27879133c-Paper.pdf
def bias_by_feature(X, feature_id=0, threshold=0.5, equality_on_left=True, left_probs=0.2):
    probs = np.full(X.shape[0], left_probs)
    X = MinMaxScaler().fit_transform(X)
    if equality_on_left:
        probs[X[:, feature_id] > threshold] = 1 - left_probs
    else:
        probs[X[:, feature_id] >= threshold] = 1 - left_probs
    # np.random.choice(X.shape[0], p=probs)
    selected = np.zeros(X.shape[0])
    selected_ids = []
    for row_id in range(X.shape[0]):
        np.random.seed(R_STATE + row_id)
        tmp_sel = np.random.choice([1, 0], p=[probs[row_id], 1 - probs[row_id]])
        selected[row_id] = tmp_sel
        if tmp_sel == 1:
            selected_ids.append(row_id)

    return np.array(selected_ids)

#https://papers.nips.cc/paper/2014/file/d67d8ab4f4c10bf22aa353e27879133c-Paper.pdf
def bias_by_joint(X):
    X = MinMaxScaler().fit_transform(X)*10
    mean = np.mean(X, axis=0)
    # probs = np.full(X.shape[0], left_probs)
    # np.random.choice(X.shape[0], p=probs)
    selected = np.zeros(X.shape[0])
    probs = []
    for row_id in range(X.shape[0]):
        np.random.seed(R_STATE + row_id)
        dist = math.exp((-0.05) * np.linalg.norm(X[row_id] - mean))
        probs.append(dist)
    probs_scaled = MinMaxScaler().fit_transform(np.array(probs).reshape(-1, 1))[:, 0]
    # probs_scaled= probs
    selected_ids = []
    for row_id in range(X.shape[0]):
        np.random.seed(R_STATE + row_id)
        tmp_sel = np.random.choice([1, 0], p=[probs_scaled[row_id], max(1 - probs_scaled[row_id], 0)])
        selected[row_id] = tmp_sel
        if tmp_sel == 1:
            selected_ids.append(row_id)

    return np.array(selected_ids)


def get_bias(bias, *args, **kwargs):
    return globals()[f'bias_by_{bias}'](*args, **kwargs)


if __name__ == '__main__':
    # skf = StratifiedKFold(n_splits=10, random_state=R_STATE, shuffle=True)
    X, y = ld.load_dataset('breast_cancer')
    sels = []
    test_percentage = 0.70
    val_percentage = 0.1
    for fold in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, test_size=test_percentage,
                                                            random_state=R_STATE + fold)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, stratify=y_train,
                                                          test_size=val_percentage,
                                                          random_state=R_STATE + fold)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25,
        #                                     random_state=R_STATE+fold)
        X_train = MinMaxScaler().fit_transform(X_train)

        # selected_ids = bias_select_joint_feature(X_train*10)
        # selected_ids2 = bias_select_joint_feature(X_train)
        selected = bias_by_hierarchy(X_train, y_train, max_size=25)
        sels.append(len(selected))
        print()
    print(np.mean(sels))
