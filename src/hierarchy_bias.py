from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import pairwise_distances_chunked
import numpy as np

# Our technique to choose which samples to label so that we have biased dataset.
def bias_by_hierarchy(X, y=None, max_size=25, prob = None, r_seed=123):
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
        for class_id, indices in class_indices.items(): #For each class separately
            np.random.seed(r_seed)
            # Memory efficient linkage
            gen = pairwise_distances_chunked(X[indices,:], n_jobs=-1, working_memory=4096)
            Z = np.concatenate(list(gen), axis=0)
            Z_cond = Z[np.triu_indices(Z.shape[0], k=1)]
            np.random.seed(r_seed)
            linked = linkage(Z_cond, 'ward').astype(int)
            
            #Find a random cluster with at least max_size
            min_selection_size = min(max_size, np.max(linked[:,3]))
            row = np.where(linked[:, 3] >= min_selection_size)[0][0]
            selected_ids_class= recursive_search(linked, row, len(y[indices]))
            
            #Select biased_samples
            replacing = False
            if len(selected_ids_class) < biased_size:
                replacing = True
            np.random.seed(r_seed)
            selected_biased_ids_class = np.random.choice(np.arange(len(indices))[selected_ids_class], biased_size,
                                                         replace=replacing)
            
            #Select unbiased samples and combine with biased samples
            replacing = False
            if len(indices) - len(selected_ids_class) < unbiased_size:
                replacing = True
            if len(selected_ids_class)<len(indices):
                np.random.seed(r_seed)
                selected_unbiased_ids_class = np.random.choice(np.delete(np.arange(len(indices)), selected_ids_class),
                                                           unbiased_size, replace=replacing)
                selected_ids_class = np.concatenate([selected_biased_ids_class, selected_unbiased_ids_class], axis=None)
            else:
                selected_ids_class = selected_biased_ids_class.copy()
            
            #Found selected sample indices
            selected_ids = indices[selected_ids_class]
            
            #Combine with selected indices from other classes
            selected_indices = [*selected_indices, *selected_ids]
    else: #Not complete...
        linked = linkage(X, 'ward')
        min_selection_size = min(max_size, np.max(linked[:,3]))
        row = np.where(linked[:, 3] == min_selection_size)[0][0]
        step = len(y)
        selected_indices = recursive_search(linked, row, step)
    #Return sorted selected indices
    return np.sort(selected_indices)