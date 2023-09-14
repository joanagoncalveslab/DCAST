import numpy as np
import warnings
from sklearn.utils import safe_mask
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import if_delegate_has_method
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import log_loss
from src.lib.sutils import *

def _validate_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if not hasattr(estimator, "predict_proba"):
        msg = "base_estimator ({}) should implement predict_proba!"
        raise ValueError(msg.format(type(estimator).__name__))

class FreeDiverseSelfTraining:
    """Self-training classifier.
        This class allows a given supervised classifier to function as a
        semi-supervised classifier, allowing it to learn from unlabeled data. It
        does this by iteratively predicting pseudo-labels for the unlabeled data
        and adding them to the training set.
        The classifier will continue iterating until either max_iter is reached, or
        no pseudo-labels were added to the training set in the previous iteration.
        Read more in the :ref:`User Guide <self_training>`.
        Parameters
        ----------
        base_estimator : estimator object
            An estimator object implementing `fit` and `predict_proba`.
            Invoking the `fit` method will fit a clone of the passed estimator,
            which will be stored in the `base_estimator_` attribute.
        threshold : float, default=0.75
            The decision threshold for use with `criterion='threshold'`.
            Should be in [0, 1). When using the `'threshold'` criterion, a
            :ref:`well calibrated classifier <calibration>` should be used.
        k_best : int, default=10
            The amount of samples to add in each iteration. Only used when
            `criterion='k_best'`.
        max_iter : int or None, default=10
            Maximum number of iterations allowed. Should be greater than or equal
            to 0. If it is `None`, the classifier will continue to predict labels
            until no new pseudo-labels are added, or all unlabeled samples have
            been labeled.
        balance : {'free', 'equal', 'ratio'}, default='equal'
            The balance criterion used to select how many labels to add to the
            training set for each class. If `'free'`; labels are added to dataset 
            looking the the highest probability prediction. If `'equal'`; equal 
            number of samples are added to each class. If `'ratio'`; k_best will 
            be divided to each class in proportion to ratio for each class. 
        diverse : int or None, default=10
            Maximum number of iterations allowed. Should be greater than or equal
            to 0. If it is `None`, the classifier will continue to predict labels
            until no new pseudo-labels are added, or all unlabeled samples have
            been labeled.
        verbose : bool, default=False
            Enable verbose output.
        Attributes
        ----------
        base_estimator_ : estimator object
            The fitted estimator.
        classes_ : ndarray or list of ndarray of shape (n_classes,)
            Class labels for each output. (Taken from the trained
            `base_estimator_`).
        transduction_ : ndarray of shape (n_samples,)
            The labels used for the final fit of the classifier, including
            pseudo-labels added during fit.
        labeled_iter_ : ndarray of shape (n_samples,)
            The iteration in which each sample was labeled. When a sample has
            iteration 0, the sample was already labeled in the original dataset.
            When a sample has iteration -1, the sample was not labeled in any
            iteration.
        n_features_in_ : int
            Number of features seen during :term:`fit`.
            .. versionadded:: 0.24
        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.
            .. versionadded:: 1.0
        n_iter_ : int
            The number of rounds of self-training, that is the number of times the
            base estimator is fitted on relabeled variants of the training set.
        termination_condition_ : {'max_iter', 'no_change', 'all_labeled'}
            The reason that fitting was stopped.
            - `'max_iter'`: `n_iter_` reached `max_iter`.
            - `'no_change'`: no new labels were predicted.
            - `'all_labeled'`: all unlabeled samples were labeled before `max_iter`
              was reached."""
    def __init__(self, base_estimator, threshold=None, k_best=None, max_iter=10, balance='equal', diverse=None, verbose=False, val_base=False):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.k_best = k_best
        self.max_iter = max_iter
        self.balance = balance
        self.diverse = diverse
        self.verbose = verbose
        self.val_base = val_base
    
    @timeit
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit self-training classifier using `X`, `y` as training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        y : {array-like, sparse matrix} of shape (n_samples,)
            Array representing the labels. Unlabeled samples should have the
            label -1.
        Returns
        -------
        self : object
            Fitted estimator.
        """

        #X, y = self.base_estimator._validate_data(
        #    X, y, accept_sparse=["csr", "csc", "lil", "dok"], force_all_finite=False
        #)

        if self.base_estimator is None:
            raise ValueError("base_estimator cannot be None!")

        self.base_estimator_ = clone(self.base_estimator)
        self.estimator_list = {}
        if X_val is not None and y_val is not None:
            self.val_loss_dict_ = {}
        else:
            self.val_loss_dict_ = None
            self.val_base = False

        if self.max_iter is not None and self.max_iter < 0:
            raise ValueError(f"max_iter must be >= 0 or None, got {self.max_iter}")

        if self.threshold is None and self.k_best is None:
            raise ValueError(f"At least one must hold: \n"
                             f"1) threshold must be in [0,1) or (1,inf), got {self.threshold}\n"
                             f"2) k_best must be >0, got {self.k_best}")

        if self.threshold is not None and not (0 <= self.threshold):
            raise ValueError(f"threshold must be in [0,1), got {self.threshold}")

        if self.k_best is not None and not (self.k_best >= 1):
            raise ValueError(f"k_best must be in [0,1), got {self.k_best}")
        if self.balance not in ['equal', 'ratio', 'free']:
            raise ValueError(f"balance must be one of [equal, ratio, free], got {self.balance}")

        if y.dtype.kind in ["U", "S"]:
            raise ValueError(
                "y has dtype string. If you wish to predict on "
                "string targets, use dtype object, and use -1"
                " as the label for unlabeled samples."
            )

        has_label = y != -1

        if np.all(has_label):
            warnings.warn("y contains no unlabeled samples", UserWarning)

        if self.k_best is not None and (
            self.k_best > X.shape[0] - np.sum(has_label)
        ):
            warnings.warn(
                "k_best is larger than the amount of unlabeled "
                "samples. All unlabeled samples will be labeled in "
                "the first iteration",
                UserWarning,
            )
        self.ratio_dict = None
        if self.balance=='ratio':
            self.ratio_dict = {}
            labeled_size = sum(has_label)
            for class_label in np.unique(y[has_label]):
                self.ratio_dict[class_label] = float(sum(y==class_label))/float(labeled_size)
        
        self.labeled_sample_size = sum(has_label)
        self.transduction_ = np.copy(y)
        self.labeled_iter_ = np.full_like(y, -1)
        self.labeled_iter_[has_label] = 0
        self.n_iter_ = 0
        #y = np.copy(y)  # copy in order not to change original data

        #all_labeled = False
        if self.val_loss_dict_ is not None and self.val_base:
            self.base_estimator_.fit(
                X[safe_mask(X, has_label)], self.transduction_[has_label],
                eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100,verbose=False)])
            self.estimator_list[self.n_iter_] = clone(self.base_estimator_).fit(
                X[safe_mask(X, has_label)], self.transduction_[has_label],
                eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100,verbose=False)])
        else:
            self.base_estimator_.fit(
                X[safe_mask(X, has_label)], self.transduction_[has_label]
            )
            self.estimator_list[self.n_iter_] = clone(self.base_estimator_).fit(
                X[safe_mask(X, has_label)], self.transduction_[has_label])
        # Validate the fitted estimator since `predict_proba` can be
        # delegated to an underlying "final" fitted estimator as
        # generally done in meta-estimator or pipeline.
        _validate_estimator(self.base_estimator_)
        #iteration = 0
        # Iterate until the result is stable or max_iterations is reached
        while not np.all(has_label) and (
                self.max_iter is None or self.n_iter_ < self.max_iter
        ):
            if self.val_loss_dict_ is not None:
                #Check validation score
                val_prob = self.base_estimator_.predict_proba(X_val, n_jobs=-1)
                val_loss = log_loss(y_val, val_prob)
                self.val_loss_dict_[self.n_iter_] = val_loss
                if self.n_iter_ > 5:
                    search_range = range(self.n_iter_-1,max(self.n_iter_-6,0),-1)
                    is_smaller = [val_loss > self.val_loss_dict_[val_idx] for val_idx in search_range]
                    if np.all(is_smaller):
                        self.termination_condition_ = "validation"
                        break

            threshold_ = self.threshold
            if self.threshold is not None and self.threshold > 1:
                thold_per_class = []
                for class_id in self.base_estimator_.classes_:
                    probs_tr_by_class = self.base_estimator_.predict_proba(X[safe_mask(X, has_label)], n_jobs=-1)[:,np.where(self.base_estimator_.classes_==class_id)[0][0]]
                    thold_per_class.append(np.percentile(probs_tr_by_class, self.threshold))
                threshold_ = max(thold_per_class)
            # Predict on the unlabeled samples
            prob = self.base_estimator_.predict_proba(X[safe_mask(X, ~has_label)], n_jobs=-1)
            pred = self.base_estimator_.classes_[np.argmax(prob, axis=1)]
            max_proba = np.max(prob, axis=1)
            selected = np.array([])
            if self.balance == 'free':
                n_to_select_criteria = [sum(max_proba >= 1.0/len(self.base_estimator_.classes_)+0.2/len(self.base_estimator_.classes_))]
                # Select new labeled samples
                if threshold_ is not None:
                    thold_n_to_select = sum(max_proba >= threshold_)
                    n_to_select_criteria.append(thold_n_to_select)
                if self.k_best is not None:
                    if self.diverse is None:
                        kb_n_to_select = self.k_best
                    else:
                        kb_n_to_select = self.k_best*self.diverse
                    n_to_select_criteria.append(kb_n_to_select)
                n_to_select_criteria.append(max_proba.shape[0])
                n_to_select = min(n_to_select_criteria)
                if n_to_select == max_proba.shape[0]:
                    selected = np.ones_like(max_proba, dtype=bool)
                elif n_to_select > self.k_best and self.diverse is not None:
                    selected_tmp = np.argpartition(-max_proba, n_to_select)[:n_to_select]
                    clustering = AgglomerativeClustering(n_clusters=self.k_best, linkage='single').fit(X[safe_mask(X, ~has_label)][selected_tmp])
                    selected = \
                        [
                            selected_tmp[c_id == clustering.labels_]
                            [
                                np.argmax(max_proba[selected_tmp[c_id == clustering.labels_]])
                            ]
                            for c_id in np.unique(clustering.labels_)
                        ]
                else:
                    # NB these are indices, not a mask
                    selected = np.argpartition(-max_proba, n_to_select)[:n_to_select]
            elif self.balance == 'equal':
                k_per_class = int(np.rint(float(self.k_best) / float(len(self.base_estimator_.classes_))))
                min_n_to_select_list = []
                for label in self.base_estimator_.classes_:
                    ids_of_labels = np.arange(len(pred))[pred==label]
                    if self.threshold is not None and self.threshold > 1:
                        probs_of_class = self.base_estimator_.predict_proba(X[safe_mask(X, has_label)], n_jobs=-1)[:,np.where(self.base_estimator_.classes_==label)[0][0]]
                        threshold_ = np.percentile(probs_of_class, self.threshold)
                    if self.k_best is not None:
                        if self.diverse is None:
                            n_to_select_criteria_for_label = [k_per_class, len(ids_of_labels), sum(max_proba[ids_of_labels] >= 1.0/len(self.base_estimator_.classes_)+0.2/len(self.base_estimator_.classes_))]
                        else:
                            n_to_select_criteria_for_label = [k_per_class*self.diverse, len(ids_of_labels), sum(max_proba[ids_of_labels] >= 1.0/len(self.base_estimator_.classes_)+0.2/len(self.base_estimator_.classes_))]
                    else:
                        n_to_select_criteria_for_label = []
                    # Select new labeled samples
                    if threshold_ is not None:
                        thold_n_to_select = sum(max_proba[ids_of_labels] >= threshold_)
                        n_to_select_criteria_for_label.append(thold_n_to_select)
                    min_n_to_select_list.append(min(n_to_select_criteria_for_label))
                n_to_select = min(min_n_to_select_list)
                selected = []
                for label in self.base_estimator_.classes_:
                    ids_of_labels = np.arange(len(pred))[pred == label]

                    if n_to_select == max_proba[ids_of_labels].shape[0]:
                        selected = selected + list(ids_of_labels)
                    elif n_to_select > k_per_class and self.diverse is not None:
                        selected_tmp_ids = np.argpartition(-max_proba[ids_of_labels], n_to_select)[:n_to_select]
                        selected_tmp = ids_of_labels[selected_tmp_ids]
                        clustering = AgglomerativeClustering(n_clusters=k_per_class, linkage='single').fit(X[safe_mask(X, ~has_label)][selected_tmp])
                        selected = selected + \
                            [
                                selected_tmp[c_id == clustering.labels_]
                                [
                                    np.argmax(max_proba[selected_tmp[c_id == clustering.labels_]])
                                ]
                                for c_id in np.unique(clustering.labels_)
                            ]

                    else:
                        selected_label_ids = np.argpartition(-max_proba[ids_of_labels], n_to_select)[:n_to_select]
                        selected = selected + list(ids_of_labels[selected_label_ids])

            elif self.balance == 'ratio':
                #k_per_class = int(np.rint(self.k_best / len(self.base_estimator_.classes_)))
                selected = []
                for label in self.base_estimator_.classes_:
                    ids_of_labels = np.arange(len(pred))[pred == label]
                    if self.threshold is not None and self.threshold > 1:
                        probs_of_class = self.base_estimator_.predict_proba(X[safe_mask(X, has_label)], n_jobs=-1)[:,np.where(self.base_estimator_.classes_==label)[0][0]]
                        threshold_ = np.percentile(probs_of_class, self.threshold)
                    if self.k_best is not None:
                        k_per_class = int(np.rint(float(self.k_best) * self.ratio_dict[label]))
                        if self.diverse is None:
                            n_to_select_criteria_for_label = [k_per_class, len(ids_of_labels), sum(max_proba[ids_of_labels] >= 1.0/len(self.base_estimator_.classes_)+0.2/len(self.base_estimator_.classes_))]
                        else:
                            n_to_select_criteria_for_label = [k_per_class*self.diverse, len(ids_of_labels), sum(max_proba[ids_of_labels] >= 1.0/len(self.base_estimator_.classes_)+0.2/len(self.base_estimator_.classes_))]
                    else:
                        n_to_select_criteria_for_label = []
                        
                    if threshold_ is not None:
                        thold_n_to_select = sum(max_proba[ids_of_labels] >= threshold_)
                        n_to_select_criteria_for_label.append(thold_n_to_select)
                        
                    n_to_select_for_label = min(n_to_select_criteria_for_label)
                    if n_to_select_for_label == max_proba[ids_of_labels].shape[0]:
                        selected = selected + list(ids_of_labels)
                    elif self.k_best is not None and n_to_select_for_label > k_per_class and self.diverse is not None:
                        selected_tmp_ids = np.argpartition(-max_proba[ids_of_labels], n_to_select_for_label)[:n_to_select_for_label]
                        selected_tmp = ids_of_labels[selected_tmp_ids]
                        clustering = AgglomerativeClustering(n_clusters=k_per_class, linkage='single').fit(X[safe_mask(X, ~has_label)][selected_tmp])
                        selected = selected + \
                            [
                                selected_tmp[c_id == clustering.labels_]
                                [
                                    np.argmax(max_proba[selected_tmp[c_id == clustering.labels_]])
                                ]
                                for c_id in np.unique(clustering.labels_)
                            ]

                    else:
                        selected_label_ids = np.argpartition(-max_proba[ids_of_labels], n_to_select_for_label)[:n_to_select_for_label]
                        selected = selected + list(ids_of_labels[selected_label_ids])

            # Map selected indices into original array
            #print(f'{len(selected)} samples selected')

            if len(selected) == 0:
                # no changed labels
                self.termination_condition_ = "no_change"
                break
            selected_full = np.nonzero(~has_label)[0][selected]
            # Add newly labeled confident predictions to the dataset
            self.transduction_[selected_full] = pred[selected]
            has_label[selected_full] = True
            self.labeled_iter_[selected_full] = self.n_iter_

            if self.verbose:
                print(
                    f"End of iteration {self.n_iter_},"
                    f" added {selected_full.shape[0]} new labels."
                )
            self.labeled_sample_size = sum(has_label)
            self.n_iter_ += 1
            if self.val_loss_dict_ is not None and self.val_base:
                self.base_estimator_.fit(
                    X[safe_mask(X, has_label)], self.transduction_[has_label],
                    eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100,verbose=False)])
                self.estimator_list[self.n_iter_] = clone(self.base_estimator_).fit(
                    X[safe_mask(X, has_label)], self.transduction_[has_label],
                    eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100,verbose=False)])
            else:
                self.base_estimator_.fit(
                    X[safe_mask(X, has_label)], self.transduction_[has_label]
                )
                self.estimator_list[self.n_iter_] = clone(self.base_estimator_).fit(
                    X[safe_mask(X, has_label)], self.transduction_[has_label])
            # Validate the fitted estimator since `predict_proba` can be
            # delegated to an underlying "final" fitted estimator as
            # generally done in meta-estimator or pipeline.
            _validate_estimator(self.base_estimator_)

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"
        if np.all(has_label):
            self.termination_condition_ = "all_labeled"
        if self.val_loss_dict_ is not None:
            self.best_iter = min(self.val_loss_dict_, key=self.val_loss_dict_.get)
            self.base_estimator_ = self.estimator_list[self.best_iter]

        #self.base_estimator_.fit(
        #    X[safe_mask(X, has_label)], self.transduction_[has_label]
        #)
        self.classes_ = self.base_estimator_.classes_
        return self


    @if_delegate_has_method(delegate="base_estimator")
    def predict(self, X):
        """Predict the classes of `X`.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Array with predicted labels.
        """
        check_is_fitted(self)
        X = self.base_estimator_._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        return self.base_estimator_.predict(X, n_jobs=-1)

    def predict_proba(self, X):
        """Predict probability for each possible outcome.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
            Array with prediction probabilities.
        """
        check_is_fitted(self)
        X = self.base_estimator_._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        return self.base_estimator_.predict_proba(X, n_jobs=-1)

    @if_delegate_has_method(delegate="base_estimator")
    def decision_function(self, X):
        """Call decision function of the `base_estimator`.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
            Result of the decision function of the `base_estimator`.
        """
        check_is_fitted(self)
        X = self.base_estimator_._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        return self.base_estimator_.decision_function(X)

    @if_delegate_has_method(delegate="base_estimator")
    def predict_log_proba(self, X):
        """Predict log probability for each possible outcome.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
            Array with log prediction probabilities.
        """
        check_is_fitted(self)
        X = self.base_estimator_._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        return self.base_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate="base_estimator")
    def score(self, X, y):
        """Call score on the `base_estimator`.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        y : array-like of shape (n_samples,)
            Array representing the labels.
        Returns
        -------
        score : float
            Result of calling score on the `base_estimator`.
        """
        check_is_fitted(self)
        X = self.base_estimator_._validate_data(
            X,
            accept_sparse=True,
            force_all_finite=False,
            reset=False,
        )
        return self.base_estimator_.score(X, y)