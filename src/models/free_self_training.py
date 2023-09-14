import numpy as np
import warnings
from sklearn.utils import safe_mask
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import if_delegate_has_method

def _validate_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if not hasattr(estimator, "predict_proba"):
        msg = "base_estimator ({}) should implement predict_proba!"
        raise ValueError(msg.format(type(estimator).__name__))

class FreeSelfTraining:
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
        criterion : {'threshold', 'k_best'}, default='threshold'
            The selection criterion used to select which labels to add to the
            training set. If `'threshold'`, pseudo-labels with prediction
            probabilities above `threshold` are added to the dataset. If `'k_best'`,
            the `k_best` pseudo-labels with highest prediction probabilities are
            added to the dataset. When using the 'threshold' criterion, a
            :ref:`well calibrated classifier <calibration>` should be used.
        k_best : int, default=10
            The amount of samples to add in each iteration. Only used when
            `criterion='k_best'`.
        max_iter : int or None, default=10
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
    def __init__(self, base_estimator, threshold=None, k_best=None, max_iter=10, balance='equal', diverse=None, verbose=False):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.k_best = k_best
        self.max_iter = max_iter
        self.balance = balance
        self.diverse = diverse
        self.verbose = verbose


    def fit(self, X, y):
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

        X, y = self.base_estimator._validate_data(
            X, y, accept_sparse=["csr", "csc", "lil", "dok"], force_all_finite=False
        )

        if self.base_estimator is None:
            raise ValueError("base_estimator cannot be None!")

        self.base_estimator_ = clone(self.base_estimator)
        self.estimator_list = []

        if self.max_iter is not None and self.max_iter < 0:
            raise ValueError(f"max_iter must be >= 0 or None, got {self.max_iter}")

        if self.threshold is None and self.k_best is None:
            raise ValueError(f"At least one must hold: \n"
                             f"1) threshold must be in [0,1), got {self.threshold}\n"
                             f"2) k_best must be >0, got {self.k_best}")

        if self.threshold is not None and not (0 <= self.threshold < 1):
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

        self.transduction_ = np.copy(y)
        self.labeled_iter_ = np.full_like(y, -1)
        self.labeled_iter_[has_label] = 0
        self.n_iter_ = 0
        #y = np.copy(y)  # copy in order not to change original data

        #all_labeled = False
        #iteration = 0
        # Iterate until the result is stable or max_iterations is reached
        while not np.all(has_label) and (
                self.max_iter is None or self.n_iter_ < self.max_iter
        ):
            self.base_estimator_.fit(
                X[safe_mask(X, has_label)], self.transduction_[has_label]
            )
            self.estimator_list.append(clone(self.base_estimator_).fit(
                X[safe_mask(X, has_label)], self.transduction_[has_label]
            ))
            # Validate the fitted estimator since `predict_proba` can be
            # delegated to an underlying "final" fitted estimator as
            # generally done in meta-estimator or pipeline.
            _validate_estimator(self.base_estimator_)

            # Predict on the unlabeled samples
            prob = self.base_estimator_.predict_proba(X[safe_mask(X, ~has_label)])
            pred = self.base_estimator_.classes_[np.argmax(prob, axis=1)]
            max_proba = np.max(prob, axis=1)

            selected = np.array([])
            if self.balance == 'free':
                n_to_select_criteria = []
                # Select new labeled samples
                if self.threshold is not None:
                    thold_n_to_select = sum(max_proba > self.threshold)
                    n_to_select_criteria.append(thold_n_to_select)
                if self.k_best is not None:
                    kb_n_to_select = self.k_best
                    n_to_select_criteria.append(kb_n_to_select)
                n_to_select_criteria.append(max_proba.shape[0])
                n_to_select = min(n_to_select_criteria)
                if n_to_select == max_proba.shape[0]:
                    selected = np.ones_like(max_proba, dtype=bool)
                else:
                    # NB these are indices, not a mask
                    selected = np.argpartition(-max_proba, n_to_select)[:n_to_select]
            elif self.balance == 'equal':
                k_per_class = int(np.floor(self.k_best / len(self.base_estimator_.classes_)))
                min_n_to_select_list = []
                for label in self.base_estimator_.classes_:
                    ids_of_labels = np.arange(len(pred))[pred==label]
                    n_to_select_criteria_for_label = [k_per_class, len(ids_of_labels)]
                    # Select new labeled samples
                    if self.threshold is not None:
                        thold_n_to_select = sum(max_proba[ids_of_labels] > self.threshold)
                        n_to_select_criteria_for_label.append(thold_n_to_select)
                    min_n_to_select_list.append(min(n_to_select_criteria_for_label))
                n_to_select = min(min_n_to_select_list)
                selected = []
                for label in self.base_estimator_.classes_:
                    ids_of_labels = np.arange(len(pred))[pred==label]
                    selected_label_ids = np.argpartition(-max_proba[ids_of_labels], n_to_select)[:n_to_select]
                    selected = selected + list(ids_of_labels[selected_label_ids])
            elif self.balance == 'ratio':
                pass

            # Map selected indices into original array
            selected_full = np.nonzero(~has_label)[0][selected]

            # Add newly labeled confident predictions to the dataset
            self.transduction_[selected_full] = pred[selected]
            has_label[selected_full] = True
            self.labeled_iter_[selected_full] = self.n_iter_

            if selected_full.shape[0] == 0:
                # no changed labels
                self.termination_condition_ = "no_change"
                break

            if self.verbose:
                print(
                    f"End of iteration {self.n_iter_},"
                    f" added {selected_full.shape[0]} new labels."
                )
            self.n_iter_ += 1

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"
        if np.all(has_label):
            self.termination_condition_ = "all_labeled"

        self.base_estimator_.fit(
            X[safe_mask(X, has_label)], self.transduction_[has_label]
        )
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
        return self.base_estimator_.predict(X)

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
        return self.base_estimator_.predict_proba(X)

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