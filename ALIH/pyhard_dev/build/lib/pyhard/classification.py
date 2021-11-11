"""
Module that provides methods for assessing performance of classifiers in the portfolio.
"""

import inspect
import logging
import os
import sys
import time
import warnings

import hyperopt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from . import metrics, get_seed
from .base import BaseLearner
from .hpo import find_best_params
from .metrics import loss_threshold


_clf_dict = {
    'svc_linear': LinearSVC,
    'svc_rbf': SVC,
    'random_forest': RandomForestClassifier,
    'gradient_boosting': GradientBoostingClassifier,
    'mlp': MLPClassifier,
    'bagging': BaggingClassifier,
    'gaussian_nb': GaussianNB,
    'logistic_regression': LogisticRegression,
    'dummy': DummyClassifier
}

_overlap_params = {
    'svc_rbf': {'probability': False, 'kernel': 'rbf'},
    'mlp': {'solver': 'lbfgs'}
}


class Classifier(BaseLearner):
    def __init__(self, data: pd.DataFrame, cv=10, labels_col=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        if labels_col is None:
            self.labels_col = data.columns[-1]
            self.y = data.iloc[:, -1].values
        else:
            self.labels_col = labels_col
            self.y = data[labels_col].values

        self.data = data.reset_index(drop=True)
        self.X = data.drop(columns=self.labels_col).values
        self.categories = list(np.unique(self.y))
        self.N = len(data)
        self.cv = cv

        self.predicted_proba = pd.DataFrame()

    def cprint(self, msg, flag):
        if flag:
            self.logger.info(msg)

    def score(self, metric: str, y_true: np.ndarray, y_pred: np.ndarray, classes_order: np.ndarray = None):
        if classes_order is None:
            n_classes = y_pred.shape[1]
            classes_order = np.array(range(0, n_classes))

        enc = OneHotEncoder(categories=[self.categories])
        y_true = enc.fit_transform(y_true.reshape(-1, 1)).toarray()
        y_true = y_true[:, classes_order.argsort()]
        return self._call_function(module=metrics, name=metric, y_true=y_true, y_pred=y_pred), y_pred[y_true == 1]

    @staticmethod
    def update_params(new_params=None):
        if new_params is None:
            new_params = _overlap_params.copy()
        else:
            new_params['svc_rbf'] = {**new_params.get('svc_rbf'), **_overlap_params['svc_rbf']} if \
                new_params.get('svc_rbf') is not None else _overlap_params['svc_rbf']

            new_params['mlp'] = {**new_params.get('mlp'), **_overlap_params['mlp']} if \
                new_params.get('mlp') is not None else _overlap_params['mlp']

        return new_params

    def run(self, algo, metric='logloss', n_folds=10, n_iter=10,
            hyper_param_optm=False, hpo_evals=100, hpo_timeout=90, hpo_name=None,
            verbose=False, **kwargs):
        """
        Evaluates the performance obtained in each instance. A cross-validation score, with `n_folds` folds, is
        estimated `n_iter` times for each instance, and the mean value is then computed at the end. During training,
        hyper parameter optimization may be performed optionally.

        :param algo: classifier (standard scikit-learn classifier class)
        :param metric: classification performance metric. Either `logloss` (default) or `brier`
        :param n_folds: number of cross-validation folds to evaluate algorithm performance
        :param n_iter: number of times the cross-validation is repeated. Instance metric is the mean over the iterations
        :param hyper_param_optm: enables HPO (default False)
        :param hpo_evals: maximum number of evaluations
        :param hpo_name: see ``algo_list`` in ``config.yaml``
        :param hpo_timeout:
        :param verbose: turn verbose mode on
        :param kwargs: fixed classifier parameters, which won't be optimized
        :return: average score per instance
        """
        if callable(algo):
            pass
        elif isinstance(algo, str):
            algo = _clf_dict[algo]
        else:
            raise ValueError("'clf' parameter must be either callable or a valid classifier name string")

        seed = get_seed()
        if hyper_param_optm:
            os.environ["HYPEROPT_FMIN_SEED"] = repr(seed) if seed is not None else ""
        if 'random_state' in inspect.signature(algo).parameters and seed is not None:
            kwargs = {**kwargs, **{'random_state': seed}}

        if verbose:
            self.logger.setLevel(logging.DEBUG)

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        score = np.zeros((self.N, n_iter))
        proba = np.zeros((self.N, n_iter))
        start = time.time()

        self.logger.info("Estimating instance performance...")
        if not hyper_param_optm:
            self.logger.debug(f"Training classifier with default parameters {kwargs}")
        # for i in range(n_iter):
        #     k = 0
        #     accuracy = 0
        #     for train_index, test_index in kf.split(self.X, self.y):
        #         k += 1
        #         self.logger.info(f"Evaluating testing fold #{k}")

        #         scaler = StandardScaler()
        #         X_train = scaler.fit_transform(self.X[train_index, :])
        #         y_train = self.y[train_index]
        #         X_test = scaler.transform(self.X[test_index, :])

        #         if hyper_param_optm:
        #             self.logger.info("Optimizing classifier hyper-parameters")
        #             best_params = find_best_params(name=hpo_name, predictor=algo, fixed_params=kwargs,
        #                                            X=X_train, y=y_train,
        #                                            max_evals=hpo_evals, hpo_timeout=hpo_timeout)
        #             self.logger.debug(f"Best hyper-parameters found: {best_params}")
        #             clf = algo(**best_params)
        #         else:
        #             clf = algo(**kwargs)

        #         # clf = clf.fit(X_train, y_train)
        #         self.logger.info("Calibrating probabilities")
        #         calibrated_clf = CalibratedClassifierCV(base_estimator=clf, method='sigmoid',
        #                                                 cv=3, ensemble=False, n_jobs=-1)
        #         calibrated_clf.fit(X_train, y_train)

        #         y_pred = calibrated_clf.predict_proba(X_test)
        #         score[test_index, i], proba[test_index, i] = self.score(metric=metric, y_true=self.y[test_index],
        #                                                                 y_pred=y_pred,
        #                                                                 classes_order=calibrated_clf.classes_)

        #         fold_accuracy = calibrated_clf.score(X_test, self.y[test_index])
        #         accuracy += fold_accuracy
        #         self.logger.info(f"Test fold mean accuracy: {fold_accuracy}")

        #     # print_progress_bar(i + 1, n_iter, prefix='Progress', suffix=f'complete (CV {i + 1}/{n_iter})', length=30)
        #     self.logger.info(f"Iteration {i + 1}/{n_iter} completed.")

        #     thres = loss_threshold(n_classes=len(self.categories), metric=metric)
        #     self.logger.info(f"Mean accuracy on test instances (iteration #{i+1}): {round(accuracy / k, 4)}")
        #     self.logger.debug(f"Proximate conversion of metric '{metric}' to accuracy (mean): "
        #                       f"{np.mean(score <= thres)}")

        end = time.time()
        self.logger.debug(f"Elapsed time: {(end - start):.2f}")

        return score.mean(axis=1), proba.mean(axis=1)

    def run_all(self, metric='logloss', n_folds=10, n_iter=10, algo_list=None, parameters=None,
                hyper_param_optm=False, hpo_evals=100, hpo_timeout=90, verbose=False):
        if not sys.warnoptions:
            warnings.filterwarnings(action='ignore', module='sklearn', category=ConvergenceWarning)
            warnings.filterwarnings(action='ignore', module='sklearn', category=UserWarning)
            os.environ["PYTHONWARNINGS"] = 'ignore::UserWarning,ignore:::sklearn'

        if hyper_param_optm:
            self.logger.info("Hyper parameter optimization enabled")
            logging.getLogger(hyperopt.__name__).setLevel(logging.WARNING)

        if algo_list is None:
            algo_dict = _clf_dict.copy()
        elif isinstance(algo_list, list):
            keys = sorted(list(set(algo_list) & set(_clf_dict.keys())))
            algo_dict = {k: _clf_dict.get(k) for k in keys}
        else:
            raise TypeError("Expected list type for parameter 'algo_list', not '{0}'".format(type(algo_list)))

        parameters = self.update_params(parameters)

        result = {}
        for name, algo in algo_dict.items():
            self.logger.info(f"Assessing performance of classifier {repr(name)}")
            algo_params = parameters.get(name)
            if algo_params is None:
                algo_params = dict()

            result[name], self.predicted_proba[name] = self.run(algo=algo, metric=metric, n_folds=n_folds,
                                                                n_iter=n_iter, verbose=verbose,
                                                                hyper_param_optm=hyper_param_optm, hpo_evals=hpo_evals,
                                                                hpo_timeout=hpo_timeout, hpo_name=name,
                                                                **algo_params)

        df_result = pd.DataFrame(result)

        warnings.resetwarnings()
        return df_result.add_prefix('algo_')
