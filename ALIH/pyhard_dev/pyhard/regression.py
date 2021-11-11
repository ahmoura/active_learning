"""
Module that provides methods for assessing performance of regressors in the portfolio.
"""

import logging
import time
import warnings

import hyperopt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ARDRegression, SGDRegressor, PassiveAggressiveRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

from . import metrics
from .base import BaseLearner
from .hpo import find_best_params


_reg_dict = {
    'ada_boost': AdaBoostRegressor,
    'svr_linear': LinearSVR,
    'svr_epsilon': SVR,
    'svr_nu': NuSVR,
    'decision_tree': DecisionTreeRegressor,
    'random_forest': RandomForestRegressor,
    'extra_tree': ExtraTreesRegressor,
    'gradient_boosting': GradientBoostingRegressor,
    'mlp': MLPRegressor,
    'bagging': BaggingRegressor,
    'bayesian_ard': ARDRegression,
    'kernel_ridge': KernelRidge,
    'sgd': SGDRegressor,
    'passive_aggressive': PassiveAggressiveRegressor,
    'dummy_regressor': DummyRegressor
}

_overlap_params = {
    'svr_epsilon': {'kernel': 'linear'},
    'svr_nu': {'kernel': 'rbf'},
    'mlp': {'solver': 'lbfgs'}
}


class Regressor(BaseLearner):
    def __init__(self, data: pd.DataFrame, cv=10, output_col=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        if output_col is None:
            self.output_col = data.columns[-1]
            self.y = data.iloc[:, -1].values
        else:
            self.output_col = output_col
            self.y = data[output_col].values

        self.data = data.reset_index(drop=True)
        self.X = data.drop(columns=self.output_col).values
        self.N = len(data)
        self.cv = cv
        self.predicted_proba = pd.DataFrame()

    @staticmethod
    def score(metric: str, y_true: np.ndarray, y_pred: np.ndarray):
        value = Regressor._call_function(module=metrics, name=metric, y_true=y_true, y_pred=y_pred)
        return value, np.exp(-value)

    @staticmethod
    def update_params(new_params=None):
        if new_params is None:
            new_params = _overlap_params.copy()
        else:
            for algo, param in _overlap_params.items():
                new_params[algo] = {**new_params.get(algo), **_overlap_params[algo]} if \
                    new_params.get(algo) is not None else _overlap_params[algo]

        return new_params

    def run(self, algo, metric='absolute_error', n_folds=10, n_iter=10, hyper_param_optm=False, hpo_evals=100,
            hpo_timeout=90, hpo_name=None, verbose=False, **kwargs):
        """
        Evaluates the performance obtained in each instance. A cross-validation score, with `n_folds` folds, is
        estimated `n_iter` times for each instance, and the mean value is then computed at the end. During training,
        hyper parameter optimization may be performed optionally.

        :param algo: regressor (standard scikit-learn regressor class)
        :param metric: regression performance metric. Either `absolute_error` (default) or `brier`
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
            algo = _reg_dict[algo]
        else:
            raise ValueError("'algo' parameter must be either callable or a valid regressor name string")

        if verbose:
            self.logger.setLevel(logging.DEBUG)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

        score = np.zeros((self.N, n_iter))
        proba = np.zeros((self.N, n_iter))
        start = time.time()

        self.logger.info("Estimating instance performance...")
        if not hyper_param_optm:
            self.logger.debug(f"Training regressor with default parameters {kwargs}")
        for i in range(n_iter):
            k = 0
            mse = 0
            for train_index, test_index in kf.split(self.X, self.y):
                k += 1
                self.logger.info(f"Evaluating testing fold #{k}")

                scaler = StandardScaler()
                X_train = scaler.fit_transform(self.X[train_index, :])
                y_train = self.y[train_index]
                X_test = scaler.transform(self.X[test_index, :])

                if hyper_param_optm:
                    self.logger.info("Optimizing regressor hyper-parameters")
                    best_params = find_best_params(name=hpo_name, predictor=algo, fixed_params=kwargs,
                                                   X=X_train, y=y_train,
                                                   max_evals=hpo_evals, hpo_timeout=hpo_timeout)
                    self.logger.debug(f"Best hyper-parameters found: {best_params}")
                    reg = algo(**best_params)
                else:
                    reg = algo(**kwargs)

                reg = reg.fit(X_train, y_train)

                y_pred = reg.predict(X_test)
                score[test_index, i], proba[test_index, i] = self.score(metric=metric, y_true=self.y[test_index],
                                                                        y_pred=y_pred)

                fold_mse = mean_squared_error(y_true=self.y[test_index], y_pred=y_pred, squared=False)
                mse += fold_mse
                self.logger.info(f"Test fold RMSE: {np.sqrt(fold_mse)}")

            # print_progress_bar(i + 1, n_iter, prefix='Progress', suffix=f'complete (CV {i + 1}/{n_iter})', length=30)
            self.logger.info(f"Iteration {i + 1}/{n_iter} completed.")
            self.logger.info(f"RMSE on test instances (iteration #{i+1}): {round(np.sqrt(mse / k), 4)}")

        end = time.time()
        self.logger.debug(f"Elapsed time: {(end - start):.2f}")

        return score.mean(axis=1), proba.mean(axis=1)

    def run_all(self, metric='absolute_error', n_folds=10, n_iter=10, algo_list=None, parameters=None,
                hyper_param_optm=False, hpo_evals=100, hpo_timeout=90, verbose=False):
        warnings.filterwarnings(action='ignore', module='sklearn', category=ConvergenceWarning)
        warnings.filterwarnings(action='ignore', module='sklearn', category=UserWarning)

        if hyper_param_optm:
            self.logger.info("Hyper parameter optimization enabled")
            logging.getLogger(hyperopt.__name__).setLevel(logging.WARNING)

        if algo_list is None:
            algo_dict = _reg_dict.copy()
        elif isinstance(algo_list, list):
            keys = sorted(list(set(algo_list) & set(_reg_dict.keys())))
            algo_dict = {k: _reg_dict.get(k) for k in keys}
        else:
            raise TypeError("Expected list type for parameter 'algo_list', not '{0}'".format(type(algo_list)))

        parameters = self.update_params(parameters)

        result = {}
        for name, algo in algo_dict.items():
            self.logger.info(f"Assessing performance of regressor {repr(name)}")
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
