"""
Hyper-parameter optimization (HPO) module. It is implemented on top of `hyperopt`, which is a Bayesian optimization
Python package.
"""

import sys

import numpy as np
from hyperopt import hp, tpe, fmin, space_eval, STATUS_OK
from hyperopt.pyll import scope
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import cross_val_score

_progressbar = True


def set_hyperopt_progressbar(flag: bool):
    global _progressbar
    _progressbar = flag


def classifier_hp_space(name, **kwargs):
    func = getattr(sys.modules[__name__], f"_{name}_hp_space")
    return func(**kwargs)


def _objective(predictor, params, X, y):
    predictor = predictor(**params)
    if is_classifier(predictor):
        score = cross_val_score(predictor, X, y, cv=3, scoring='f1_micro', n_jobs=-1).mean()
    elif is_regressor(predictor):
        score = cross_val_score(predictor, X, y, cv=3, scoring='neg_median_absolute_error', n_jobs=-1).mean()
    else:
        raise TypeError("Predictor must be either a classifier or a regressor.")
    return {'loss': -score, 'status': STATUS_OK}


def find_best_params(name, predictor, fixed_params, X, y, max_evals=100, hpo_timeout=90):
    # TODO: timeout proportional to number of instances

    def objective(params):
        return _objective(predictor, params, X, y)

    space = classifier_hp_space(name)
    space = {**space, **fixed_params}
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals,
                show_progressbar=_progressbar, timeout=hpo_timeout)

    return space_eval(space, best)


def _random_forest_hp_space():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    :return: Random Forest parameter search space
    """
    space = {'n_estimators': hp.uniformint('n_estimators', 2, 200),
             'max_depth': hp.uniformint('max_depth', 1, 100),
             'criterion': hp.choice('criterion', ["gini", "entropy"])
             }
    return space


def _svc_linear_hp_space():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    :return: SVM Linear parameter search space
    """
    space = {
        'C': hp.loguniform('C', np.log(1e-3), np.log(1e3))
    }
    return space


def _svc_rbf_hp_space(n_features=10):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    :param n_features:
    :return: SVM RBF parameter search space
    """
    space = {'kernel': 'rbf',
             'probability': True,
             'C': hp.loguniform('C', np.log(1e-3), np.log(1e3)),
             'gamma': hp.loguniform('gamma', np.log(1. / n_features * 1e-1), np.log(1. / n_features * 1e1))}
    return space


def _gradient_boosting_hp_space():
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    :return: Gradient Boosting parameter search space
    """
    space = {'learning_rate': hp.lognormal('learning_rate', np.log(0.01), np.log(10.0)),
             'n_estimators': scope.int(hp.qloguniform('n_estimators', np.log(10.5), np.log(1000.5), 1)),
             'loss': hp.choice('loss', ['deviance'])
             }
    return space


def _bagging_hp_space():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
    :return: Bagging parameter search space
    """
    space = {'n_estimators': hp.uniformint('n_estimators', 2, 200)}
    return space


def _gaussian_nb_hp_space():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    :return: Gaussian Naive Bayes classifier parameter search space
    """
    space = {'var_smoothing': hp.loguniform('var_smoothing', np.log(1e-9), np.log(1e-8))}
    return space


def _logistic_regression_hp_space():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    :return: Logistic Regression classifier parameter search space
    """
    space = {'C': hp.loguniform('C', np.log(1e-1), np.log(1e1))}
    return space


def _mlp_hp_space():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    :return: Multi-layer Perceptron parameter search space
    """
    space = {'max_iter': 300,
             'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
             'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive'])
             }
    return space


def _dummy_hp_space():
    """
    https://scikit-learn.org/0.16/modules/generated/sklearn.dummy.DummyClassifier.html
    :return: Dummy parameter search space
    """
    return {'random_state': None}
