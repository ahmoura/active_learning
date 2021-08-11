import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.feature_selection import VarianceThreshold

from . import get_seed
from .thirdparty import skfeature
from .utils import call_module_func
from .thirdparty import rank_aggregation as ra
from .thirdparty.entropy_estimators import set_random_generator


def _select(F, J, how='cumsum', **kwargs):
    sorted_idx = np.argsort(J)[::-1]
    F_sorted = F[sorted_idx]

    if how == 'cumsum':
        if 'eta' in kwargs:
            eta = kwargs['eta']
        else:
            eta = 0.8
        s_value = softmax(J[sorted_idx])
        p = 0
        selected = []
        for i in range(len(s_value)):
            p += s_value[i]
            selected.append(F_sorted[i])
            if p >= eta:
                break
        return selected

    elif how == 'top':
        if 'N' in kwargs:
            N = kwargs['N']
        else:
            N = len(F) // 2
        return F_sorted[:N]


def _prefilter(X, var_threshold=1e-3):
    sel = VarianceThreshold(threshold=var_threshold)
    sel.fit(X)
    return sel.get_support()


def featfilt(df_metadata: pd.DataFrame, max_n_features=10, method='icap', var_filter=True, var_threshold=0, **kwargs):
    """
    Supervised feature filtering function. It involves three steps:

    1. Removes features whose variance is below ``var_threshold``, if ``var_filter`` is set true
    2. For each algo, it applies an information theoretic based method and select the most relevant features whose
       cumulative sum of score values is greater than or equal ``eta``
    3. Aggregation of the ranks obtained in (2), and selection of the top ``max_n_features``

    Input dataframe (``df_metadata``) should use Matilda standard: *feature_* prefix for measure columns, and *algo_*
    prefix for algorithm performances.

    According to `Matilda documentation <https://github.com/andremun/InstanceSpace>`_, it is recommended *using no
    more than 10 features as input to PILOT's optimal projection algorithm* (default ``max_n_features=10``).

    :param df_metadata: metadata dataframe.
    :type df_metadata: pandas.DataFrame
    :param max_n_features: maximum number of selected features at the end.
    :type max_n_features: int
    :param method: score method (see :py:mod:`pyhard.thirdparty.skfeature` module)
    :type method: str
    :param var_filter: enables variance filter
    :type var_filter: bool
    :param var_threshold: variance filter threshold
    :type var_threshold: float
    :param kwargs: specific for the used method
    :return: list of selected features, ``df_metadata`` with not selected features dropped
    """

    set_random_generator(get_seed())
    df_features = df_metadata.filter(regex='^feature_')
    df_algo = df_metadata.filter(regex='^algo_')
    orig_feat = df_features.columns.to_list()

    kwargs = {**kwargs, **{'n_selected_features': max_n_features}}

    if var_filter:
        mask = _prefilter(df_features.values, var_threshold)
        df_features = df_features.iloc[:, mask]

    agg = ra.RankAggregator()
    rank = []
    feat_list = df_features.columns.to_list()

    for algo in df_algo:
        args = [df_features.values, df_algo[[algo]].values]
        F, J, _ = call_module_func(skfeature, method, *args, **kwargs)
        idx = np.argsort(J)[::-1]
        assert (np.diff(J[idx]) <= 0).all()
        # rank.append(_select(F[idx], J[idx], eta=eta))
        rank.append(F)

    rank = [[feat_list[i] for i in l] for l in rank]
    selected_list = agg.instant_runoff(rank)[:max_n_features]
    blacklist = list(set(orig_feat).difference(set(selected_list)))

    return selected_list, df_metadata.drop(columns=blacklist)
