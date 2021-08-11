import inspect
import traceback
import warnings

import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_yaml_file(path):
    try:
        with open(path, 'r') as file:
            try:
                return yaml.unsafe_load(file)
            except yaml.YAMLError:
                traceback.print_exc()
    except FileNotFoundError:
        traceback.print_exc()


def get_param_names(method):
    assert callable(method)
    sig = inspect.signature(method)
    parameters = [p for p in sig.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]
    return sorted([p.name for p in parameters])


def call_module_func(module, name, *args, **kwargs):
    return getattr(module, name)(*args, **kwargs)


def reduce_dim(X, y, n_dim=2, method='LDA'):
    method = str.upper(method)
    if method == 'LDA':
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=n_dim))
    elif method == 'NCA':
        model = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=n_dim))
    else:
        model = make_pipeline(StandardScaler(), PCA(n_components=n_dim))

    model.fit(X, y)
    X_embedded = model.transform(X)

    return X_embedded


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar

    :param iteration: current iteration
    :type iteration: int
    :param total: total iterations
    :type total: int
    :param prefix: prefix string
    :type prefix: str
    :param suffix: suffix string
    :type suffix: str
    :param decimals: positive number of decimals in percent complete
    :type decimals: int
    :param length: character length of bar
    :type length: int
    :param fill: bar fill character
    :type fill: str
    :param printEnd: end character (e.g. "\r", "\r\n")
    :type printEnd: str
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ' ' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def check_dtype_consistency(s: pd.Series, expected_type: str) -> bool:
    if expected_type == 'classification':
        if is_float_dtype(s):
            return False
    elif expected_type == 'regression':
        if not is_numeric_dtype(s):
            return False
        elif is_integer_dtype(s):
            warnings.warn("")
    return True
