import glob
import json
import logging
import os
import re
import shutil
import warnings
from importlib import import_module
from inspect import signature, Parameter
from pathlib import Path

import pandas as pd
from deprecation import deprecated
from pyispace import train_is
from pyispace.utils import scriptcsv

from .classification import Classifier
from .feature_selection import featfilt
from .measures import ClassificationMeasures
from .utils import get_param_names

metadata_file = 'metadata.csv'
options_file = 'options.json'
ih_file = 'ih.csv'
instances_index = 'instances'
_cache_path = Path('/root/.pyhard/')
_shared_dir = Path('/home/shared/')
_mcr_envvar = "LD_LIBRARY_PATH"
_mcr_envvar_value = "/usr/local/MATLAB/MATLAB_Runtime/v98/runtime/glnxa64:" \
                    "/usr/local/MATLAB/MATLAB_Runtime/v98/bin/glnxa64:" \
                    "/usr/local/MATLAB/MATLAB_Runtime/v98/sys/os/glnxa64:" \
                    "/usr/local/MATLAB/MATLAB_Runtime/v98/extern/bin/glnxa64"


def build_metadata(data, rootdir=' ', problem='classification', output_col=None, save=True, return_ih=False, **kwargs):
    if problem == 'classification':
        measures = ClassificationMeasures(data, labels_col=output_col)
        learners = Classifier(data, labels_col=output_col)
    elif problem == 'regression':
        # measures = ...
        # learners = ...
        raise NotImplementedError("Regression problems not yet supported. Coming soon!")
    else:
        raise ValueError(f"Unknown type of problem: '{problem}'.")

    m_params = {k: kwargs.get(k) for k in get_param_names(measures.calculate_all) if kwargs.get(k) is not None}
    df_measures = measures.calculate_all(**m_params)

    c_params = {k: kwargs.get(k) for k in get_param_names(learners.run_all) if kwargs.get(k) is not None}
    df_algo = learners.run_all(**c_params)

    df_metadata = pd.concat([df_measures, df_algo], axis=1)
    n_inst = len(df_metadata)
    df_metadata.insert(0, instances_index, range(1, n_inst + 1))
    df_metadata.set_index(instances_index, inplace=True)

    path = Path(rootdir)
    if save:
        if path.is_dir():
            df_metadata.to_csv(path / metadata_file)
        else:
            warnings.warn("Invalid directory '{0}'. File will not be saved.".format(path))
    if return_ih:
        ih_values = learners.estimate_ih()
        df_ih = pd.DataFrame({'instance_hardness': ih_values},
                             index=pd.Index(range(1, n_inst + 1), name=instances_index))
        if save:
            df_ih.to_csv(path / ih_file)
        return df_metadata, df_ih
    else:
        return df_metadata


def run_matilda(rootdir, matildadir, metadata=None):
    """
    This is a wrapper function to call Matilda Matlab routine via Python.

    :param rootdir: directory that contains the files 'metadata.csv' and 'options.json'. This is also the location of
        all the software outputs.
    :param matildadir: it points to the directory with matilda Matlab source code.
    :param metadata: pandas dataframe (default None, if file already exists). It is saved as 'metadata.csv' in rootdir.
    """
    rootdir = Path(rootdir)
    matildadir = Path(matildadir)

    if not rootdir.is_dir():
        raise NotADirectoryError("Invalid rootdir '{0}'".format(rootdir))
    elif not matildadir.is_dir():
        raise NotADirectoryError("Invalid matildadir '{0}'".format(matildadir))

    file = rootdir / metadata_file
    if metadata is not None:
        if isinstance(metadata, pd.DataFrame):
            metadata.to_csv(file)
        else:
            raise TypeError("Expected metadata as pandas DataFrame. Received '{0}' instead".format(type(metadata)))
    else:
        if not file.is_file():
            raise FileNotFoundError("File {0} not found in rootdir '{1}'".format(metadata_file, rootdir))

    options = rootdir / options_file
    if not options.is_file():
        warnings.warn("File {0} not found in '{1}'. "
                      "Building default options file with example script...".format(options_file, rootdir))

    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd(f'{matildadir}{os.sep}', nargout=0)
    eng.trainIS(f'{rootdir}{os.sep}', nargout=0)


def run_matilda_module(rootdir):
    matilda = import_module('matilda')
    obj = matilda.initialize()
    try:
        _ = obj.trainIS(rootdir)
    except ValueError:
        pass
    finally:
        obj.terminate()


def update_options(d1: dict, d2: dict):
    d1 = d1.copy()
    for k in d1.keys():
        d1[k] = {**d1[k], **d2.get(k, dict())}
    return d1


def run_isa(rootdir: Path, metadata=None, settings=None, save_output=True, rotation_adjust=True):
    if rotation_adjust is None:
        rotation_adjust = False

    if not rootdir.exists():
        raise NotADirectoryError(f"Invalid directory {repr(repr(rootdir))}.")

    opts_path = rootdir / options_file
    if opts_path.is_file():
        with open(str(opts_path)) as f:
            opts = json.load(f)
    else:
        raise FileNotFoundError(f"File 'options.json' not found in specified path {repr(str(rootdir))}.")

    meta_path = rootdir / metadata_file
    if isinstance(metadata, pd.DataFrame):
        pass
    elif meta_path.is_file():
        metadata = pd.read_csv(meta_path, index_col='instances')
    else:
        raise FileNotFoundError(f"File 'metadata.csv' not found in specified path {repr(str(rootdir))}.")

    opts = update_options(opts, settings)

    logging.getLogger('pyispace').setLevel(logging.DEBUG)
    out = train_is(metadata, opts, rotation_adjust)
    if save_output:
        scriptcsv(out, rootdir)
    return out


@deprecated(deprecated_in='1.7.4',
            removed_in='2.0.0',
            details="only used in module graphene, which was discontinued")
def clear_cache():
    files = glob.glob(str(_cache_path / '*'))
    for f in files:
        try:
            os.remove(str(_cache_path / f))
        except FileNotFoundError:
            pass


@deprecated(deprecated_in='1.7.4',
            removed_in='2.0.0',
            details="only used in module graphene, which was discontinued")
def _set_environment():
    # delete cached files from previous run
    clear_cache()

    # copy 'options.json' from shared folder to cache files directory
    shutil.copy(str(_shared_dir / options_file), str(_cache_path))

    if os.environ.get(_mcr_envvar) is None:
        os.environ[_mcr_envvar] = _mcr_envvar_value


@deprecated(deprecated_in='1.7.4',
            removed_in='2.0.0',
            details="only used in module graphene, which was discontinued")
def run_pipeline(df_data: pd.DataFrame, configurations: dict):
    logger = logging.getLogger(__name__)
    logging.getLogger().setLevel(logging.INFO)

    logger.debug(f"Preparing cache folder '{_cache_path}'")
    _set_environment()

    kwargs = configurations.copy()

    logger.info("Building metadata.")
    df_metadata, df_ih = build_metadata(data=df_data, save=False, return_ih=True, **kwargs)

    if configurations.get('feat_select'):
        logger.info("Feature selection on")

        sig = signature(featfilt)
        param_dict = {param.name: kwargs[param.name] for param in sig.parameters.values()
                      if param.kind == param.POSITIONAL_OR_KEYWORD and param.default != Parameter.empty and
                      param.name in kwargs}
        selected, df_metadata = featfilt(df_metadata, **param_dict)
        df_metadata.to_csv(_cache_path / metadata_file)
        logger.info("Selected features: {0}".format(selected))
    else:
        df_metadata.to_csv(_cache_path / metadata_file)

    df_ih.to_csv(_cache_path / ih_file)

    logger.info("Running matilda.")
    run_matilda_module(str(_cache_path) + os.sep)
    df_data.to_csv(_cache_path / 'data.csv', index=False)

    logger.info("Instance Hardness analysis finished.")


@deprecated(deprecated_in='1.7.4',
            removed_in='2.0.0',
            details="only used in module graphene, which was discontinued")
def load_cached_files():
    df_data = pd.read_csv(_cache_path / 'data.csv')
    df_metadata = pd.read_csv(_cache_path / metadata_file, index_col=instances_index)
    df_ih = pd.read_csv(_cache_path / ih_file, index_col=instances_index)
    df_metadata = df_ih.join(df_metadata, how='right')

    df_is = pd.read_csv(_cache_path / 'coordinates.csv', index_col='Row')
    df_foot_perf = pd.read_csv(_cache_path / 'footprint_performance.csv', index_col='Row')
    df_foot_perf.index.name = 'Algorithm'

    pattern = re.compile('(^footprint)_(.+)_(good|bad|best)', re.IGNORECASE)
    footprint_files = [u.name for u in _cache_path.glob('*.csv')
                       if u.is_file() and bool(pattern.search(u.name))]
    fp_dict = dict()
    for file in footprint_files:
        g = pattern.match(file).groups()
        try:
            fp_dict[(g[1], g[2])] = pd.read_csv(_cache_path / file, usecols=['Row', 'z_1', 'z_2'], index_col='Row')
        except ValueError:
            continue
    df_footprint = pd.concat(fp_dict)
    df_footprint.reset_index(level='Row', drop=True, inplace=True)
    df_footprint.index.names = ['algo', 'type']
    df_footprint.sort_index(inplace=True)

    df_is.index.name = df_metadata.index.name

    return df_data, df_metadata, df_is, df_footprint, df_foot_perf
