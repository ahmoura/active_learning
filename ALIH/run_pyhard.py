from environment.config import *
from environment.which_arff_dataset import which_arff_dataset
from tqdm import tqdm
from strategies.pyHard.pyhard_framework import pyhard_framework
from copy import deepcopy
from environment.results_to_file import result_to_file


def run_pyhard(datasets, n_splits = 5, init_size = 5, cost = 10):

    for ds in datasets:
        for classifier in classifiers:
            X_raw, y_raw, idx_data, dataset_name = which_arff_dataset(ds)
            #para cada i em idx_bag ("n_splits") (1 a 5)
            for idx_bag in range(n_splits):
                for ph_strategy in pyhard_strategies_names:
                    tqdm.write("Testando: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " " + ph_strategy)
                    result = pyhard_framework(deepcopy(X_raw), deepcopy(y_raw), idx_data, idx_bag, classifier, init_size, cost, ph_strategy)
                    result['dataset'] = ds[:-5]
                    result_to_file(result, dataset_name, classifier, ph_strategy, "pyhard", idx_bag)
                    # CRIAR FUNCAO PARA SALVAR NO ARQUIVO PATH('.')
                    tqdm.write("Passou: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " " + ph_strategy)