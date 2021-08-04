from tqdm import tqdm
from environment.config import *
from environment.which_arff_dataset import which_arff_dataset
from tqdm import tqdm
from strategies.pyHard.pyhard_framework import pyhard_framework
from copy import deepcopy
from environment.results_to_file import result_to_file
from threading import Thread, Barrier
from pathlib import Path

barrier = Barrier(6)

def pyhard_thread(ds, X_raw, y_raw, idx_data, dataset_name, classifier, idx_bag, n_splits, ph_strategy, init_size, cost):
    if ('pyhard' + dataset_name + classifier + ph_strategy + str(idx_bag) + '.csv' not in list(
            os.listdir(Path('.') / 'output'))):
            tqdm.write("Testando: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(
            n_splits) + " " + ph_strategy)
            result = pyhard_framework(deepcopy(X_raw), deepcopy(y_raw), idx_data, idx_bag, classifier, init_size, cost,
                                  ph_strategy)
            result['dataset'] = ds[:-5]
            result_to_file(result, dataset_name, classifier, ph_strategy, "pyhard", idx_bag)
            # CRIAR FUNCAO PARA SALVAR NO ARQUIVO PATH('.')
            tqdm.write("Passou: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(
                n_splits) + " " + ph_strategy)
    else:
        print('modal' + dataset_name + classifier + ph_strategy + str(idx_bag) + '.csv already exists!')
    barrier.wait()


def run_pyhard(datasets, n_splits = 5, init_size = 50, cost = 10):

    for ds in tqdm(datasets,  desc ="Dataset"):
        X_raw, y_raw, idx_data, dataset_name = which_arff_dataset(ds, n_splits=n_splits)
        for classifier in classifiers:
            # para cada i em idx_bag ("n_splits") (1 a 5)
            for ph_strategy in pyhard_strategies_names:
                thr_list = []
                for idx_bag in range(n_splits):
                    thr_list.append(Thread(target=pyhard_thread, args=(ds, X_raw, y_raw, idx_data, dataset_name, classifier, idx_bag, n_splits, ph_strategy, init_size, cost,)))
                for thr in thr_list:
                    thr.start()
                barrier.wait()
                for thr in thr_list:
                    thr.join()