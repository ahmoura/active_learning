from tqdm import tqdm
from environment.which_arff_dataset import which_arff_dataset
from strategies.baseline_strategies.baseline_framework import baseline_framework
from environment.config import *
from environment.results_to_file import result_to_file
from threading import Thread, Barrier
import os
from pathlib import Path

# strategies + 1 (a propria run_modal)
barrier = Barrier(6)


def baseline_thread(ds, X_raw, y_raw, idx_data, dataset_name, classifier, idx_bag, n_splits):
    if ('baseline' + dataset_name + classifier + str(idx_bag) + '.csv' not in list(
            os.listdir(Path('.') / 'output'))):
        tqdm.write(
            "Testando: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag + 1) + "/" + str(n_splits))
        result = baseline_framework(X_raw, y_raw, idx_data, idx_bag, classifier)
        result['dataset'] = ds[:-5]
        result_to_file(result, dataset_name, classifier, "baseline", "baseline", idx_bag)
        tqdm.write(
            "Passou: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag + 1) + "/" + str(n_splits))
    else:
        print('baseline' + dataset_name + classifier + str(idx_bag) + '.csv already exists!')
    barrier.wait()


def run_baseline(datasets, n_splits=5):
    for ds in tqdm(datasets,  desc="Dataset"):
        X_raw, y_raw, idx_data, dataset_name = which_arff_dataset(ds, n_splits)
        for classifier in classifiers:
            thr_list = []
            for idx_bag in range(n_splits):
                thr_list.append(Thread(target=baseline_thread, args=(
                ds, X_raw, y_raw, idx_data, dataset_name, classifier, idx_bag, n_splits,)))
            for thr in thr_list:
                thr.start()
            barrier.wait()
            for thr in thr_list:
                thr.join()

