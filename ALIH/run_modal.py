from tqdm import tqdm
from environment.which_arff_dataset import which_arff_dataset
from strategies.modal_strategies.modal_framework import modal_framework
from environment.config import *
from environment.results_to_file import result_to_file

def run_modal(datasets, n_splits = 5, init_size = 5, cost = 10):

    for ds in tqdm(datasets,  desc ="Dataset"):
        for classifier in classifiers:
            X_raw, y_raw, idx_data, dataset_name = which_arff_dataset(ds, n_splits) # Seta aqui mesmo ou no loop do dataset?
            for idx_bag in range(n_splits):
                for key, value in modal_strategies.items():
                    tqdm.write("Testando: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag+1) + "/" + str(n_splits) + " " + key)
                    result = modal_framework(X_raw, y_raw, idx_data, idx_bag, classifier, init_size, cost, [key, value])
                    result['dataset'] = ds[:-5]
                    result_to_file(result, dataset_name, classifier, value, "modal", idx_bag)
                    tqdm.write("Passou: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag+1) + "/" + str(n_splits) + " " + key)