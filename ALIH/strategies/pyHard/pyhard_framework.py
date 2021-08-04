import logging
import os
from environment.config import *
from timeit import default_timer as timer
from environment.pyhard_config import pyhard_config
from sklearn.model_selection import train_test_split
import numpy as np
from modAL.models import ActiveLearner
from environment.which_classifier import which_classifier
from modAL.uncertainty import uncertainty_sampling
from environment.compute_f1 import compute_f1
import pandas as pd
from pathlib import Path
from math import ceil

from pyhard.measures import ClassificationMeasures
from modAL.uncertainty import classifier_uncertainty

def pyhard_framework(X_raw, y_raw, idx_data, idx_bag, classifier, init_size, cost, strategy):

    sample_size = 0  # contador de amostras utilizadas pela estratégia
    accuracy_history = []
    f1_history = []
    start = timer()

    strategy = pyhard_config(strategy)

    X_train, X_test, y_train, y_test = train_test_split(X_raw[idx_data[idx_bag][TRAIN]],
                                                        y_raw[idx_data[idx_bag][TRAIN]],
                                                        train_size= ceil(len(np.unique(y_raw)) + init_size),
                                                        stratify=y_raw[idx_data[idx_bag][TRAIN]])

   # print(np.unique(y_raw[idx_data[idx_bag][TRAIN]]))
    #print(np.unique(y_train))
    sample_size = sample_size + len(X_train)

    learner = ActiveLearner(
        estimator=which_classifier(classifier),
        query_strategy=uncertainty_sampling,
        X_training=X_train, y_training=y_train
    )

    accuracy_history.append(learner.score(X_test, y_test))
    f1_history.append(compute_f1(learner, X_test, y_test, "weighted"))

    X_pool = X_raw[idx_data[idx_bag][TRAIN]]    # Resolve o problema de reposição do loop
    y_pool = y_raw[idx_data[idx_bag][TRAIN]]    # Resolve o problema de reposição do loop
    len_x_pool = int(ceil(len(X_pool)*0.03))

    for i in range(cost):
        try:
            print("COST: {x}".format(x=i))
            print("Y UNIQUE: {}".format(len(np.unique(y_raw))))
            print("LEN_X_POOL: {}".format(len_x_pool))
            print("LEN DE X_POOL: {x}".format(x = len(y_pool)))
            print("LEARNER SCORE: {x}".format(x = learner.score(X_test, y_test)))
            X_train, X_pool, y_train, y_pool = train_test_split(X_pool, y_pool, train_size=len_x_pool)

            pred = learner.predict(X_train)
            print("LEARNER SCORE: {x}".format(x = learner.score(X_pool, y_pool)))

            path_to_bag_files = (Path('.') / 'strategies' / 'pyHard' / 'pyhard_files' / strategy['measure'] / str(
                'bag_' + str(idx_bag)))

            print('XTRAIN {} PRED {}'.format(len(X_train),len(pred)))
            X_rawAndY_raw = np.column_stack([X_train, pred])
            print(path_to_bag_files)
            np.savetxt(str(path_to_bag_files / 'data.csv'), X_rawAndY_raw, fmt='%i', delimiter=",")

            os.system('pyhard --no-isa -c ' + str(path_to_bag_files / 'config.yaml'))
        except Exception as e:
            print(e)
        else:
            df = pd.read_csv(path_to_bag_files /'metadata.csv')

            idx = list(df.sort_values(by=strategy['sortby'], ascending=strategy['ascending'])['instances'][:cost]) #pega as `cost` primeiras amostras (talvez precise mudar pra algo relacionado ao init size ou criar alguma funcao nova)
            print("IDX LIST LEN : {} {} :LEN_X_POOL".format(len(list(df.sort_values(by=strategy['sortby'], ascending=strategy['ascending'])['instances'])), len_x_pool))
            print("IDX: {}".format(idx))

            X_train =X_train[idx] # ORACULO RECEBE
            y_train = y_train[idx] # ORACULO ROTULOU

            sample_size = sample_size + init_size
            learner.teach(X_train, y_train)

            accuracy_history.append(learner.score(X_test, y_test))
            f1_history.append(compute_f1(learner, X_test, y_test, "weighted"))

    end = timer()
    time_elapsed = end - start

    return {"accuracy_history": accuracy_history,
            "f1_history": f1_history,
            "package": "Pyhard",
            "id_bag": idx_bag,
            "time_elapsed": time_elapsed,
            "classifier": classifier,
            "sample_size": sample_size / len(X_raw),
            "strategy": strategy['name']}