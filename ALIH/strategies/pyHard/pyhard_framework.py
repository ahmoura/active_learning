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


def pyhard_framework(X_raw, y_raw, idx_data, idx_bag, classifier, init_size, cost, strategy):
    from modAL.uncertainty import classifier_uncertainty

    sample_size = 0  # contador de amostras utilizadas pela estratégia
    accuracy_history = []
    f1_history = []
    start = timer()

    strategy = pyhard_config(strategy)

    # parte randomica inicial da estratégia

    X_train, X_test, y_train, y_test = train_test_split(X_raw[idx_data[idx_bag][TRAIN]],
                                                        y_raw[idx_data[idx_bag][TRAIN]],
                                                        train_size=len(np.unique(y_raw)) + init_size,
                                                        stratify=y_raw[idx_data[idx_bag][TRAIN]])

    sample_size = sample_size + len(X_train)

    learner = ActiveLearner(
        estimator=which_classifier(classifier),  # cls,
        query_strategy=uncertainty_sampling,
        X_training=X_train, y_training=y_train  # AL AJUSTA O CLASSIFIER
    )

    accuracy_history.append(learner.score(X_test, y_test))
    f1_history.append(compute_f1(learner, X_test, y_test, "weighted"))

    total_of_samples = 1

    # X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, train_size=0.03)

    #idx = np.random.choice(range(len(idx_data[idx_bag][TRAIN])), size=init_size, replace=False)
    #X_train, y_train = X_raw[idx_data[idx_bag][TRAIN][idx]], y_raw[idx_data[idx_bag][TRAIN][idx]]

    X_rawAndY_raw = np.column_stack([X_raw[idx_data[idx_bag][TRAIN]], y_raw[idx_data[idx_bag][TRAIN]]])
    np.savetxt(str(Path('.') / 'strategies' / 'pyHard' / 'pyhard_files' / strategy['measure'] / 'data.csv'), X_rawAndY_raw, fmt='%i', delimiter=",")

    os.system('pyhard --no-isa -c' + str(Path('.') / 'strategies' / 'pyHard' / 'pyhard_files' / strategy['measure'] / 'config.yaml'))

    df = pd.read_csv(Path('.') / 'strategies' / 'pyHard' / 'pyhard_files' / strategy['measure'] /'metadata.csv')

    idx = list(df.sort_values(by=strategy['sortby'], ascending=strategy['ascending'])['instances'][:cost])

    X_train = X_raw[idx_data[idx_bag][TRAIN][idx]]
    y_train = y_raw[idx_data[idx_bag][TRAIN][idx]]

    sample_size = cost
    learner.teach(X_train, y_train)

    accuracy_history.append(learner.score(X_test, y_test))
    f1_history.append(compute_f1(learner, X_test, y_test, "weighted"))

    end = timer()
    time_elapsed = end - start

    return {"accuracy_history": accuracy_history,
            "f1_history": f1_history,
            "package": "Pyhard",
            "time_elapsed": time_elapsed,
            "classifier": classifier,
            "sample_size": sample_size / len(X_raw),
            "strategy": strategy['name']}