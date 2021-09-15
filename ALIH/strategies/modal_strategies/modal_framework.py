from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from environment.config import *  # importing variables
import numpy as np
from modAL.models import ActiveLearner, Committee
from environment.which_classifier import which_classifier
from modAL.uncertainty import uncertainty_sampling
from environment.compute_f1 import compute_f1
from modAL.disagreement import vote_entropy_sampling
from strategies.modal_strategies.uncertain_sampling import uncertain_sampling
from strategies.modal_strategies.random_sampling import random_sampling
from strategies.modal_strategies.query_by_committee import query_by_committee
from strategies.modal_strategies.exp_error_reduction import exp_error_reduction
from strategies.modal_strategies.exp_model_change import exp_model_change
from math import floor, ceil


def modal_framework(X_raw, y_raw, idx_data, idx_bag, classifier, init_size, cost, strategy):

    sample_size = 0  # contador de amostras utilizadas pela estratégia
    accuracy_history = []
    f1_history = []
    start = timer()

    print(strategy[1])
    if strategy[1] == "query_by_committee":
        learner_list = []
        for j in range(1, cost + 1):  # Loop para criação do comitê

            X_train, X_pool, y_train, y_pool = train_test_split(X_raw[idx_data[idx_bag][TRAIN]],
                                                                y_raw[idx_data[idx_bag][TRAIN]],
                                                                train_size=floor(len(X_raw[idx_data[idx_bag][TRAIN]]) * 0.10),
                                                                stratify=y_raw[idx_data[idx_bag][TRAIN]],
                                                                random_state=1)
            sample_size = sample_size + len(X_train)

            # initializing learner
            learner = ActiveLearner(
                estimator=which_classifier(classifier),
                X_training=X_train, y_training=y_train
            )
            learner_list.append(learner)

        # assembling the committee
        committee = Committee(
            learner_list=learner_list,
            query_strategy=vote_entropy_sampling)

    else:
        X_train, X_pool, y_train, y_pool = train_test_split(X_raw[idx_data[idx_bag][TRAIN]],
                                                            y_raw[idx_data[idx_bag][TRAIN]],
                                                            train_size=floor(len(X_raw[idx_data[idx_bag][TRAIN]]) * 0.10),
                                                            stratify=y_raw[idx_data[idx_bag][TRAIN]],
                                                            random_state=1)

        sample_size = sample_size + len(X_train)
        committee = None

        if strategy[1] != "random_sampling":
            learner = ActiveLearner(
                estimator=which_classifier(classifier),  # cls,
                query_strategy=uncertainty_sampling,
                X_training=X_train, y_training=y_train  # AL AJUSTA O CLASSIFIER
            )
        else:
            learner = None

    if (strategy[1] != "random_sampling"):
        accuracy_history.append(learner.score(X_raw[idx_data[idx_bag][TEST]], y_raw[idx_data[idx_bag][TEST]]))
        f1_history.append(compute_f1(learner, X_pool, y_pool, "weighted"))

    for i in range(cost):
        init_size = int(ceil(len(X_raw) * 0.05))
        sample_size, accuracy_history, f1_history = eval(strategy[
                                                             1] + "(X_raw, y_raw, idx_data, idx_bag, classifier, init_size, cost, strategy, sample_size, accuracy_history, f1_history, X_train, X_pool, y_train, y_pool, learner, committee)")

    end = timer()
    time_elapsed = end - start

    return {"accuracy_history": accuracy_history,
            "f1_history": f1_history,
            "package": "modAL",
            "id_bag": idx_bag,
            "time_elapsed": time_elapsed,
            "classifier": classifier,
            "sample_size": sample_size / len(X_raw),
            "strategy": strategy[0]}