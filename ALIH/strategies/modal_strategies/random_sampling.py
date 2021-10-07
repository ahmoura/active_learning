from sklearn.model_selection import train_test_split
from environment.config import *
import numpy as np
from environment.which_classifier import which_classifier
from environment.compute_f1 import compute_f1

def random_sampling(X_raw, y_raw, idx_data, idx_bag, classifier, init_size, cost, strategy, sample_size,
                    accuracy_history, f1_history, X_train, X_test, y_train, y_test, learner=None, committee=None):

    idx = np.random.choice(range(len(idx_data[idx_bag][TRAIN])), size=init_size, replace=False)
    X_train, y_train = X_raw[idx_data[idx_bag][TRAIN][idx]], y_raw[idx_data[idx_bag][TRAIN][idx]]

    # X_train, X_test, y_train, y_test = train_test_split(X_raw[idx_data[idx_bag][TRAIN]],
    #                                                     y_raw[idx_data[idx_bag][TRAIN]],
    #                                                     train_size=len(np.unique(y_raw)) + init_size,
    #                                                     stratify=y_raw[idx_data[idx_bag][TRAIN]])
    sample_size = sample_size + len(idx)

    X_train, y_train = X_train[:cost], y_train[:cost]

    learner.teach(X_train, y_train)

    accuracy_history.append(learner.score(X_raw[idx_data[idx_bag][TEST]], y_raw[idx_data[idx_bag][TEST]]))
    f1_history.append(compute_f1(learner, X_raw[idx_data[idx_bag][TEST]], y_raw[idx_data[idx_bag][TEST]], "weighted"))

    return sample_size, accuracy_history, f1_history