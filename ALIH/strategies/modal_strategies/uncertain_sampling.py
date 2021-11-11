import numpy as np
from environment.config import *
from modAL.uncertainty import classifier_uncertainty
from environment.compute_f1 import compute_f1


def uncertain_sampling(X_raw, y_raw, idx_data, idx_bag, classifier, init_size, cost, strategy, sample_size,
                       accuracy_history, f1_history, X_train, X_test, y_train, y_test, learner, committee=None):
    idx = np.random.choice(range(len(idx_data[idx_bag][TRAIN])), size=init_size, replace=False)
    X_train, y_train = X_raw[idx_data[idx_bag][TRAIN][idx]], y_raw[idx_data[idx_bag][TRAIN][idx]]

    if classifier_uncertainty(learner, X_train[0].reshape(1, -1)) > 0.2:
        # print("IF", learner.score(X_test, y_test))
        sample_size = sample_size + len(X_train)
        learner.teach(X_train, y_train)
    accuracy_history.append(learner.score(X_raw[idx_data[idx_bag][TEST]], y_raw[idx_data[idx_bag][TEST]]))
    f1_history.append(compute_f1(learner, X_raw[idx_data[idx_bag][TEST]], y_raw[idx_data[idx_bag][TEST]], "weighted"))

    return sample_size, accuracy_history, f1_history