from sklearn.model_selection import train_test_split
from environment.config import *
import numpy as np
from environment.which_classifier import which_classifier
from environment.compute_f1 import compute_f1

def random_sampling(X_raw, y_raw, idx_data, idx_bag, classifier, init_size, cost, strategy, sample_size,
                    accuracy_history, f1_history, X_train, X_test, y_train, y_test, learner=None, committee=None):
    X_train, X_test, y_train, y_test = train_test_split(X_raw[idx_data[idx_bag][TRAIN]],
                                                        y_raw[idx_data[idx_bag][TRAIN]],
                                                        train_size=len(np.unique(y_raw)) + init_size,
                                                        stratify=y_raw[idx_data[idx_bag][TRAIN]])
    sample_size = sample_size + len(X_train)

    cls = which_classifier(classifier)
    cls.fit(X_train, y_train)

    accuracy_history.append(cls.score(X_test, y_test))
    f1_history.append(compute_f1(cls, X_test, y_test, "weighted"))

    return sample_size, accuracy_history, f1_history