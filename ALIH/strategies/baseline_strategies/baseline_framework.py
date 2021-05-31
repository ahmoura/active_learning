from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from environment.config import *  # importing variables
from environment.which_classifier import which_classifier
from environment.compute_f1 import compute_f1


def baseline_framework(X_raw, y_raw, idx_data, idx_bag, classifier):

    accuracy_history = []
    f1_history = []
    start = timer()

    X_train, X_test, y_train, y_test = train_test_split(X_raw[idx_data[idx_bag][TRAIN]],
                                                        y_raw[idx_data[idx_bag][TRAIN]],
                                                        train_size=0.75,
                                                        stratify=y_raw[idx_data[idx_bag][TRAIN]])

    cls = which_classifier(classifier)
    cls.fit(X_train, y_train)

    accuracy_history.append(cls.score(X_test, y_test))
    f1_history.append(compute_f1(cls, X_test, y_test, "weighted"))

    end = timer()
    time_elapsed = end - start

    return {"accuracy_history": accuracy_history,
            "f1_history": f1_history,
            "package": "baseline",
            "id_bag": idx_bag,
            "time_elapsed": time_elapsed,
            "classifier": classifier,
            "sample_size": len(X_train),
            "strategy": "baseline"}