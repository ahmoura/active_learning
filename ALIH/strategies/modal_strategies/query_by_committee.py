import numpy as np
from environment.compute_f1 import compute_f1

def query_by_committee(X_raw, y_raw, idx_data, idx_bag, classifier, init_size, cost, strategy, sample_size,
                       accuracy_history, f1_history, X_train, X_pool, y_train, y_pool, learner, committee):
    # print("\t Size of X_pool:", len(X_pool))
    query_idx, query_instance = committee.query(X_pool, n_instances=init_size)
    sample_size = sample_size + len(query_idx)

    committee.teach(
        X=X_pool[query_idx],
        y=y_pool[query_idx]
    )

    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

    accuracy_history.append(committee.score(X_pool, y_pool))
    f1_history.append(compute_f1(committee, X_pool, y_pool, "weighted"))

    return sample_size, accuracy_history, f1_history