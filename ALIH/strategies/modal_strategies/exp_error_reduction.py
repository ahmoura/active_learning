from modAL.expected_error import expected_error_reduction
from environment.config import *
from environment.compute_f1 import compute_f1

def exp_error_reduction(X_raw, y_raw, idx_data, idx_bag, classifier, init_size, cost, strategy, sample_size,
                        accuracy_history, f1_history, X_train, X_pool, y_train, y_pool, learner, committee=None):
    # print("\t Size of X_pool:", len(X_pool))
    exp_error_idx = expected_error_reduction(learner, X_pool, 'binary', n_instances=init_size)

    learner.teach(X_pool[exp_error_idx], y_pool[exp_error_idx])
    sample_size = sample_size + init_size

    # X_pool = np.delete(X_pool, exp_error_idx, axis=0)
    # y_pool = np.delete(y_pool, exp_error_idx)

    accuracy_history.append(learner.score(X_raw[idx_data[idx_bag][TEST]], y_raw[idx_data[idx_bag][TEST]]))
    f1_history.append(compute_f1(learner, X_raw[idx_data[idx_bag][TEST]], y_raw[idx_data[idx_bag][TEST]], "weighted"))

    return sample_size, accuracy_history, f1_history