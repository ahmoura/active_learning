import os

TRAIN = 0
TEST = 1

modal_strategies = {"Uncertain Sampling": "uncertain_sampling", "Random Sampling": "random_sampling", "Query by Committee": "query_by_committee", "Expected Error Reduction": "exp_error_reduction", "Expected Model Change": "exp_model_change"}
parameters = "(deepcopy(X_raw), deepcopy(y_raw), idx_data, idx_bag, classifier, init_size, cost)"
classifiers = ['5NN', 'C4.5', 'NB','RF']
datasets = os.listdir('./datasets')
#pyhard_strategies_names = ['H', 'U', 'H_U', 'LSC', 'N2', 'F3']

pyhard_strategies_names = ['H', 'U', 'H_U', 'LSC', 'N2']