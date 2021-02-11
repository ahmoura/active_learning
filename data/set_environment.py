TRAIN = 0
TEST = 1

cost = 10
n_splits = 5 # number of bags cv
total_performance_history = []
k = 5 #parameter to knn classifier

strategies = {"Uncertain Sampling": 0, "Random Sampling": 1, "Query by Committee": 2, "Expected Error Reduction": 3, "Expected Error Reduction": 4}

#strategies = { 0: "Uncertain Sampling", 1: "Random Sampling", 2: "Query by Committee", 3:"Expected Error Reduction", 4: "Expected Error Reduction"}

# list(strategies.keys())[list(strategies.values())[0]]
