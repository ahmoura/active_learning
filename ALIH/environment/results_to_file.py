import pandas as pd
from pathlib import Path

def result_to_file(results, dataset_name, classifier, strategy, lib, idx_bag):
    output = pd.DataFrame.from_dict(results)
    output = output.set_index(['package', 'time_elapsed', 'classifier', 'sample_size', 'strategy', 'dataset']).apply(pd.Series.explode).reset_index()
    filename = lib + dataset_name + classifier + strategy + str(idx_bag) + '.csv'
    output.to_csv(Path('./output') / filename, index=False)