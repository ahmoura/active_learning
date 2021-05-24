from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

def which_arff_dataset(dataset, n_splits=5):

    data = arff.loadarff('./datasets/' + dataset)
    data = pd.DataFrame(data[0])

    X_raw = data[data.columns[:-1]].to_numpy()
    y_raw = data[data.columns[-1]].to_numpy()

    lex = OneHotEncoder(handle_unknown='ignore', sparse=False)
    lex.fit(X_raw)
    X_raw = lex.transform(X_raw)

    # pq o y sem o enconding não funciona?
    ley = LabelEncoder()
    ley.fit(y_raw)
    y_raw = ley.transform(y_raw)

    # cross validation bags
    data_cv = StratifiedShuffleSplit(n_splits=n_splits, train_size=0.2, random_state=0)  # n_splits
    data_cv.get_n_splits(X_raw, y_raw)

    # extraindo ids do data_cv
    idx_data = []
    for train_index, test_index in data_cv.split(X_raw, y_raw):
        idx_data.append([train_index, test_index])

    return X_raw, y_raw, idx_data, dataset