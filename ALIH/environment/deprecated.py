def compute_auc(learner, X, y_true, average=None, multi_class="ovo"):
    y_pred = learner.predict_proba(X)
    return metrics.roc_auc_score(y_true, y_pred, average=average, multi_class=multi_class)

def which_dataset(dataset="iris", n_splits=5):
    # Futuramente essa etapa será ajustada para receber qualquer dataset (ou lista com datasets)
    if (dataset == "iris"):
        data = load_iris()
        X_raw = data['data']
        y_raw = data['target']

    if (dataset == "wine"):
        data = load_wine()
        X_raw = data['data']
        y_raw = data['target']

    if (dataset == "digits"):
        data = load_digits()
        X_raw = data['data']
        y_raw = data['target']

    # cross validation bags
    data_cv = StratifiedShuffleSplit(n_splits=n_splits, train_size=0.7, random_state=0)  # n_splits

    # extraindo ids do data_cv
    idx_data = []
    for train_index, test_index in data_cv.split(X_raw):
        idx_data.append([train_index, test_index])

    return X_raw, y_raw, idx_data


def which_oml_dataset(dataset_id, n_splits=5):
    data = openml.datasets.get_dataset(dataset_id)

    X_raw, y_raw, categorical_indicator, attribute_names = data.get_data(
        dataset_format="array", target=data.default_target_attribute)

    le = preprocessing.LabelEncoder()
    le.fit(y_raw)
    y_raw = le.transform(y_raw)

    X_raw = np.nan_to_num(X_raw)

    data_cv = StratifiedShuffleSplit(n_splits=n_splits, train_size=0.7, random_state=0)  # n_splits

    idx_data = []
    for train_index, test_index in data_cv.split(X_raw):
        idx_data.append([train_index, test_index])

    return X_raw, y_raw, idx_data, data.name