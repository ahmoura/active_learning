def config(section, filename='strategies.config'):
    from configparser import ConfigParser

    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read("../" + filename)
    # get section, default to postgresql
    strategy = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            strategy[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    # transformando texto em bool
    strategy['ascending'] = list(map(lambda x: bool(0 if x == "False" else 1), strategy['ascending'].split(',')))
    strategy['sortby'] = strategy['sortby'].split(',')
    
    print(strategy)
    
    return strategy

def compute_f1(learner, X, y_true, average = None):
    y_pred = learner.predict(X)
    return metrics.f1_score(y_true, y_pred, average = average)

def compute_auc(learner, X, y_true, average = None, multi_class = "ovo"):
    y_pred = learner.predict_proba(X)
    return metrics.roc_auc_score(y_true, y_pred, average = average, multi_class = multi_class)

def which_pyhard_measure(measure='LSC'):
    import yaml
    with open(r'config-template.yaml') as file:
        configs_list = yaml.load(file, Loader=yaml.FullLoader)

        if measure == 'LSC':
            configs_list['measures_list'] = ['LSC']
        elif measure == 'Harmfulness':
            configs_list['measures_list'] = ['Harmfulness']
        elif measure == 'Usefulness':
            configs_list['measures_list'] = ['Usefulness']
        elif measure == 'U+H':
            configs_list['measures_list'] = ['Harmfulness','Usefulness']
        elif measure == 'N2':
            configs_list['measures_list'] = ['N2']
        elif measure == 'F3':
            configs_list['measures_list'] = ['F3']

    with open(r'config.yaml', 'w') as file:
        yaml.dump(configs_list, file)
        
def which_dataset(dataset = "iris", n_splits = 5):
    
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
    data_cv = StratifiedShuffleSplit(n_splits= n_splits, train_size=0.7, random_state=0) #n_splits
    
    # extraindo ids do data_cv
    idx_data = []
    for train_index, test_index in data_cv.split(X_raw):
            idx_data.append([train_index, test_index])

    return X_raw, y_raw, idx_data


def which_oml_dataset(dataset_id, n_splits = 5):
    data = openml.datasets.get_dataset(dataset_id)
    
    X_raw, y_raw, categorical_indicator, attribute_names = data.get_data(
    dataset_format="array", target=data.default_target_attribute)
    
    le = preprocessing.LabelEncoder()
    le.fit(y_raw)
    y_raw = le.transform(y_raw)
    
    X_raw = np.nan_to_num(X_raw)
    
    data_cv = StratifiedShuffleSplit(n_splits= n_splits, train_size=0.7, random_state=0) #n_splits
    
    idx_data = []
    for train_index, test_index in data_cv.split(X_raw):
            idx_data.append([train_index, test_index])

    return X_raw, y_raw, idx_data, data.name

def which_arff_dataset(dataset, n_splits = 5):
   
    from sklearn.preprocessing import OneHotEncoder
    
    data = arff.loadarff('datasets/luis/' + dataset)
    data = pd.DataFrame(data[0])

    X_raw = data[data.columns[:-1]].to_numpy()
    y_raw = data[data.columns[-1]].to_numpy()
    
    lex = preprocessing.OrdinalEncoder()
    lex.fit(X_raw)
    X_raw = lex.transform(X_raw)
        
    ley = preprocessing.LabelEncoder()
    ley.fit(y_raw)
    y_raw = ley.transform(y_raw)
    
    # cross validation bags
    data_cv = StratifiedShuffleSplit(n_splits= n_splits, train_size=0.2, random_state=0) #n_splits
    data_cv.get_n_splits(X_raw,y_raw)
    
    # extraindo ids do data_cv
    idx_data = []
    for train_index, test_index in data_cv.split(X_raw, y_raw):
            idx_data.append([train_index, test_index])

    return X_raw, y_raw, idx_data, dataset

def which_classifier(classifier = '5NN'):
    
    if (classifier == '5NN'):
        return KNeighborsClassifier(5)
    elif (classifier == 'C4.5'):
        return tree.DecisionTreeClassifier()
    elif (classifier == 'NB'):
        return GaussianNB()
    elif (classifier == 'SVM'):
        return SVC(probability=True, gamma='auto')
    elif (classifier == 'RF'):
        return RandomForestClassifier()
    
def fetch_datasets(dataset):
    
    data = arff.loadarff('./datasets/luis/' + dataset)
    metadata = data[1]
    data = pd.DataFrame(data[0])
    
    instances = len(data)
    classes = len(data.iloc[:,-1].value_counts())
    attributes = len(data.columns)- 1
    nominal_attributes = str(metadata).count("nominal")
    
    proportion = data.iloc[:,-1].value_counts()
    proportion = proportion.map(lambda x: round(x/instances*100,2))

    majority = max(proportion)
    minority = min(proportion)

    
    return {
        "name": dataset[:-5],
        "instances": instances,
        "classes": classes,
        "attributes": attributes,
        "nominal attributes": nominal_attributes,
        "majority": majority,
        "minority": minority
    }