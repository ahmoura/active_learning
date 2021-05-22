def fetch_datasets(dataset):
    data = arff.loadarff('./datasets/luis/' + dataset)
    metadata = data[1]
    data = pd.DataFrame(data[0])

    instances = len(data)
    classes = len(data.iloc[:, -1].value_counts())
    attributes = len(data.columns) - 1
    nominal_attributes = str(metadata).count("nominal")

    proportion = data.iloc[:, -1].value_counts()
    proportion = proportion.map(lambda x: round(x / instances * 100, 2))

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