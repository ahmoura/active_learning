#tqdm_datasets = tqdm(datasets, desc=" Dataset: "+ str(ds[:-5]))
#tqdm_classifier = tqdm(classifiers, desc="Classifier: "+ str(classifier))

for ds in tqdm(datasets,  desc ="Dataset"):
    for classifier in tqdm(classifiers,  desc ="Classifier"):
        X_raw, y_raw, idx_data, dataset_name = which_arff_dataset(ds)

        #para cada i em idx_bag ("n_splits") (1 a 5)
        for idx_bag in tqdm(range(n_splits),  desc ="Bag"):
            tqdm.write("Testando: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " uncertain_sampling")
            result = uncertain_sampling(deepcopy(X_raw), deepcopy(y_raw), idx_data, idx_bag, classifier, k, cost)
            result['dataset'] = ds[:-5]
            total_performance_history.append(result)
            tqdm.write("Passou: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " uncertain_sampling")
            
            tqdm.write("Testando: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " random_sampling")
            result = random_sampling(deepcopy(X_raw), deepcopy(y_raw), idx_data, idx_bag, classifier, k, cost)
            result['dataset'] = ds[:-5]
            total_performance_history.append(result)
            tqdm.write("Passou: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " random_sampling")
            
            tqdm.write("Testando: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " query_by_committee")
            result = query_by_committee(deepcopy(X_raw), deepcopy(y_raw), idx_data, idx_bag, classifier, k, cost)
            result['dataset'] = ds[:-5]
            total_performance_history.append(result)
            tqdm.write("Passou: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " query_by_committee")
           
            tqdm.write("Testando: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " exp_error_reduction")
            result = exp_error_reduction(deepcopy(X_raw), deepcopy(y_raw), idx_data, idx_bag, classifier, k, cost)
            result['dataset'] = ds[:-5]
            total_performance_history.append(result)
            tqdm.write("Passou: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " exp_error_reduction")
            
            tqdm.write("Testando: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " exp_model_change")
            result = exp_model_change(deepcopy(X_raw), deepcopy(y_raw), idx_data, idx_bag, classifier, k, cost)
            result['dataset'] = ds[:-5]
            total_performance_history.append(result)
            tqdm.write("Passou: " + str(ds[:-5]) + " " + str(classifier) + " " + str(idx_bag) + "/" + str(n_splits) + " exp_model_change")
