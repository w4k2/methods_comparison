import os
import numpy as np
import pandas as pd
from utils.wilcoxon_ranking import pairs_metrics_multi_line
from utils.datasets_table_description import make_description_table, result_tables_IR

# method_names = ["MOLO-MDH", "SEMOOS", "DE-Forest", "MOOforest"]
# method_names = ["SEMOOS", "DE-Forest", "MOOforest", "MOLO-MDH"]
# method_names = ["DE-Forest", "SEMOOS", "MOOforest", "MOLO-MDH"]
method_names = ["MOOforest", "SEMOOS", "DE-Forest", "MOLO-MDH"]

metrics_alias = ["BAC", "Gmean", "Recall", "Specificity", "Precision"]

# Find common datasets
dataset_names_dict = {}
dataset_paths = []
for method_name in method_names:
    dataset_names_dict[method_name] = []
# Load datasets' names
for method_name in method_names:
    DATASETS_DIR = "results/" + method_name + "/Recall/"
    for root, _, files in os.walk(DATASETS_DIR):
        dataset_name = root.split("/")[-1]
        if len(dataset_name) > 0:
            dataset_names_dict[method_name].append(dataset_name)
# common datasets
dataset_names = list((set(dataset_names_dict[method_names[0]]).intersection(dataset_names_dict[method_names[1]], dataset_names_dict[method_names[2]], dataset_names_dict[method_names[3]])))

# datasets' paths
for dataset_name in dataset_names:
    dataset_paths.append(os.path.join("datasets/", dataset_name + ".dat"))

n_folds = 10
n_methods = len(method_names)
n_metrics = len(metrics_alias)
n_datasets = len(dataset_names)
print(n_datasets)
# Load data from file
data_np = np.zeros((n_datasets, n_metrics, n_methods, n_folds))
mean_scores = np.zeros((n_datasets, n_metrics, n_methods))
stds = np.zeros((n_datasets, n_metrics, n_methods))
for dataset_id, dataset_name in enumerate(dataset_names):
    for clf_id, clf_name in enumerate(method_names):
        for metric_id, metric in enumerate(metrics_alias):
            try:
                filename = "results/" + clf_name + "/" + metric + "/" + dataset_name + "/" + clf_name + ".csv"
                if not os.path.isfile(filename):
                    print("File not exist - %s" % filename)
                    # continue
                scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                data_np[dataset_id, metric_id, clf_id] = scores
                mean_score = np.mean(scores)
                mean_scores[dataset_id, metric_id, clf_id] = mean_score
                std = np.std(scores)
                stds[dataset_id, metric_id, clf_id] = std
            except:
                print("Error loading data!", dataset_name, clf_name, metric)


# WILCOXON

# pairs_metrics_multi_line(method_names=list(method_names), data_np=data_np, dataset_names=dataset_names, metrics=metrics_alias, filename="wilcoxon_MOLO-MDH", ref_methods=method_names)
# pairs_metrics_multi_line(method_names=list(method_names), data_np=data_np, dataset_names=dataset_names, metrics=metrics_alias, filename="wilcoxon_SEMOOS", ref_methods=method_names)
# pairs_metrics_multi_line(method_names=list(method_names), data_np=data_np, dataset_names=dataset_names, metrics=metrics_alias, filename="wilcoxon_DE-Forest", ref_methods=method_names)
# pairs_metrics_multi_line(method_names=list(method_names), data_np=data_np, dataset_names=dataset_names, metrics=metrics_alias, filename="wilcoxon_MOOforest", ref_methods=method_names)

# DATASET TABLE

# make_description_table(dataset_names)

# RESULTS TABLE
result_tables_IR(dataset_paths, metrics_alias, mean_scores, method_names, stds)