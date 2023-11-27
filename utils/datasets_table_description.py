import os
import numpy as np
import warnings
from pathlib import Path
from .load_datasets import load_dataset


warnings.filterwarnings("ignore")


def calc_imbalance_ratio(X, y):
    unique, counts = np.unique(y, return_counts=True)

    if len(counts) == 1:
        raise ValueError("Only one class in procesed data.")
    elif counts[0] > counts[1]:
        majority_name = unique[0]
        minority_name = unique[1]
    else:
        majority_name = unique[1]
        minority_name = unique[0]

    minority_ma = np.ma.masked_where(y == minority_name, y)
    minority = X[minority_ma.mask]

    majority_ma = np.ma.masked_where(y == majority_name, y)
    majority = X[majority_ma.mask]

    imbalance_ratio = majority.shape[0]/minority.shape[0]

    return imbalance_ratio


def make_description_table(dataset_names, DATASETS_DIR="./datasets"):
    print(DATASETS_DIR)
    X_all = []
    y_all = []
    imbalance_ratios = []
    for root, _, files in os.walk(DATASETS_DIR):
        for filename in filter(lambda _: _.endswith('.dat'), files):
            dataset_path = os.path.join(root, filename)
            dataset_name = Path(dataset_path).stem
            if dataset_name in dataset_names:
                X, y = load_dataset(dataset_path)
                IR = calc_imbalance_ratio(X, y)
                imbalance_ratios.append(IR)
                X_all.append(X)
                y_all.append(y)

    IR_argsorted = np.argsort(imbalance_ratios)
    if not os.path.exists("./results/tables/"):
        os.makedirs("./results/tables/")
    with open("./results/tables/datasets.tex", "w+") as file:
        for id, arg in enumerate(IR_argsorted):
            id += 1
            number_of_features = X_all[arg].shape[1]
            number_of_objects = len(y_all[arg])
            ds_name = dataset_names[arg].replace("_", "\\_")
            print("%d & \\emph{%s} & %0.2f & %d & %d \\\\" % (id, ds_name, imbalance_ratios[arg], number_of_objects, number_of_features), file=file)


def result_tables_IR(dataset_names, metrics_alias, mean_scores, methods, stds):
    imbalance_ratios = []
    for dataset_name in dataset_names:
        X, y = load_dataset(dataset_name)
        IR = calc_imbalance_ratio(X, y)
        imbalance_ratios.append(IR)
    IR_argsorted = np.argsort(imbalance_ratios)
    print(len(IR_argsorted))
    for metric_id, metric in enumerate(metrics_alias):
        if not os.path.exists("results/tables_IR/"):
            os.makedirs("results/tables_IR/")
        with open("results/tables_IR/results_%s.tex" % (metric), "w+") as file:
            print("\\begin{table}[!ht]", file=file)
            print("\\centering", file=file)
            print("\\caption{%s}" % (metric), file=file)
            columns = "l"
            for i in methods:
                columns += " c"
            print("\\scalebox{0.4}{", file=file)
            print("\\begin{tabular}{%s}" % columns, file=file)
            print("\\hline", file=file)
            columns_names = "\\textbf{dataset} &"
            for name in methods:
                name = name.replace("_", "-")
                columns_names += f'\\textbf{{{name}}} & '
            columns_names = columns_names[:-3]
            columns_names += "\\\\"
            print(columns_names, file=file)
            print("\\hline", file=file)
            for id, arg in enumerate(IR_argsorted):
                # id += 1
                # line = "%d" % (id)
                lineir = "%s" % (dataset_names[arg])
                lineir = lineir.split("/")[-1]
                lineir = lineir.split(".")[0]
                lineir = lineir.replace("_", "-")
                print(lineir)
                line = "%s" % (lineir)
                # print(line, lineir)
                line_values = []
                line_values = mean_scores[arg, metric_id, :]
                max_value = np.amax(line_values)
                for clf_id, clf_name in enumerate(methods):
                    if mean_scores[arg, metric_id, clf_id] == max_value:
                        line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                    else:
                        line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                line += " \\\\"
                print(line, file=file)
            print("\\end{tabular}}", file=file)
            print("\\end{table}", file=file)

