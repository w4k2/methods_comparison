import shutil
import glob
import os


metrics_alias = ["ACC", "BAC", "Gmean", "Gmean2", "Gmean_pr", "F1score", "Recall", "Specificity", "Precision"]

def copy_files_with_pattern(source_folder, method, pattern):
    for metric in metrics_alias:
        file_tree = metric + "/*/" + pattern
        print(os.path.join(source_folder, file_tree))
        for file_path in glob.glob(os.path.join(source_folder, file_tree)):
            filename = "results/" + method + "/" + metric + "/" + file_path.split("/")[-2]
            print(filename)
            if not os.path.exists(filename):
                os.makedirs(filename)
            if os.path.isfile(file_path):
                target_folder = filename + "/" + method + ".csv"
                shutil.copy(file_path, target_folder)


# method = "SEMOOS"
# source_folder = "/home/joannagrzyb/work/SEMOOS/results/experiment_server/experiment1_9higher_part1/raw_results"
# source_folder = "/home/joannagrzyb/work/SEMOOS/results/experiment_server/experiment2_9higher_part2/raw_results"
# source_folder = "/home/joannagrzyb/work/SEMOOS/results/experiment_server/experiment3_9higher_part3/raw_results"
# source_folder = "/home/joannagrzyb/work/SEMOOS/results/experiment_server/experiment4_9lower/raw_results"
# copy_files_with_pattern(source_folder, method, 'MooEnsembleSVC.csv')

# method = "DE-Forest"
# source_folder = "/home/joannagrzyb/work/DE-Forest/results/experiment1/raw_results"
# copy_files_with_pattern(source_folder, method, 'DE_Forest.csv')

method = "MOOforest"
source_folder = "/home/joannagrzyb/work/MOOforest/results/experiment1/raw_results"
copy_files_with_pattern(source_folder, method, 'MOOforest.csv')

# method = "MOLO-MDH"
# source_folder = "/home/joannagrzyb/work/HellingerMOO/results/experiment1/raw_results"
# copy_files_with_pattern(source_folder, method, 'Margin_Diversity_Hellinger_Classifier.csv')


