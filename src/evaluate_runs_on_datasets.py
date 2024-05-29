import os
import json
from evaluate_run_on_synthetic import evaluate_synth_run
from argparse import Namespace
import pandas as pd
experiments_root = "experiments/task_detect_xss_simple_prompt/template_create_function_readable"
#find all folders named run_0 recursively inside experiments_root
runs = []
for root, dirs, files in os.walk(experiments_root):
    for dir in dirs:
        if dir.startswith("run_0"):
            runs.append(os.path.join(root, dir))
datasets_root = "data/synthetic_datasets/task_detect_xss_simple_prompt/template_create_synthetic_dataset/prompt_parameters_medium_dataset/"
#find all folders named run_0 recursively inside datasets_root
datasets = []
for root, dirs, files in os.walk(datasets_root):
    for dir in dirs:
        if dir.startswith("run_"):
            #open results file
            results_json = os.path.join(root, dir,"results.json")
            #open json file and see if it is failed
            with open(results_json, 'r') as f:
                results = json.load(f)
                if results["success"] == True:
                    datasets.append(os.path.join(root, dir))

top_ks = [1,3,5,10,15]
for run in runs: 
    for dataset in datasets:
        opt = Namespace(run=run, 
                        dataset_folder=dataset, 
                        top_k = top_ks,
                        plot = True,
                        isolated_execution = False,
                        parameters_file_name = "parameters.json",
                        function_name = "detect_xss",
                        evaluation_folder = "synthetic_results",
                        result_file_name = "results.json",
                        top_k_metric = "accuracy",
                        test_results_file_name = "test_results.json"
                        )
        evaluate_synth_run(opt)
#find in runs all the files names test_results_csv
df = pd.DataFrame()
for run in runs:
    for root, dirs, files in os.walk(run):
        for file in files:
            if file == "test_results.csv":
                df = pd.concat([df, pd.read_csv(os.path.join(root, file))])
df.to_csv("test_results_synth.csv")
        



# diff_metrics = get_accuracy_diff_metrics(runs, datasets)
# hue_order = diff_metrics.groupby('synth_dataset_name')["synth_dataset_name"].first().sort_values().index

# box_metrics = os.path.join(experiments_root, "box_plots")
# experiments_synth_boxplots_by_model(diff_metrics, os.path.join(box_metrics,"accuracy_diff", "top_k"), "top_k","accuracy_diff",hue_order)
# experiments_synth_boxplots_by_model(diff_metrics, os.path.join(box_metrics,"accuracy_diff", "temperature"), "temperature","accuracy_diff",hue_order)

# experiments_synth_boxplots_by_model(diff_metrics, os.path.join(box_metrics,"avg_std", "top_k"), "top_k","avg_std",hue_order)
# experiments_synth_boxplots_by_model(diff_metrics, os.path.join(box_metrics,"avg_std", "temperature"), "temperature","avg_std",hue_order)


# for model in diff_metrics["model"].unique():
#     experiments_synth_boxplots(diff_metrics, os.path.join(box_metrics,"accuracy_diff","temperature_top_k"), model, "temperature", "top_k", "accuracy_diff",hue_order)
#     experiments_synth_boxplots(diff_metrics, os.path.join(box_metrics,"accuracy_diff","temperature_generation_mode"),model, "temperature","generation_mode", "accuracy_diff",hue_order)
#     experiments_synth_boxplots(diff_metrics, os.path.join(box_metrics,"avg_std","temperature_top_k"), model, "temperature", "top_k", "avg_std",hue_order)
#     experiments_synth_boxplots(diff_metrics, os.path.join(box_metrics,"avg_std","temperature_generation_mode"),model, "temperature","generation_mode", "avg_std",hue_order)


