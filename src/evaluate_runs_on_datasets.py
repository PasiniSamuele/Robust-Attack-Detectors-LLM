import os
from evaluate_run_on_synthetic import evaluate_synth_run
from argparse import Namespace
experiments_root = "experiments/task_detect_xss_simple_prompt/template_create_function_readable"
#find all folders named run_0 recursively inside experiments_root
runs = []
for root, dirs, files in os.walk(experiments_root):
    for dir in dirs:
        if dir.startswith("run_"):
            runs.append(os.path.join(root, dir))
datasets_root = "data/synthetic_datasets/task_detect_xss_simple_prompt/template_create_synthetic_dataset/prompt_parameters_medium_dataset/model_gpt-4-0125-preview"
#find all folders named run_0 recursively inside datasets_root
datasets = []
for root, dirs, files in os.walk(datasets_root):
    for dir in dirs:
        if dir.startswith("run_"):
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
                        top_k_metric = "accuracy"
                        )
        evaluate_synth_run(opt)
        #print(f"python src/evaluate_runs_on_datasets.py --run {run} --dataset_folder {dataset}")
