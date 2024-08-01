import os
import json
from evaluate_run_on_synthetic import evaluate_synth_run
from argparse import Namespace
import pandas as pd
import concurrent.futures

experiments_root = "experiments/task_detect_xss_simple_prompt/template_create_function_readable"

#find all folders named run_0 recursively inside experiments_root
runs = []

allowed_models = ["claude-3", "chat-bison", "llama3", "mixtral", "gpt-4"]
for root, dirs, files in os.walk(experiments_root):
    if 'run_0' in dirs and any([model in root for model in allowed_models]):
            runs.append(os.path.join(root, 'run_0'))

datasets_root = "data/synthetic_datasets/task_detect_xss_simple_prompt/template_create_synthetic_dataset/prompt_parameters_medium_dataset/"

pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)

datasets = []
for root, dirs, files in os.walk(datasets_root):
    if 'run_0' in dirs:
        results_json = os.path.join(root, 'run_0',"results.json")
        #open json file and see if it is failed
        with open(results_json, 'r') as f:
            results = json.load(f)
            if results["success"] == True:
                datasets.append(os.path.join(root, 'run_0'))

top_ks = [1,3,5,10,15]
for run in runs: 
    for dataset in datasets:
        test_results_file_path = os.path.join(run, 
                                              "synthetic_results",
                                              '/'.join(dataset.split("/")[2:-1]),
                                              "test_results.csv")
        # print(test_results_file_path)
        # asdas
        if os.path.exists(test_results_file_path):
            print("Skipping", test_results_file_path)
            continue

        opt = Namespace(run=run, 
                        dataset_folder=dataset, 
                        top_k = top_ks,
                        plot = True,
                        isolated_execution = False,
                        parameters_file_name = "parameters.json",
                        function_name = "detect_sqli",
                        evaluation_folder = "synthetic_results",
                        result_file_name = "results.json",
                        top_k_metric = "accuracy",
                        test_results_file_name = "test_results.json"
                        )
        #pool.submit(evaluate_synth_run, opt)
        evaluate_synth_run(opt)
        
#find in runs all the files names test_results_csv
pool.shutdown(wait=True)
df = pd.DataFrame()
