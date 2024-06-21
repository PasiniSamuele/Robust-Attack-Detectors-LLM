import os
from argparse import Namespace
from evaluate_run import evaluate_run
import os
#experiments_root = "new_experiments_sap/task_detect_xss_simple_prompt/template_create_function_readable"
experiments_root = "experiments/task_detect_sqli_extended/template_create_function_readable"
#find all folders named run_0 recursively inside experiments_root
runs = []
for root, dirs, files in os.walk(experiments_root):
    #remove from dirs all the directories starting with exp_ or synthetic_
    for dir in dirs:
        if dir.startswith("run_0"):
            if not os.path.exists(os.path.join(root, dir, "test_results.json")) or True:
                runs.append(os.path.join(root, dir))
data = "data/test_sqli.csv"


for run in runs:
    print(run)
    evaluation_namespace = Namespace()

    evaluation_namespace.create_confusion_matrix = False
    evaluation_namespace.summarize_results = True
    evaluation_namespace.run = run
    evaluation_namespace.top_k = [1, 3, 5, 10, 15]
    evaluation_namespace.data = data
    evaluation_namespace.result_file_name = "test_results.json"
    evaluation_namespace.function_name = "detect_sqli"
    evaluation_namespace.isolated_execution = False
    evaluation_namespace.top_k_metric = "accuracy"


    evaluate_run(evaluation_namespace)
    #break
#find all folders named run_0 recursively inside datasets_root