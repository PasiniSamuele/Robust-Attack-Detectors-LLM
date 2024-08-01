import os
from argparse import Namespace
from evaluate_run import evaluate_run
import os
experiments_root = "generated_function_runs/task_detect_sqli_extended/template_create_function_readable"
#find all folders named run_0 recursively inside experiments_root
total = 0
runs = []
for root, dirs, files in os.walk(experiments_root):
    #remove from dirs all the directories starting with exp_ or synthetic_
    for dir in dirs:
        if dir.startswith("run_0"):
            if not os.path.exists(os.path.join(root, dir, "val_results.json")):
                runs.append(os.path.join(root, dir))
data = "data/sqli/val.csv"

for run in runs:
    print(run)
    evaluation_namespace = Namespace()

    evaluation_namespace.create_confusion_matrix = False
    evaluation_namespace.summarize_results = True
    evaluation_namespace.run = run
    evaluation_namespace.data = data
    evaluation_namespace.result_file_name = "val_results.json"
    evaluation_namespace.function_name = "detect_sqli"
    evaluation_namespace.top_k_metric = "accuracy"


    evaluate_run(evaluation_namespace)
