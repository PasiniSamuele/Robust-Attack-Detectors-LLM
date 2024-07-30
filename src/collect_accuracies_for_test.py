import os
import json
import pandas as pd
from utils.path_utils import get_exp_subfolders
rq1_file = "rq1_large.csv"
df_rq1 = pd.read_csv(rq1_file)
#get the values of the column model_temperature as a list
mts = df_rq1["model_temperature"].values.tolist()

test_synth_results_root = "experiments/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty"
test_synth_results_root_sap = "new_experiments_sap/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty"
test_synth_results_root_sap_open = "new_experiments_sap_open_source/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty"
runs = []

for root, dirs, files in os.walk(test_synth_results_root):
    for dir in dirs:
        if dir.startswith("run_0"):
            runs.append(os.path.join(root, dir))

            
for root, dirs, files in os.walk(test_synth_results_root_sap):
    
    for dir in dirs:
        if dir.startswith("run_0"):
            runs.append(os.path.join(root, dir))
for root, dirs, files in os.walk(test_synth_results_root_sap_open):
    
    for dir in dirs:
        if dir.startswith("run_0"):
            runs.append(os.path.join(root, dir))

rag_accuracies = []
no_rag_accuracies = []

accuracies = pd.DataFrame(columns=["model_temperature","use_rag", "few_shot_examples", "accuracy", "f2"])
for run in runs:
    #open parameters.json file
    parameters_file = os.path.join(run, "parameters.json")
    with open(parameters_file) as f:
        parameters = json.load(f)
    model_temperature = parameters["model_name"]+"_"+str(parameters["temperature"])
    if model_temperature not in mts:
        continue
    use_rag = True if parameters["generation_mode"] =="rag" or parameters["generation_mode"] =="rag_few_shot" else False
    exps = get_exp_subfolders(run)
    for exp in exps:
        #open test_results.json file
        test_results_file = os.path.join(exp, "test_results.json")
        with open(test_results_file) as f:
            test_results = json.load(f)
            if not test_results["failed"]:
                if test_results["results"]["precision"] != 0 and  test_results["results"]["recall"] != 0:
                    f2 = (((1 + 4) * test_results["results"]["precision"] * test_results["results"]["recall"]) / (4 * test_results["results"]["precision"] + test_results["results"]["recall"]))
                else:
                    f2 = 0  
                row = pd.DataFrame([{"model_temperature":model_temperature, "use_rag":use_rag, "few_shot_examples":parameters["examples_per_class"], "accuracy":test_results["results"]["accuracy"], "f2":f2}])
                accuracies = pd.concat([accuracies, row])

#save json file with accuracies
accuracies.to_csv("accuracies.csv", index=False)
        
