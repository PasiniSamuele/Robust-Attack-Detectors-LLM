import pandas as pd
import os
test_synth_results_root = "experiments/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty"
test_synth_results_root_sap = "new_experiments_sap/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty"
test_synth_results_root_sap_open = "new_experiments_sap_open_source/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty"
runs = []
allowed_models = ["claude-3", "chat-bison", "llama3", "mixtral", "gpt-4"]
for root, dirs, files in os.walk(test_synth_results_root):
    if 'run_0' in dirs and any([model in root for model in allowed_models]):
            runs.append(os.path.join(root, 'run_0'))
for root, dirs, files in os.walk(test_synth_results_root_sap):
    if 'run_0' in dirs and any([model in root for model in allowed_models]):
            runs.append(os.path.join(root, 'run_0'))
for root, dirs, files in os.walk(test_synth_results_root_sap_open):
    if 'run_0' in dirs and any([model in root for model in allowed_models]):
            runs.append(os.path.join(root, 'run_0'))
            
#print(runs)


# runs_file = "runs_xss.txt"
# runs = []
# with open(runs_file, "r") as f:
#     for line in f:
#         runs.append(line.strip())

def from_dataset_to_splits(row):
   dataset = row["dataset"]
   #check in dataset ends with zero_shot or rag
   if dataset.endswith("zero_shot") or dataset.endswith("rag"):
      dataset = dataset +"_0"
   #print(dataset)
   parts = dataset.split("_")
   generation = "_".join(parts[2:-1])
   row["dataset_model"] = parts[0]
   row["dataset_temperature"] = parts[1]
   row["dataset_generation_mode"] = 0 if generation == "zero_shot" or generation == "few_shot" else 1
   row["dataset_examples_per_class"] = parts[3]
   return row

def create_experiment(row):
    row['experiment'] = row['model_name']+'_'+str(row['temperature'])+'_'+row['generation_mode']+'_'+str(row['examples_per_class'])
    return row
merge_path = "test_results_synth_merged.csv"
df_synth = pd.DataFrame()
for run in runs:
    for root, dirs, files in os.walk(run):
        #check if dirs is empty
        if not dirs and "test_results.csv" in files:
            df_synth = pd.concat([df_synth, pd.read_csv(os.path.join(root, "test_results.csv"))])

# test_results_file = "test_results_xss.txt"
# with open(test_results_file, "r") as f:
#     for line in f:
#         df_synth = pd.concat([df_synth, pd.read_csv(line.strip())])
df_synth["experiment"] = df_synth["model_name"]+"_"+df_synth["temperature"].astype(str) + "_"+df_synth["generation_mode"]+"_"+df_synth["examples_per_class"].astype(str)
df_synth = df_synth.apply(from_dataset_to_splits,axis=1)

df_synth.to_csv(merge_path, index=False)