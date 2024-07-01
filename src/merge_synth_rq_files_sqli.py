import pandas as pd
import os
test_synth_results_root = "experiments/task_detect_sqli_extended/template_create_function_readable/prompt_parameters_empty"
#test_synth_results_root_sap = "new_experiments_sap/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty"
#test_synth_results_root_sap_open = "new_experiments_sap_open_source/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty"
runs = []
for root, dirs, files in os.walk(test_synth_results_root):
    for dir in dirs:
        if dir.startswith("run_0"):
            runs.append(os.path.join(root, dir))
# for root, dirs, files in os.walk(test_synth_results_root_sap):
#     for dir in dirs:
#         if dir.startswith("run_0"):
#             runs.append(os.path.join(root, dir))
# for root, dirs, files in os.walk(test_synth_results_root_sap_open):
#     for dir in dirs:
#         if dir.startswith("run_0"):
#             runs.append(os.path.join(root, dir))
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
merge_path = "test_results_synth_merged_sqli.csv"
df_synth = pd.DataFrame()
for run in runs:
    for root, dirs, files in os.walk(run):
        for file in files:
            if file == "test_results.csv":
                df_synth = pd.concat([df_synth, pd.read_csv(os.path.join(root, file))])
df_synth["experiment"] = df_synth["model_name"]+"_"+df_synth["temperature"].astype(str) + "_"+df_synth["generation_mode"]+"_"+df_synth["examples_per_class"].astype(str)
df_synth = df_synth.apply(from_dataset_to_splits,axis=1)

df_synth.to_csv(merge_path, index=False)