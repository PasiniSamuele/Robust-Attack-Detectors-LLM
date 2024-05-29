import pandas as pd

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

test_results_file = "experiments/task_detect_xss_simple_prompt/experiments_summary_test.csv"
test_synth_results_file = "experiments/task_detect_xss_simple_prompt/test_results_synth.csv"

rq1_file = "rq1.csv"
rq2_file = "rq2.csv"

best_exp = "gpt-4-0125-preview_0.0_rag_few_shot_5"
worst_case = "gpt-4-0125-preview_1.0_zero_shot_0"
middle_case = "gpt-4-0125-preview_0.5_few_shot_3"

dataset_to_keep = "gpt-4-0125-preview_1.0_rag_few_shot_5"

df = pd.read_csv(test_results_file)
examples_values = set(df["examples_per_class"].values.tolist())
new_columns = ["model_temperature"]
new_columns.extend(list(map(lambda x:f"no_rag_{x}",examples_values)))
new_columns.extend(list(map(lambda x:f"rag_{x}",examples_values)))
new_df = pd.DataFrame(columns=new_columns)
df["model_temperature"] = df["model_name"]+"_"+df["temperature"].astype(str)
df = df.sort_values(by=["model_temperature"])

for model_temperature in df["model_temperature"].unique():
    df_model_temperature = df[df["model_temperature"]==model_temperature]
    new_row = {"model_temperature":model_temperature}
    for examples_per_class in examples_values:
        if int(examples_per_class) == 0:
            # generation mode zero shot and examples per class = 0
            new_row[f"no_rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="zero_shot") & (df_model_temperature["examples_per_class"]==0)]["accuracy"].values[0]
            # generation mode rag and examples per class = 0
            new_row[f"rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="rag") & (df_model_temperature["examples_per_class"]==0)]["accuracy"].values[0]
        else:
            # generation mode few_shot and examples per class = examples per class
            new_row[f"no_rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["accuracy"].values[0]
            # generation mode rag_few_shot and examples per class = examples per class
            new_row[f"rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="rag_few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["accuracy"].values[0]
    new_df = pd.concat([new_df,pd.DataFrame(new_row,index=[0])], )

new_df.to_csv(rq1_file,index=False,float_format='%.3f')
df = df.apply(create_experiment, axis=1)

df_synth = pd.read_csv(test_synth_results_file)
df_synth["experiment"] = df_synth["model_name"]+"_"+df_synth["temperature"].astype(str) + "_"+df_synth["generation_mode"]+"_"+df_synth["examples_per_class"].astype(str)

top_ks = set(df_synth["top_k"].values.tolist())

new_columns = ["experiment", "dataset"]
new_columns.extend(list(map(lambda x:f"top_{x}_acc_diff",top_ks)))
new_columns.append("avg_accuracy")
new_columns.extend(list(map(lambda x:f"top_{x}_acc",top_ks)))

experiments_to_keep = [best_exp,middle_case,worst_case]
new_df_synth = pd.DataFrame(columns=new_columns)

# for experiment in experiments_to_keep:
#     df_keep = df_synth[df_synth["experiment"]==experiment]
#     df_keep = df_keep.apply(from_dataset_to_splits,axis=1)

#     df_keep = df_keep.sort_values(by=["dataset_model", "dataset_temperature", "dataset_generation_mode", "dataset_examples_per_class"])
#     for dataset in df_keep["dataset"].unique():
#         df_dataset = df_keep[df_keep["dataset"]==dataset]
#         new_row = {"experiment":experiment,"dataset":dataset}
#         #group by top_k and avg the accuracy and acc_diff
#         for top_k in top_ks:
#             df_top = df_dataset[df_dataset["top_k"]==top_k]
#             new_row[f"top_{top_k}_acc_diff"] = df_top["accuracy_diff"].mean()
#             new_row[f"top_{top_k}_acc"] = df_top["accuracy"].mean()
#             new_row["avg_accuracy"] = df[df["experiment"]==experiment]["accuracy"].mean()
#         new_df_synth = pd.concat([new_df_synth,pd.DataFrame(new_row,index=[0])])

for experiment in experiments_to_keep:
    df_keep = df_synth[df_synth["experiment"]==experiment]
    df_keep = df_keep.apply(from_dataset_to_splits,axis=1)

    df_keep = df_keep.sort_values(by=["dataset_model", "dataset_temperature", "dataset_generation_mode", "dataset_examples_per_class"])
    df_dataset = df_keep[df_keep["dataset"]==dataset_to_keep]
    new_row = {"experiment":experiment,"dataset":dataset_to_keep}
    #group by top_k and avg the accuracy and acc_diff
    for top_k in top_ks:
        df_top = df_dataset[df_dataset["top_k"]==top_k]
        new_row[f"top_{top_k}_acc_diff"] = df_top["accuracy_diff"].mean()
        new_row[f"top_{top_k}_acc"] = df_top["accuracy"].mean()
        new_row["avg_accuracy"] = df[df["experiment"]==experiment]["accuracy"].mean()
    new_df_synth = pd.concat([new_df_synth,pd.DataFrame(new_row,index=[0])])

new_df_synth.to_csv(rq2_file,index=False, float_format='%.3f')