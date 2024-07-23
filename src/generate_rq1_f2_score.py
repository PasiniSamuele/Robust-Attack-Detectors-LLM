import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
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

test_results_file_old_sap = "old_experiments_sap/task_detect_xss_simple_prompt/experiments_summary_test.csv"
test_results_file_sap = "new_experiments_sap/task_detect_xss_simple_prompt/template_create_function_readable/experiments_summary_test.csv"
test_results_file_sap_open_source = "new_experiments_sap_open_source/task_detect_xss_simple_prompt/template_create_function_readable/experiments_summary_test.csv"

test_results_file = "experiments/task_detect_xss_simple_prompt/experiments_summary_test.csv"
test_synth_results_file = "test_results_synth_merged.csv"
gen_rq1 = True

rq1_large_file = "rq1_large_f2.csv"
rq1_file = "rq1_f2.csv"



rq1_plot_folder = "rq1_plots_f2"

def calculate_f2(row):
    precision = float(row["precision"])
    recall = float(row["recall"])
    if precision == 0 and recall == 0:
        row["f2"] = 0
    else:
        row["f2"] = (((1 + 4) * precision * recall) / (4 * precision + recall))
    return row


df = pd.concat([pd.read_csv(test_results_file),pd.read_csv(test_results_file_sap), pd.read_csv(test_results_file_sap_open_source)])
df["experiment"] = df["model_name"]+"_"+df["temperature"].astype(str)+"_"+df["generation_mode"]+"_"+df["examples_per_class"].astype(str)
df["model_temperature"] = df["model_name"]+"_"+df["temperature"].astype(str)

#apply calculate_f2
df = df.apply(calculate_f2, axis=1)
df["f2_std"] = df["accuracy_std"]

#keep only columns with  "gpt-4" in model_temperature or "opus" in model_temperature or "sonnet" in model_temperature or "gpt-4" in model_temperature or "gcp" in model_temperature or "llama3"  in model_temperature or "mixtral-8x7b"  in model_temperature:
df = df[df["model_temperature"].str.contains("gpt-4|opus|sonnet|gpt-4|gcp|llama3|mixtral-8x7b", regex=True)]

#drop_columns with model_temperature == "gcp-chat-bison-001_0.0" or model_temperature == "anthropic-claude-3-sonnet_0.0" or model_temperature == "mixtral-8x7b-instruct-v01_0.0" or model_temperature == "llama3-70b-instruct_0.0"
df = df[~df["model_temperature"].isin(["gcp-chat-bison-001_0.0","gcp-chat-bison-001_0.5","anthropic-claude-3-sonnet_0.0","mixtral-8x7b-instruct-v01_0.0","llama3-70b-instruct_0.0"])]

best_exp_global = df[df["f2"]==df["f2"].max()]["experiment"].values[0]
worst_exp_global = df[df["f2"]==df["f2"].min()]["experiment"].values[0]
#keep the 3 closest cases to the mean of the avg_f2_diff
middle_case_global = df.iloc[(df["f2"]-df["f2"].mean()).abs().argsort()[:3]]["experiment"].values


if gen_rq1:
    #df = pd.concat([pd.read_csv(test_results_file),pd.read_csv(test_results_file_sap)])
    #df = pd.read_csv(test_results_file)
    examples_values = set(df["examples_per_class"].values.tolist())
    new_columns = ["model_temperature"]
    new_columns.extend(list(map(lambda x:f"no_rag_{x}",examples_values)))
    new_columns.extend(list(map(lambda x:f"rag_{x}",examples_values)))
    new_df = pd.DataFrame(columns=new_columns)
    df = df.sort_values(by=["model_temperature"])
    success_improvement_df = pd.DataFrame(columns=["model_temperature","examples_per_class","success_improvement"])
    rag_improvement_df = pd.DataFrame(columns=["model_temperature","examples_per_class","rag_improvement"])
    var_improvement_df = pd.DataFrame(columns=["model_temperature","examples_per_class","var_improvement"])

    for model_temperature in df["model_temperature"].unique():
        #keep only the model_temperature with gpt-4 or claude-3
        if "gpt-4" not in model_temperature and "claude-3" not in model_temperature and "gpt-4" not in model_temperature and "gcp" not in model_temperature and "llama3" not in model_temperature and "mixtral-8x7b" not in model_temperature:
            continue
        df_model_temperature = df[df["model_temperature"]==model_temperature]
        if model_temperature == "gcp-chat-bison-001_0.0" or model_temperature == "gcp-chat-bison-001_0.5" or model_temperature == "anthropic-claude-3-sonnet_0.0":   #bison and sonnet working only with higher temperature
            continue
        new_row = {"model_temperature":model_temperature}
        for examples_per_class in examples_values:
            if int(examples_per_class) == 0:
                # generation mode zero shot and examples per class = 0
                new_row[f"no_rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="zero_shot") & (df_model_temperature["examples_per_class"]==0)]["f2"].values[0]
                # generation mode rag and examples per class = 0
                new_row[f"rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="rag") & (df_model_temperature["examples_per_class"]==0)]["f2"].values[0]

                successes_without_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="zero_shot") & (df_model_temperature["examples_per_class"]==0)]["successes"].values[0]
                successes_with_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="rag") & (df_model_temperature["examples_per_class"]==0)]["successes"].values[0]
                if successes_without_rag <=20 or successes_with_rag <=20:
                    success_difference = successes_with_rag - successes_without_rag
                    success_improvement_df = pd.concat([success_improvement_df,pd.DataFrame({"model_temperature":model_temperature,"examples_per_class":examples_per_class,"success_improvement":success_difference},index=[0])])
                        
                if new_row[f"rag_{examples_per_class}"]!= 0 and new_row[f"no_rag_{examples_per_class}"]!= 0:
                    if successes_without_rag !=0 and successes_with_rag !=0:
                        rag_difference =  new_row[f"rag_{examples_per_class}"] - new_row[f"no_rag_{examples_per_class}"]
                        rag_improvement_df = pd.concat([rag_improvement_df,pd.DataFrame({"model_temperature":model_temperature,"examples_per_class":examples_per_class,"rag_improvement":rag_difference},index=[0])])
                    if successes_without_rag - successes_with_rag < 10 and successes_with_rag - successes_without_rag < 10:
                        f2_var_no_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="zero_shot") & (df_model_temperature["examples_per_class"]==0)]["f2_std"].values[0]
                        f2_var_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="rag") & (df_model_temperature["examples_per_class"]==0)]["f2_std"].values[0]
                        var_difference = f2_var_rag - f2_var_no_rag
                        var_improvement_df = pd.concat([var_improvement_df,pd.DataFrame({"model_temperature":model_temperature,"examples_per_class":examples_per_class,"var_improvement":var_difference},index=[0])])
            else:
                # generation mode few_shot and examples per class = examples per class
                new_row[f"no_rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["f2"].values[0]
                # generation mode rag_few_shot and examples per class = examples per class
                new_row[f"rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="rag_few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["f2"].values[0]
                
                successes_without_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["successes"].values[0]
                successes_with_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="rag_few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["successes"].values[0]
                if successes_without_rag <=20 or successes_with_rag <=20:
                    success_difference = successes_with_rag - successes_without_rag
                    success_improvement_df = pd.concat([success_improvement_df,pd.DataFrame({"model_temperature":model_temperature,"examples_per_class":examples_per_class,"success_improvement":success_difference},index=[0])])
                   
                if new_row[f"rag_{examples_per_class}"]!= 0 and new_row[f"no_rag_{examples_per_class}"]!= 0:
                    if successes_without_rag !=0 and successes_with_rag !=0:
                        rag_difference =  new_row[f"rag_{examples_per_class}"] - new_row[f"no_rag_{examples_per_class}"]
                        rag_improvement_df = pd.concat([rag_improvement_df,pd.DataFrame({"model_temperature":model_temperature,"examples_per_class":examples_per_class,"rag_improvement":rag_difference},index=[0])])
                    if successes_without_rag - successes_with_rag < 10 and successes_with_rag - successes_without_rag < 10:

                        f2_var_no_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["f2_std"].values[0]
                        f2_var_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="rag_few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["f2_std"].values[0]
                        var_difference = f2_var_rag - f2_var_no_rag
                        var_improvement_df = pd.concat([var_improvement_df,pd.DataFrame({"model_temperature":model_temperature,"examples_per_class":examples_per_class,"var_improvement":var_difference},index=[0])])
        new_df = pd.concat([new_df,pd.DataFrame(new_row,index=[0])], )
    #get the number of the columns starting with no_rag
    no_rag_columns = [col for col in new_df.columns if col.startswith("no_rag")]
    #get the difference between all the columns starting with rag_ and all the columns starting with no_rag
    new_df["avg_f2_diff"] = new_df[[col for col in new_df.columns if col.startswith("rag")]].mean(axis=1) - new_df[no_rag_columns].mean(axis=1)
    
    new_df.to_csv(rq1_large_file,index=False,float_format='%.3f')

    no_rag_columns = [col for col in new_df.columns if col.startswith("no_rag")]
    #get the difference between all the columns starting with rag_ and all the columns starting with no_rag
    new_df["avg_f2_diff"] = new_df[[col for col in new_df.columns if col.startswith("rag")]].mean(axis=1) - new_df[no_rag_columns].mean(axis=1)
    best_exp = new_df[new_df["avg_f2_diff"]==new_df["avg_f2_diff"].max()]["model_temperature"].values[0]
    worst_exp = new_df[new_df["avg_f2_diff"]==new_df["avg_f2_diff"].min()]["model_temperature"].values[0]
    #keep the 3 closest cases to the mean of the avg_f2_diff
    middle_case = new_df.iloc[(new_df["avg_f2_diff"]-new_df["avg_f2_diff"].mean()).abs().argsort()[:3]]["model_temperature"].values


    #keep only rows with best exp, middle_case and worst_exp
    to_keep = [best_exp]
    to_keep.extend(middle_case)
    to_keep.append(worst_exp)
    new_df_reduced = new_df.loc[new_df["model_temperature"].isin(to_keep)]
    #order by avg_f2_diff decreasing
    new_df_reduced = new_df_reduced.sort_values(by=["avg_f2_diff"],ascending=False)

    new_df_reduced.to_csv(rq1_file,index=False,float_format='%.3f')
    sns.set_theme(style="whitegrid")
    rag_improvement_df_gb = rag_improvement_df.groupby("model_temperature").mean().reset_index()[["model_temperature", "rag_improvement"]]
    os.makedirs(rq1_plot_folder, exist_ok=True)
    plt.figure(figsize=(10, 10))
    rag_improvement_df_gb = rag_improvement_df_gb.replace({"anthropic-claude-3-opus":"OPUS",
                                                            "anthropic-claude-3-sonnet": "SONNET",
                                                            "gcp-chat-bison-001":"PALM",
                                                            "gpt-4-0125-preview":"GPT-4T",
                                                            "gpt-4-1106-preview":"GPT-4",
                                                            "llama3-70b-instruct":"LLAMA",
                                                            "mixtral-8x7b-instruct-v01":"MIXTRAL"                   
                                                            },
                                                                                        regex = True)
    #order rag_improvement_df by rag_improvement
    rag_improvement_df_gb = rag_improvement_df_gb.sort_values(by=["rag_improvement"])
    ax = sns.barplot(data=rag_improvement_df_gb, x = "model_temperature", y = "rag_improvement", palette = sns.color_palette(palette='Oranges', n_colors = len(rag_improvement_df_gb)))
    # ax.figure.set_size_inches(9,8)
    #ax.set_title(f"Improvement of Avg f2 using RAG", fontsize=22) 
    ax.set_ylabel("AVG f2 Difference", fontsize=28)
    ax.set_xlabel("Model-Temperature pairs", fontsize=28)
    #rotate x_ticks
    plt.xticks(rotation=60)
    #ax.set_yticks(np.arange(-0.4,0.5, 0.1))
    #set the font size of ytickes to 19
    #ax.tick_params(axis='y', labelsize=25)
    # [single_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) for single_ax in ax.axes.flat]
    # [single_ax.set_xlim(-0.1,0.5) for single_ax in ax.axes.flat]
    # [single_ax.set_xticks(range(-0.1,0.5, 0.05)) for single_ax in ax.axes.flat]
    plt.tight_layout()
    plt.savefig(os.path.join(rq1_plot_folder,f"rag_improvement_hist.pdf"), transparent=True)
    plt.close()

    os.makedirs(rq1_plot_folder, exist_ok=True)
    plt.figure(figsize=(10, 10))
    ax = sns.violinplot(data=rag_improvement_df, y=f"rag_improvement", color = "firebrick",  linewidth = 2, fill = True, alpha = 0.6)
    # ax.figure.set_size_inches(9,8)
    #ax.set_title(f"Improvement of Avg f2 using RAG", fontsize=22) 
    ax.set_ylabel("AVG f2 Difference", fontsize=28)
    #ax.set_xlabel("Density")
    ax.set_yticks(np.arange(-0.4,0.5, 0.1))
    #set the font size of ytickes to 19
    ax.tick_params(axis='y', labelsize=25)
    # [single_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) for single_ax in ax.axes.flat]
    # [single_ax.set_xlim(-0.1,0.5) for single_ax in ax.axes.flat]
    # [single_ax.set_xticks(range(-0.1,0.5, 0.05)) for single_ax in ax.axes.flat]
    plt.tight_layout()
    plt.savefig(os.path.join(rq1_plot_folder,f"rag_improvement.pdf"), transparent=True)
    plt.close()

    plt.figure(figsize=(10, 10))
    ax = sns.violinplot(data=success_improvement_df, y=f"success_improvement", color = "red",  linewidth = 2, fill = True, alpha = 0.6)
    ax.figure.set_size_inches(7,8)
    ax.set_title(f"Improvement of Successes using RAG") 
    ax.set_ylabel("Successes Difference")
    ax.set_yticks(np.arange(-100 ,100, 10))
    # [single_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) for single_ax in ax.axes.flat]
    # [single_ax.set_xlim(-0.1,0.5) for single_ax in ax.axes.flat]
    # [single_ax.set_xticks(range(-0.1,0.5, 0.05)) for single_ax in ax.axes.flat]

    plt.savefig(os.path.join(rq1_plot_folder,f"success_improvement.jpg"), transparent=True)
    plt.close()

    plt.figure(figsize=(10, 10))
    ax = sns.violinplot(data=var_improvement_df, y=f"var_improvement",  linewidth = 2, fill = True, color = "green", alpha = 0.6)
    ax.figure.set_size_inches(7,8)
    ax.set_title(f"Improvement of f2 Variance using RAG")
    ax.set_ylabel("f2 Variance Difference")
    ax.set_yticks(np.arange(-0.3,0.3, 0.05))
    # [single_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) for single_ax in ax.axes.flat]
    # [single_ax.set_xlim(-0.1,0.5) for single_ax in ax.axes.flat]
    # [single_ax.set_xticks(range(-0.1,0.5, 0.05)) for single_ax in ax.axes.flat]

    plt.savefig(os.path.join(rq1_plot_folder,f"variance_improvement.jpg"))
    plt.close()

