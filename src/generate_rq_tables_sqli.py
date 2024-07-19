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

#test_results_file_old_sap = "old_experiments_sap/task_detect_xss_simple_prompt/experiments_summary_test.csv"
#test_results_file = "new_experiments_sap_open_source/task_detect_xss_simple_prompt/template_create_function_readable/experiments_summary_test.csv"
test_results_file = "experiments/task_detect_sqli_extended/template_create_function_readable/experiments_summary_test.csv"
test_results_file_sap = "new_experiments_sap_sqli/task_detect_sqli_extended/template_create_function_readable/experiments_summary_test.csv"

test_synth_results_file = "test_results_synth_merged_sqli.csv"

gen_rq1 = False
gen_rq2 = True

rq1_file = "rq1_sqli.csv"
rq1_large_file = "rq1_large_sqli.csv"

rq2_file = "rq2_sqli.csv"

rq1_plot_folder = "rq1_plots_sqli"
rq2_plot_folder = "rq2_plots_sqli"

df = pd.concat([pd.read_csv(test_results_file),pd.read_csv(test_results_file_sap)])
df["experiment"] = df["model_name"]+"_"+df["temperature"].astype(str)+"_"+df["generation_mode"]+"_"+df["examples_per_class"].astype(str)

df["model_temperature"] = df["model_name"]+"_"+df["temperature"].astype(str)


#keep only columns with  "gpt-4" in model_temperature or "claude-3" in model_temperature or "gpt-4" in model_temperature or "gcp" in model_temperature or "llama3"  in model_temperature or "mixtral-8x7b"  in model_temperature:
df = df[df["model_temperature"].str.contains("gpt-4|opus|sonnet|gpt-4|gcp|llama3|mixtral-8x7b", regex=True)]

#drop_columns with model_temperature == "gcp-chat-bison-001_0.0" or model_temperature == "anthropic-claude-3-sonnet_0.0" or model_temperature == "mixtral-8x7b-instruct-v01_0.0" or model_temperature == "llama3-70b-instruct_0.0"
df = df[~df["model_temperature"].isin(["gcp-chat-bison-001_0.0","gcp-chat-bison-001_0.5","anthropic-claude-3-sonnet_0.0","mixtral-8x7b-instruct-v01_0.0","llama3-70b-instruct_0.0"])]

best_exp_global = df[df["accuracy"]==df["accuracy"].max()]["experiment"].values[0]
worst_exp_global = df[df["accuracy"]==df["accuracy"].min()]["experiment"].values[0]
#keep the 3 closest cases to the mean of the avg_acc_diff
middle_case_global = df.iloc[(df["accuracy"]-df["accuracy"].mean()).abs().argsort()[:3]]["experiment"].values


if gen_rq1:
    #df = pd.concat([pd.read_csv(test_results_file),pd.read_csv(test_results_file_sap)])
    #df = pd.read_csv(test_results_file)
    examples_values = set(df["examples_per_class"].values.tolist())
    new_columns = ["model_temperature"]
    new_columns.extend(list(map(lambda x:f"no_rag_{x}",examples_values)))
    new_columns.extend(list(map(lambda x:f"rag_{x}",examples_values)))
    new_df = pd.DataFrame(columns=new_columns)
    df["model_temperature"] = df["model_name"]+"_"+df["temperature"].astype(str)
    df = df.sort_values(by=["model_temperature"])
    success_improvement_df = pd.DataFrame(columns=["model_temperature","examples_per_class","success_improvement"])
    rag_improvement_df = pd.DataFrame(columns=["model_temperature","examples_per_class","rag_improvement"])
    var_improvement_df = pd.DataFrame(columns=["model_temperature","examples_per_class","var_improvement"])

    for model_temperature in df["model_temperature"].unique():
        #keep only the model_temperature with gpt-4 or claude-3

        df_model_temperature = df[df["model_temperature"]==model_temperature]
        if model_temperature == "gcp-chat-bison-001_0.0" or model_temperature == "gcp-chat-bison-001_0.5" or model_temperature == "anthropic-claude-3-sonnet_0.0":   #bison and sonnet working only with higher temperature
            continue
        new_row = {"model_temperature":model_temperature}
        for examples_per_class in examples_values:
            if int(examples_per_class) == 0:
                # generation mode zero shot and examples per class = 0
                new_row[f"no_rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="zero_shot") & (df_model_temperature["examples_per_class"]==0)]["accuracy"].values[0]
                # generation mode rag and examples per class = 0
                new_row[f"rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="rag") & (df_model_temperature["examples_per_class"]==0)]["accuracy"].values[0]

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
                        accuracy_var_no_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="zero_shot") & (df_model_temperature["examples_per_class"]==0)]["accuracy_std"].values[0]
                        accuracy_var_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="rag") & (df_model_temperature["examples_per_class"]==0)]["accuracy_std"].values[0]
                        var_difference = accuracy_var_rag - accuracy_var_no_rag
                        var_improvement_df = pd.concat([var_improvement_df,pd.DataFrame({"model_temperature":model_temperature,"examples_per_class":examples_per_class,"var_improvement":var_difference},index=[0])])
            else:
                # generation mode few_shot and examples per class = examples per class
                new_row[f"no_rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["accuracy"].values[0]
                # generation mode rag_few_shot and examples per class = examples per class
                new_row[f"rag_{examples_per_class}"] = df_model_temperature[(df_model_temperature["generation_mode"]=="rag_few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["accuracy"].values[0]
                
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

                        accuracy_var_no_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["accuracy_std"].values[0]
                        accuracy_var_rag = df_model_temperature[(df_model_temperature["generation_mode"]=="rag_few_shot") & (df_model_temperature["examples_per_class"]==examples_per_class)]["accuracy_std"].values[0]
                        var_difference = accuracy_var_rag - accuracy_var_no_rag
                        var_improvement_df = pd.concat([var_improvement_df,pd.DataFrame({"model_temperature":model_temperature,"examples_per_class":examples_per_class,"var_improvement":var_difference},index=[0])])
        new_df = pd.concat([new_df,pd.DataFrame(new_row,index=[0])], )
    #get the number of the columns starting with no_rag
    no_rag_columns = [col for col in new_df.columns if col.startswith("no_rag")]
    #get the difference between all the columns starting with rag_ and all the columns starting with no_rag
    new_df["avg_acc_diff"] = new_df[[col for col in new_df.columns if col.startswith("rag")]].mean(axis=1) - new_df[no_rag_columns].mean(axis=1)
    
    new_df.to_csv(rq1_large_file,index=False,float_format='%.3f')

    no_rag_columns = [col for col in new_df.columns if col.startswith("no_rag")]
    #get the difference between all the columns starting with rag_ and all the columns starting with no_rag
    new_df["avg_acc_diff"] = new_df[[col for col in new_df.columns if col.startswith("rag")]].mean(axis=1) - new_df[no_rag_columns].mean(axis=1)
    best_exp = new_df[new_df["avg_acc_diff"]==new_df["avg_acc_diff"].max()]["model_temperature"].values[0]
    worst_exp = new_df[new_df["avg_acc_diff"]==new_df["avg_acc_diff"].min()]["model_temperature"].values[0]
    #keep the 3 closest cases to the mean of the avg_acc_diff
    middle_case = new_df.iloc[(new_df["avg_acc_diff"]-new_df["avg_acc_diff"].mean()).abs().argsort()[:3]]["model_temperature"].values

    #keep only rows with best exp, middle_case and worst_exp
    to_keep = [best_exp]
    to_keep.extend(middle_case)
    to_keep.append(worst_exp)
    new_df_reduced = new_df.loc[new_df["model_temperature"].isin(to_keep)]
    #order by avg_acc_diff decreasing
    new_df_reduced = new_df_reduced.sort_values(by=["avg_acc_diff"],ascending=False)

    new_df_reduced.to_csv(rq1_file,index=False,float_format='%.3f')
    sns.set_theme(style="whitegrid")

    os.makedirs(rq1_plot_folder, exist_ok=True)
    plt.figure(figsize=(10, 10))
    ax = sns.violinplot(data=rag_improvement_df, y=f"rag_improvement", color = "blue",  linewidth = 2, fill = True, alpha = 0.6)
    # ax.figure.set_size_inches(9,8)
    #ax.set_title(f"Improvement of Avg Accuracy using RAG", fontsize=22) 
    ax.set_ylabel("AVG Accuracy Difference", fontsize=28)
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

    plt.savefig(os.path.join(rq1_plot_folder,f"success_improvement.jpg"))
    plt.close()

    plt.figure(figsize=(10, 10))
    ax = sns.violinplot(data=var_improvement_df, y=f"var_improvement",  linewidth = 2, fill = True, color = "green", alpha = 0.6)
    ax.figure.set_size_inches(7,8)
    ax.set_title(f"Improvement of Accuracy Variance using RAG")
    ax.set_ylabel("Accuracy Variance Difference")
    ax.set_yticks(np.arange(-0.3,0.3, 0.05))
    # [single_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) for single_ax in ax.axes.flat]
    # [single_ax.set_xlim(-0.1,0.5) for single_ax in ax.axes.flat]
    # [single_ax.set_xticks(range(-0.1,0.5, 0.05)) for single_ax in ax.axes.flat]

    plt.savefig(os.path.join(rq1_plot_folder,f"variance_improvement.jpg"))
    plt.close()

if gen_rq2:
    df = df.apply(create_experiment, axis=1)

    df_synth = pd.read_csv(test_synth_results_file)
    df_synth = df_synth[df_synth["accuracy"] != 0]
    top_ks = set(df_synth["top_k"].values.tolist())
    top_ks = [1,3,5]

    new_columns = ["experiment", "dataset"]
    new_columns.extend(list(map(lambda x:f"top_{x}_acc_diff",top_ks)))
    new_columns.append("avg_accuracy")
    new_columns.extend(list(map(lambda x:f"top_{x}_acc",top_ks)))

    experiments_to_keep = [best_exp_global,middle_case_global[0],worst_exp_global]
    new_df_synth = pd.DataFrame(columns=new_columns)
    big_df_synth = pd.DataFrame(columns=new_columns)

    plots_df = pd.DataFrame()
    for experiment in df_synth.experiment.unique():
        if  "gcp-chat-bison-001_0.0" in experiment or "gcp-chat-bison-001_0.5" in experiment or "gpt-3.5" in experiment or "anthropic-claude-3-sonnet_0.0" in experiment or "mixtral-8x7b-instruct-v01_0.0" in experiment or "llama3-70b-instruct_0.0" in experiment:   #bison, sonnet, mistral and llama working only with higher temperature
            continue
        print(experiment)
        df_keep = df_synth[df_synth["experiment"]==experiment]
        df_keep = df_keep.sort_values(by=["dataset_model", "dataset_temperature", "dataset_generation_mode", "dataset_examples_per_class"])
        for dataset in df_keep["dataset"].unique():
            df_dataset = df_keep[df_keep["dataset"]==dataset]
            new_row = {"experiment":experiment,"dataset":dataset}
            #group by top_k and avg the accuracy and acc_diff
            for top_k in top_ks:
                df_top = df_dataset[df_dataset["top_k"]==top_k]
                new_row[f"top_{top_k}_acc_diff"] = df_top["accuracy_diff"].mean()
                new_row[f"top_{top_k}_acc"] = df_top["accuracy"].mean()
                new_row["avg_accuracy"] = df[df["experiment"]==experiment]["accuracy"].mean()
            big_df_synth = pd.concat([big_df_synth,pd.DataFrame(new_row,index=[0])])
    big_df_synth.to_csv("big_sqli.csv")
    
    def get_avg_acc_diff(row, top_ks):
        cumulative_acc_diff = 0
        for top_k in top_ks:
            cumulative_acc_diff += row[f"top_{top_k}_acc_diff"]
        row["avg_acc_diff"] = cumulative_acc_diff/len(top_ks)
        return row
    for experiment in experiments_to_keep:
        df_keep = df_synth[df_synth["experiment"]==experiment]

        df_keep = df_keep.sort_values(by=["dataset_model", "dataset_temperature", "dataset_generation_mode", "dataset_examples_per_class"])
        exp_in_dataset = big_df_synth[big_df_synth["experiment"]==experiment]
        exp_in_dataset = exp_in_dataset.apply(get_avg_acc_diff, axis=1, top_ks=top_ks)
        #select the row with the lowest avg_acc_diff
        best_dataset = exp_in_dataset[exp_in_dataset["avg_acc_diff"]==exp_in_dataset["avg_acc_diff"].min()]["dataset"].values[0]
        #select the row with the highest avg_acc_diff
        worst_dataset = exp_in_dataset[exp_in_dataset["avg_acc_diff"]==exp_in_dataset["avg_acc_diff"].max()]["dataset"].values[0]
        #get the mean of all the avg_acc_diff
        avgerage_acc_diff = exp_in_dataset["avg_acc_diff"].mean()
        #select the row with the closes avg_acc_diff to average_acc_diff
        avg_dataset = exp_in_dataset.iloc[(exp_in_dataset["avg_acc_diff"]-avgerage_acc_diff).abs().argsort()[:1]]["dataset"].values[0]
        datasets_to_keep = [best_dataset,avg_dataset,worst_dataset]
        for dataset in datasets_to_keep:
            df_dataset = df_keep[df_keep["dataset"]==dataset]
            new_row = {"experiment":experiment,"dataset":dataset}
            new_row["avg_accuracy"] = df[df["experiment"]==experiment]["accuracy"].mean()

            #group by top_k and avg the accuracy and acc_diff
            for top_k in top_ks:
                df_top = df_dataset[df_dataset["top_k"]==top_k]
                new_row[f"top_{top_k}_acc"] = df_top["accuracy"].mean()
                new_row[f"top_{top_k}_acc_diff"] = df_top["accuracy_diff"].mean()
                new_row[f"top_{top_k}_acc_improvement"] = df_top["accuracy"].mean() - df[df["experiment"]==experiment]["accuracy"].mean()
            new_df_synth = pd.concat([new_df_synth,pd.DataFrame(new_row,index=[0])])

    new_df_synth.to_csv(rq2_file,index=False, float_format='%.3f')


    df_keep = df_synth.copy()
    #df_keep = df_keep.apply(from_dataset_to_splits,axis=1)
    top_ks = [1,3,5]
    df_keep = df_keep.sort_values(by=["dataset_model", "dataset_temperature", "dataset_generation_mode", "dataset_examples_per_class"])
    
    for experiment in df_synth.experiment.unique():
        #if experiment does not contain one of gpt-4, bison, opus, sonnet,  llama3, mixtral-8x7b continue
        if "gpt-4" not in experiment and "bison" not in experiment and "opus" not in experiment and "sonnet" not in experiment and "llama3" not in experiment and "mixtral-8x7b" not in experiment:
            continue
        if  "gcp-chat-bison-001_0.0" in experiment or "gcp-chat-bison-001_0.5" in experiment or "gpt-3.5" in experiment or "anthropic-claude-3-sonnet_0.0" in experiment or "mixtral-8x7b-instruct-v01_0.0" in experiment or "llama3-70b-instruct_0.0" in experiment:   #bison, sonnet, mistral and llama working only with higher temperature
            continue

        df_keep_exp = df_keep[df_keep["experiment"]==experiment]
        for dataset in df_keep_exp["dataset"].unique():
            df_dataset = df_keep_exp[df_keep_exp["dataset"]==dataset]
            new_row = {"experiment":experiment,"dataset":dataset}
            #group by top_k and avg the accuracy and acc_diff
            new_row["avg_accuracy"] = df[df["experiment"]==experiment]["accuracy"].mean()

            for top_k in top_ks:
                df_top = df_dataset[df_dataset["top_k"]==top_k]
                new_row[f"top_{top_k}_acc"] = df_top["accuracy"].mean()
                new_row[f"top_{top_k}_acc_diff"] = df_top["accuracy_diff"].mean()
                new_row[f"top_{top_k}_acc_improvement"] = df_top["accuracy"].mean() - df[df["experiment"]==experiment]["accuracy"].mean()

            plots_df = pd.concat([plots_df,pd.DataFrame(new_row,index=[0])])
    plots_df.to_csv("plots_rq2_sqli.csv",index=False, float_format='%.3f')

    for top_k in top_ks:
        output_plots_folder = os.path.join(rq2_plot_folder, f"top_{top_k}")
        os.makedirs(output_plots_folder, exist_ok=True)
        ax = sns.displot(data=plots_df, x=f"top_{top_k}_acc_diff", palette = "hls", kind = "kde")
        plt.savefig(os.path.join(output_plots_folder,f"acc_diff.jpg"))
        ax = sns.displot(data=plots_df, x=f"top_{top_k}_acc_improvement", palette = "hls", kind = "kde")
        plt.savefig(os.path.join(output_plots_folder,f"acc_improvement.jpg"))
            