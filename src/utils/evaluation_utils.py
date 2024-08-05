from argparse import Namespace
import pandas as pd
import seaborn as sn
import os 
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import Levenshtein
from utils.path_utils import  get_exp_subfolders, get_not_exp_subfolders, get_run_parameters
import json
from utils.utils import  NpEncoder

def create_confusion_matrix(test_set:pd.DataFrame,
                            label_column:str = "label",
                            prediction_column:str = "prediction",
                            to_aggregate_colum:str = "Payloads")->pd.DataFrame:
    grouped_results = test_set.groupby([label_column, prediction_column])[to_aggregate_colum].count().reset_index(name="count")
    cm = grouped_results.pivot(index=label_column, columns=prediction_column, values='count').fillna(0)
    missing_cols = [col for col in cm.index if col not in cm.columns]
    for col in missing_cols:
        cm[col] = 0
    cm = cm[cm.index.values]
    return cm

def save_confusion_matrix(cm:pd.DataFrame, path:str)->None:
    plt.clf()
    sn.heatmap(cm, annot=True, fmt='g',cmap="YlGnBu", cbar=False) 
    plt.savefig(os.path.join(path, 'confusion_matrix.png'))

def get_results_from_cm(cm:pd.DataFrame)->dict:
    results = dict()
    true_positives = cm[1][1]
    true_negatives = cm[0][0]
    false_positives = cm[1][0]
    false_negatives = cm[0][1]

    classified_positives = true_positives + false_positives
    labeled_positives = true_positives + false_negatives
    all_instances = (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / classified_positives if classified_positives > 0 else 0
    recall = true_positives / labeled_positives if labeled_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (true_positives + true_negatives) / all_instances if all_instances > 0 else 0

    results["true_positives"] = true_positives
    results["true_negatives"] = true_negatives
    results["false_positives"] = false_positives
    results["false_negatives"] = false_negatives
    results["total"] = true_positives + true_negatives + false_positives + false_negatives
    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = f1
    results["accuracy"] = accuracy

    for k, v in results.items():
        if np.isnan(v):
            results[k] = 0
    return results


def average_metric(single_results:list, metric:str)->Tuple[float, float, float]:
    avg =  sum(map(lambda x: x["results"][metric], single_results)) / len(single_results) if len(single_results) > 0 else 0
    std = np.std(list(map(lambda x: x["results"][metric], single_results))) if len(single_results) > 0 else 0
    var = np.var(list(map(lambda x: x["results"][metric], single_results))) if len(single_results) > 0 else 0
    return avg, std, var

def get_top_k_results(exps):
        top_k_results = dict()
        #write the top_k experiments names
        top_k_results["experiments"] = list(map(lambda x: x["experiment"], exps))
        #calculate avg, std and var for each metric
        top_k_results["accuracy"], top_k_results["accuracy_std"], top_k_results["accuracy_var"] = average_metric(exps, "accuracy")
        top_k_results["precision"], top_k_results["precision_std"], top_k_results["precision_var"] = average_metric(exps, "precision")
        top_k_results["recall"], top_k_results["recall_std"], top_k_results["recall_var"] = average_metric(exps, "recall")
        top_k_results["f1"], top_k_results["f1_std"], top_k_results["f1_var"] = average_metric(exps, "f1")
        return top_k_results

def summarize_results(single_results:list,
                      top_k_metric:str = "accuracy",
                      top_k:list = [])->dict:
    results = dict()
    #filter out the failed experiments
    successful_experiments = list(filter(lambda x: not x["failed"], single_results))
    #count the number of successful experiments
    results["successes"] = len(successful_experiments)
    #count the number of failed experiments
    results["failures"] = len(single_results) - results["successes"]
    #count the total number of experiments
    results["total"] = len(single_results)
    #calculate avg, std and var for each metric
    results["accuracy"], _, _ = average_metric(successful_experiments, "accuracy")
    results["precision"], _, _ = average_metric(successful_experiments, "precision")
    results["recall"], _, _ = average_metric(successful_experiments, "recall")
    results["f1"], _, _ = average_metric(successful_experiments, "f1")
    results["f2"] = (((1 + 4) * results["precision"] * results["recall"]) / (4 * results["precision"] + results["recall"]))
    for top in top_k:
        if top > len(successful_experiments):
            continue
        #keep only the top n experiments based on the top_k_metric
        exps = sorted(successful_experiments, key=lambda x: x["results"][top_k_metric], reverse=True)[:top]
        results[f"top_{top}"] = get_top_k_results(exps)
    return results

def get_top_k_avg_std(exps, top_k_metric:str = "accuracy"):
    top_k_results = dict()
    #write the top_k experiments names
    top_k_results["experiments"] = list(map(lambda x: x["experiment"], exps))
    #calculate avg, std and var for each metric
    top_k_results[f'avg_std_{top_k_metric}'], _, _ = average_metric(exps, f"{top_k_metric}_std")
    return top_k_results

def get_avg_std(single_results:list,
                top_k_metric:str = "accuracy",
                top_k:list = [1,3,5,10,15,40]):
    results = dict()

    #filter out the failed experiments
    successful_experiments = list(filter(lambda x: not x["failed"], single_results))

    #count the number of successful experiments
    results["successes"] = len(successful_experiments)

    #count the number of failed experiments
    results["failures"] = len(single_results) - results["successes"]

    #count the total number of experiments
    results["total"] = len(single_results)

    results[f'avg_std_{top_k_metric}'], _, _ = average_metric(successful_experiments, f"{top_k_metric}_std")
    for top in top_k:
        if top > len(successful_experiments):
            results[f"top_{top}"] = get_top_k_avg_std(successful_experiments)

        #keep only the top n experiments based on the top_k_metric
        exps = sorted(successful_experiments, key=lambda x: x["results"][top_k_metric], reverse=True)[:top]
        results[f"top_{top}"] = get_top_k_avg_std(exps)
    return results

def get_results_from_name(experiments, names):
    results = list(filter(lambda x: x["experiment"] in names, experiments))
    return results

def list_distance(A, B):
    # Assign each unique value of the list to a unicode character
    unique_map = {v:chr(k) for (k,v) in enumerate(set(A+B))}
    
    # Create string versions of the lists
    a = ''.join(list(map(unique_map.get, A)))
    b = ''.join(list(map(unique_map.get, B)))

    return Levenshtein.distance(a, b)

def get_results_from_synthetic(synthetic_experiments:list,
                                experiments:list,
                                top_k_metric:str = "accuracy",
                                top_k:list = [1,3,5,10,15])->dict:
    results = dict()
    successful_experiments = list(filter(lambda x: not x["failed"], experiments))
    successful_syn_experiments = list(filter(lambda x: not x["failed"], synthetic_experiments))
    #top_k.append(min(len(successful_experiments), len(successful_syn_experiments)))
    #print(successful_syn_experiments)
    for top in top_k:
        if top > len(successful_experiments) or top > len(successful_syn_experiments):
            top_exp_syn = sorted(successful_syn_experiments, key=lambda x: x["results"][top_k_metric], reverse=True)
        else:
            top_exp_syn = sorted(successful_syn_experiments, key=lambda x: x["results"][top_k_metric], reverse=True)[:top]
        names = list(map(lambda x: x["experiment"], top_exp_syn))
        exps = get_results_from_name(successful_experiments, names)
        all_best_exps = sorted(successful_experiments, key=lambda x: x["results"][top_k_metric], reverse=True)
        best_exps = all_best_exps[:top]
        val_top_k_results = get_top_k_results(best_exps)
        best_exps_names = list(map(lambda x: x["experiment"], best_exps))
        results[f"top_{top}"] = get_top_k_results(exps)
        results[f"top_{top}"]["distance"] = list_distance(names, best_exps_names)
        results[f"top_{top}"]["accuracy_diff"] = abs(val_top_k_results["accuracy"] - results[f"top_{top}"]["accuracy"])

        #find the keys of exps in all_best_exps
        top_indexes = list(map(lambda x: all_best_exps.index(x), exps))
        results[f"top_{top}"]["indexes_sum"] = sum(top_indexes)

    return results

def summarize_synth_subset_results(subfolder:str,
                                      exp_folder:str,
                                      result_file_name:str,
                                      n_datasets:int,
                                      top_k_metric:str,
                                      subset:int = None)->dict:
    exp_folder_in_subfolder =os.path.join(subfolder, exp_folder)
    single_results = list(map(lambda x: json.load(open(os.path.join(exp_folder_in_subfolder, os.path.join(subset or "",f"exp_{x}",result_file_name)))), range(n_datasets)))
    summarized_results = dict()
    summarized_results['results'] = summarize_results(single_results, top_k_metric, [])   
    summarized_results['failed'] = True if summarized_results["results"]["successes"] == 0 else False
    summarized_results['experiment'] = subfolder.split('/')[-1]
    return summarized_results

def summarize_synth_subfolder_results(subfolder:str,
                                      exp_folder:str,
                                      result_file_name:str,
                                      n_datasets:int,
                                      top_k_metric:str)->dict:
        exp_folder_in_subfolder =os.path.join(subfolder, exp_folder)
        summarized_results = summarize_synth_subset_results(subfolder, exp_folder, result_file_name, n_datasets, top_k_metric)
        #list subfolders in the experiment folder
        # subsets = get_not_exp_subfolders(exp_folder_in_subfolder)
        # #filter out keeping only last folder
        # subsets = list(map(lambda x: x.split("/")[-1], subsets))
        # summarized_results["subsets"] = dict()
        # for subset in subsets:
        #     summarized_results["subsets"][subset] = summarize_synth_subset_results(subfolder, exp_folder, result_file_name, n_datasets, top_k_metric, subset)

        return summarized_results

def summarize_synth_subset_results_on_synth(single_results_syn:list,
                                            top_k_metric:str,
                                            top_k:list,
                                            subset:int = None)->dict:
    if subset is not None:
        subset_results = list(map(lambda x: x["subsets"][str(subset)],single_results_syn))
    else:
        subset_results = single_results_syn

    summarized_results = summarize_results(subset_results, top_k_metric, top_k) 
    summarized_results["avg_std_accuracy"] = get_avg_std(subset_results, "accuracy")
    return summarized_results
    
def summarize_synth_results_on_synth(single_results_syn:list,
                                     top_k_metric:str,
                                     top_k:list,
                                     subsets:list)->dict:

    summarized_results = summarize_synth_subset_results_on_synth(single_results_syn, top_k_metric, top_k)
    summarized_results["subsets"] = dict()
    # for subset in subsets:
    #     summarized_results["subsets"][subset] = summarize_synth_subset_results_on_synth(single_results_syn, top_k_metric, top_k, subset)
    return summarized_results

def summarize_synth_results_on_top_k(single_results_syn:list,
                                     single_results_val,
                                     top_k_metric,
                                     top_k,
                                     subsets)->dict:
    summarized_results = get_results_from_synthetic(single_results_syn, 
                                                                     single_results_val, 
                                                                     top_k_metric, 
                                                                     top_k)

    summarized_results["subsets"] = dict()
    # for k in subsets:
    #     summarized_results["subsets"][str(k)] = get_results_from_synthetic(list(map(lambda x: x["subsets"][str(k)],single_results_syn)), 
    #                                                                                      single_results_val, 
    #                                                                                      top_k_metric, 
    #                                                                                      top_k)
    return summarized_results

def summarize_synth_results(subfolders:list,
                            n_datasets:int,
                            exp_folder:str,
                            opt:Namespace)->dict:
    

    subsets = get_not_exp_subfolders(os.path.join(subfolders[0], exp_folder)) if len(subfolders)> 0 else []
    #filter subset keeping only last folder
    subsets = list(map(lambda x: x.split("/")[-1], subsets))
    single_dataset_results = get_single_dataset_exp_results(subfolders, exp_folder, n_datasets, opt.result_file_name, [])

    # for subfolder in subfolders:
    #     exp_folder_in_subfolder = os.path.join(subfolder, exp_folder)

    #     summarized_results = summarize_synth_subfolder_results(subfolder, exp_folder, opt.result_file_name, n_datasets, opt.top_k_metric)
    #     with open(os.path.join(exp_folder_in_subfolder, opt.result_file_name), 'w') as f:
    #         json.dump(summarized_results, f,ensure_ascii=False,indent=4, cls=NpEncoder)

    summarized_results = dict()
    #print('\n', subfolders[0], '\n',exp_folder, '\n',opt.result_file_name)

    single_results_syn = list(map(lambda x: json.load(open(os.path.join(x, exp_folder,opt.result_file_name))), subfolders))
    
 
    summarized_results["synthetic_dataset"] = summarize_synth_results_on_synth(single_results_syn, opt.top_k_metric, opt.top_k, None)

    single_results_val = list(map(lambda x: json.load(open(os.path.join(x, opt.result_file_name))), subfolders))

    summary_dir = os.path.join(opt.run, exp_folder)
    filename = os.path.join(summary_dir, opt.result_file_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(summarized_results, f,ensure_ascii=False,indent=4, cls=NpEncoder)

    return summarized_results

def get_single_dataset_exp_results(subfolders, exp_folder, n_datasets, file_name, top_ks):
    #print(subfolders)
    results = dict()
    for dataset in range(n_datasets):
        results[f"dataset_{dataset}"] = {}
        results[f"dataset_{dataset}"]["accuracies"] = {}
        for subfolder in subfolders:
            exp_folder_in_subfolder = os.path.join(subfolder, exp_folder)
            content = json.load(open(os.path.join(exp_folder_in_subfolder, f"exp_{dataset}", file_name)))
            if not content['failed']:
                results[f"dataset_{dataset}"]["accuracies"][content['experiment']] = content['results']['accuracy']
        for top_k in top_ks:
            #get the keys associated with the k highest accuracies
            results[f"dataset_{dataset}"][str(top_k)] = {}
            results[f"dataset_{dataset}"][str(top_k)]["experiments"] = sorted(results[f"dataset_{dataset}"]["accuracies"], key=results[f"dataset_{dataset}"]["accuracies"].get, reverse=True)[:top_k]
            #results[f"dataset_{dataset}"][str(top_k)]["accuracy"] = sum([results[f"dataset_{dataset}"]["accuracies"][exp] for exp in results[f"dataset_{dataset}"][str(top_k)]["experiments"]]) / top_k
    return results

def summarize_synth_results_on_top_k_per_single_dataset(syn_results, single_results_val, top_k_metric, top_k, n_datasets):
    results = dict()
    successful_experiments = list(filter(lambda x: not x["failed"], single_results_val))
    #top_k.append(min(len(successful_experiments), len(successful_syn_experiments)))
    #print(successful_syn_experiments)
    for top in top_k:
        results[f"top_{top}"] = {}
        for dataset in range(n_datasets):
            results[f"top_{top}"][f"dataset_{dataset}"] = {}
            names = syn_results[f"dataset_{dataset}"][str(top)]["experiments"]
            exps = get_results_from_name(successful_experiments, names)
        
            all_best_exps = sorted(successful_experiments, key=lambda x: x["results"][top_k_metric], reverse=True)
            best_exps = all_best_exps[:top]
            val_top_k_results = get_top_k_results(best_exps)
            best_exps_names = list(map(lambda x: x["experiment"], best_exps))
            results[f"top_{top}"][f"dataset_{dataset}"] = get_top_k_results(exps)
            results[f"top_{top}"][f"dataset_{dataset}"]["distance"] = list_distance(names, best_exps_names)
            avg_accuracy = sum([ex["results"]["accuracy"] for ex in exps]) / top
            results[f"top_{top}"][f"dataset_{dataset}"]["accuracy_diff"] = abs(val_top_k_results["accuracy"] - avg_accuracy)

            #find the keys of exps in all_best_exps
            top_indexes = list(map(lambda x: all_best_exps.index(x), exps))
            results[f"top_{top}"][f"dataset_{dataset}"]["indexes_sum"] = sum(top_indexes)
    return results

def summarize_synth_test_results(run_folder, test_results_file_name, synth_results, dest, top_ks, n_datasets, synth_datasets_root, dataset_params):
    #read json files
    synth_results = json.load(open(synth_results))
    subfolders = get_exp_subfolders(run_folder)

    #map subfolders to a dict with the name of the folder as key and the accuracy in the json file as value
    subfolders_dict = {}
    subfolders_dict_f2 = {}

    for subfolder in subfolders:
        content = json.load(open(os.path.join(subfolder, test_results_file_name)))
        if not content['failed']:
            subfolders_dict[subfolder.split('/')[-1]] = content["results"]["accuracy"]
            if content["results"]["precision"] != 0 and content["results"]["recall"] != 0:
                subfolders_dict_f2[subfolder.split('/')[-1]] = (((1 + 4) * content["results"]["precision"] * content["results"]["recall"]) / (4 * content["results"]["precision"] + content["results"]["recall"]))
            else:
                subfolders_dict_f2[subfolder.split('/')[-1]] = 0

    #open dataset params json file
    dataset_dict = json.load(open(dataset_params))
    if dataset_dict["examples_per_class"] == 0:
        name =  f'{dataset_dict["model_name"]}_{dataset_dict["temperature"]}_{dataset_dict["generation_mode"]}'
    else:
        name =  f'{dataset_dict["model_name"]}_{dataset_dict["temperature"]}_{dataset_dict["generation_mode"]}_{dataset_dict["examples_per_class"]}'
    model_parameters = get_run_parameters(run_folder)
    df = pd.DataFrame(columns = ['top_k', 'dataset', 'accuracy', 'accuracy_diff', "accuracy_diff_on_top_k_f2", "f2", "f2_diff","f2_diff_on_top_k_acc", 'dataset_index', 'task', 'template', 'prompt_parameters', 'model_name', 'generation_mode', 'examples_per_class', 'temperature', 'seed'])
    for top_k in top_ks:
        #get the highet top_k values in the values of subfolder dicts
        best_top_k = sorted(subfolders_dict, key=subfolders_dict.get, reverse=True)[:top_k]
        best_top_k_f2 = sorted(subfolders_dict_f2, key=subfolders_dict_f2.get, reverse=True)[:top_k]

        best_top_k_selected_on_f2 = sorted(subfolders_dict, key=subfolders_dict_f2.get, reverse=True)[:top_k]
        best_top_k_f2_selected_on_acc = sorted(subfolders_dict_f2, key=subfolders_dict.get, reverse=True)[:top_k]

        #average accuracy of the top_k experiments
        avg_accuracy = sum([subfolders_dict[exp] for exp in best_top_k if exp in subfolders_dict.keys()]) / top_k
        avg_f2 = sum([subfolders_dict_f2[exp] for exp in best_top_k_f2 if exp in subfolders_dict_f2.keys()]) / top_k

        avg_accuracy_on_top_k_f2 = sum([subfolders_dict[exp] for exp in best_top_k if exp in subfolders_dict_f2.keys()]) / top_k
        avg_f2_on_top_k_acc = sum([subfolders_dict_f2[exp] for exp in best_top_k_f2 if exp in subfolders_dict.keys()]) / top_k

        for dataset in range(n_datasets):
            experiments = synth_results['single_dataset_results'][f"dataset_{dataset}"][str(top_k)]["experiments"]
            to_subtract = max(1, top_k -len([exp for exp in experiments if exp not in subfolders_dict.keys()]))
            #get the avg accuracy of the top_k experiments
            avg_accuracy_syn = sum([subfolders_dict[exp] for exp in experiments if exp in subfolders_dict.keys()]) / to_subtract
            avg_f2_syn = sum([subfolders_dict_f2[exp] for exp in experiments if exp in subfolders_dict_f2.keys()]) / to_subtract

            avg_accuracy_syn_on_top_k_f2 = sum([subfolders_dict[exp] for exp in experiments if exp in subfolders_dict_f2.keys()]) / to_subtract
            avg_f2_syn_on_top_k_acc = sum([subfolders_dict_f2[exp] for exp in experiments if exp in subfolders_dict.keys()]) / to_subtract

            acc_diff = abs(avg_accuracy - avg_accuracy_syn)
            f2_diff = abs(avg_f2 - avg_f2_syn)

            acc_diff_on_top_k_f2 = abs(avg_accuracy_on_top_k_f2 - avg_accuracy_syn_on_top_k_f2)
            f2_diff_on_top_k_acc = abs(avg_f2_on_top_k_acc - avg_f2_syn_on_top_k_acc)

            row = {
                "top_k": top_k,
                "dataset": name,
                "accuracy": avg_accuracy_syn,
                "accuracy_diff": acc_diff,
                "accuracy_diff_on_top_k_f2": acc_diff_on_top_k_f2,
                "f2": avg_f2_syn,
                "f2_diff": f2_diff,
                "f2_diff_on_top_k_acc": f2_diff_on_top_k_acc,
                "dataset_index": str(dataset),
                "task":model_parameters["task"],
                "template":model_parameters["template"],
                "prompt_parameters":model_parameters["prompt_parameters"],
                "model_name":model_parameters["model"],
                "generation_mode":model_parameters["generation_mode"],
                "examples_per_class":model_parameters["n_few_shot"],
                "temperature":model_parameters["temperature"],
                "seed":model_parameters["seed"]
            }
            df = pd.concat([df, pd.DataFrame([row.values()], columns=df.columns)], ignore_index=True)
    print(dest)
    df.to_csv(dest, index=False)

def get_winner(df, criterion, query_on_functions = None):
    if query_on_functions is not None:
        df = df.query(query_on_functions)
    #get all the values of the criterion column
    values = df[f"dataset_{criterion}"].unique()
    new_df = pd.DataFrame(columns = ["experiment", criterion, "acc_diff"])
    results = {}
    for value in values:
        results[value] = 0
    for exp in df["experiment"].unique():
        exp_results = {}
        exp_df = df.query(f"experiment == '{exp}'")
        for value in values:
            exp_results[value] = exp_df.query(f"dataset_{criterion} == '{value}'")["accuracy_diff"].mean()
        #get the key with the lower acc diff in exp_results
        best_key = min(exp_results, key=exp_results.get)
        new_row = {"experiment":exp,
                   criterion:best_key,
                   "acc_diff":exp_results[best_key]}
        new_df = pd.concat([new_df, pd.DataFrame(new_row, index=[1])])
        results[best_key] += 1
    return results, new_df

