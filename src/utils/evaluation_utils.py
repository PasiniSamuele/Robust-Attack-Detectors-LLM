import re
import pandas as pd
import seaborn as sn
import os 
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import Levenshtein

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
                      top_k:list = [1,3,5,10,15])->dict:
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
    results["accuracy"], results["accuracy_std"], results["accuracy_var"] = average_metric(successful_experiments, "accuracy")
    results["precision"], results["precision_std"], results["precision_var"] = average_metric(successful_experiments, "precision")
    results["recall"], results["recall_std"], results["recall_var"] = average_metric(successful_experiments, "recall")
    results["f1"], results["f1_std"], results["f1_var"] = average_metric(successful_experiments, "f1")
    for top in top_k:
        if top > len(successful_experiments):
            continue
        #keep only the top n experiments based on the top_k_metric
        exps = sorted(successful_experiments, key=lambda x: x["results"][top_k_metric], reverse=True)[:top]
        results[f"top_{top}"] = get_top_k_results(exps)
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
    top_k.append(min(len(successful_experiments), len(successful_syn_experiments)))
    #print(successful_syn_experiments)
    for top in top_k:
        if top > len(successful_experiments) or top > len(successful_syn_experiments):
            continue
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