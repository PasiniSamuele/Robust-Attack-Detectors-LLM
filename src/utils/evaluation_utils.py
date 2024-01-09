import re
import pandas as pd
import seaborn as sn
import os 
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

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

def summarize_results(single_results:list,
                      top_n_metric:str = "accuracy",
                      top_n:list = [1,3,5,10,15])->dict:
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
    for top in top_n:
        if top > len(successful_experiments):
            continue
        top_n_results = dict()
        #keep only the top n experiments based on the top_n_metric
        exps = sorted(successful_experiments, key=lambda x: x["results"][top_n_metric], reverse=True)[:top]

        #write the top_n experiments names
        top_n_results["experiments"] = list(map(lambda x: x["experiment"], exps))
        #calculate avg, std and var for each metric
        top_n_results["accuracy"], top_n_results["accuracy_std"], top_n_results["accuracy_var"] = average_metric(exps, "accuracy")
        top_n_results["precision"], top_n_results["precision_std"], top_n_results["precision_var"] = average_metric(exps, "precision")
        top_n_results["recall"], top_n_results["recall_std"], top_n_results["recall_var"] = average_metric(exps, "recall")
        top_n_results["f1"], top_n_results["f1_std"], top_n_results["f1_var"] = average_metric(exps, "f1")
        results[f"top_{top}"] = top_n_results
    return results