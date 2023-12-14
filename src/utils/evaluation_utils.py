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

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

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
    avg =  sum(map(lambda x: x["results"][metric], single_results)) / len(single_results)
    std = np.std(list(map(lambda x: x["results"][metric], single_results)))
    var = np.var(list(map(lambda x: x["results"][metric], single_results)))
    return avg, std, var

def summarize_results(single_results:list)->dict:
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

    return results