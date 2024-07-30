from scipy.stats import mannwhitneyu, wilcoxon
import json
import seaborn as sns
import pandas as pd
from functools import reduce
import os
import json

def a12_unpaired(lst1, lst2):
    more = same = 0.0
    for x in lst1:
        for y in lst2:
            if x == y: same +=1
            elif x > y: more +=1
    return (more + 0.5*same)/(len(lst1)*len(lst2))

def interpret_effect_size(effect_size_score):
    if effect_size_score < 0.147:
        return "negligible"
    elif effect_size_score < 0.33:
        return "small"
    elif effect_size_score < 0.474:
        return "medium"
    else:
       return "large"

accuracies_file = "accuracies_sqli.csv"
rq2_file = "plots_rq2_sqli.csv"
rq2_file_f2 = "plots_rq2_sqli_f2.csv"

folder_to_save = 'statistical/sqli'
os.makedirs(folder_to_save, exist_ok=True)

accuracies = pd.read_csv(accuracies_file)

#create 2 lists, one with all the accuracies for the RAG rows and one for the others
rag_accuracies = accuracies[accuracies["use_rag"] == True]["accuracy"].values.tolist()
no_rag_accuracies = accuracies[accuracies["use_rag"] == False]["accuracy"].values.tolist()

stat = mannwhitneyu(rag_accuracies, no_rag_accuracies)
pvalue = stat.pvalue

a12 = a12_unpaired(rag_accuracies, no_rag_accuracies)

effect_size_score = 2*abs(a12 -0.5)
effect_size = interpret_effect_size(effect_size_score)


rag_accuracies_f2 = accuracies[accuracies["use_rag"] == True]["f2"].values.tolist()
no_rag_accuracies_f2 = accuracies[accuracies["use_rag"] == False]["f2"].values.tolist()

stat_f2 = mannwhitneyu(rag_accuracies_f2, no_rag_accuracies_f2)
pvalue_f2 = stat_f2.pvalue

a12_f2 = a12_unpaired(rag_accuracies_f2, no_rag_accuracies_f2)

effect_size_score_f2 = 2*abs(a12_f2 -0.5)
effect_size_f2 = interpret_effect_size(effect_size_score_f2)


with(open(f"{folder_to_save}/rq1.json", "w")) as f:
    json.dump({"acc":{"pvalue": pvalue, 
    "a12": a12,
    "effect_size_score":effect_size_score,
    "effect_size":effect_size},
    "f2":{"pvalue": pvalue_f2, 
    "a12": a12_f2,
    "effect_size_score":effect_size_score_f2,
    "effect_size":effect_size_f2}}, f,ensure_ascii=False,indent=4)

#rq2
rq2 = pd.read_csv(rq2_file)
rq2_f2 = pd.read_csv(rq2_file_f2)

top_ks = [1,3,5]
rq2_scores = {}

for top_k in top_ks:
    rq2_scores[top_k] = dict()
    top_k_accuracies = rq2[f"top_{top_k}_acc"].values.tolist()
    avg_accuracies = rq2[f"avg_accuracy"].values.tolist()
    stat = wilcoxon(top_k_accuracies, avg_accuracies)
    pvalue = stat.pvalue
    print(pvalue)

    a12 = a12_unpaired(top_k_accuracies, avg_accuracies)

    effect_size_score = 2*abs(a12 -0.5)
    effect_size = interpret_effect_size(effect_size_score)
    rq2_scores[top_k]["accuracy"] = {"pvalue": pvalue,
    "a12": a12,
    "effect_size_score":effect_size_score,
    "effect_size":effect_size}


    top_k_accuracies = rq2_f2[f"top_{top_k}_f2"].values.tolist()
    avg_accuracies = rq2_f2[f"avg_f2"].values.tolist()
    stat = wilcoxon(top_k_accuracies, avg_accuracies)
    pvalue = stat.pvalue
    print(pvalue)

    a12 = a12_unpaired(top_k_accuracies, avg_accuracies)

    effect_size_score = 2*abs(a12 -0.5)
    effect_size = interpret_effect_size(effect_size_score)
    rq2_scores[top_k]["f2"] = {"pvalue": pvalue,
    "a12": a12,
    "effect_size_score":effect_size_score,
    "effect_size":effect_size}
with(open(f"{folder_to_save}/rq2.json", "w")) as f:
    json.dump(rq2_scores, f,ensure_ascii=False,indent=4)
