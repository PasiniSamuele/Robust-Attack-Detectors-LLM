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

accuracies_file = "accuracies.csv"
rq2_file = "plots_rq2.csv"

folder_to_save = 'statistical/xss'
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


with(open(f"{folder_to_save}/rq1.json", "w")) as f:
    json.dump({"pvalue": pvalue, 
    "a12": a12,
    "effect_size_score":effect_size_score,
    "effect_size":effect_size}, f)

#rq2
rq2 = pd.read_csv(rq2_file)
top_ks = [1,3,5]
rq2_scores = {}

for top_k in top_ks:
    top_k_accuracies = rq2[f"top_{top_k}_acc"].values.tolist()
    avg_accuracies = rq2[f"avg_accuracy"].values.tolist()
    stat = wilcoxon(top_k_accuracies, avg_accuracies)
    pvalue = stat.pvalue
    print(pvalue)

    a12 = a12_unpaired(top_k_accuracies, avg_accuracies)

    effect_size_score = 2*abs(a12 -0.5)
    effect_size = interpret_effect_size(effect_size_score)
    rq2_scores[top_k] = {"pvalue": pvalue,
    "a12": a12,
    "effect_size_score":effect_size_score,
    "effect_size":effect_size}
with(open(f"{folder_to_save}/rq2.json", "w")) as f:
    json.dump(rq2_scores, f)


