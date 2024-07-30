import os
import json
from utils.path_utils import get_exp_subfolders
import pandas as pd

best_results_sqli_file = "experiments/task_detect_sqli_extended/template_create_function_readable/prompt_parameters_empty/best_experiment_sqli_f2.json"
best_results_xss_file = "experiments/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty/best_experiment_xss_f2.json"

rq2_xss_file = "plots_rq2_f2.csv"
rq2_sqli_file = "plots_rq2_sqli_f2.csv"



with open(best_results_sqli_file, 'r') as f:
    best_results_sqli = json.load(f)

with open(best_results_xss_file, 'r') as f:
    best_results_xss = json.load(f)

top_ks = [ 1,3,5]

trasfer_results = dict()
for k in top_ks:
    trasfer_results[f"top_{k}"] = dict()
    top_k_results = best_results_xss["validation_results"]["top_k_results_filtered"][f"top_{k}"]["results"]
    folder_basic = "".join(f"""experiments/task_detect_sqli_extended/template_create_function_readable/prompt_parameters_empty/
                model_{best_results_xss['validation_results']['model_name']}/
                generation_mode_{best_results_xss['validation_results']['generation_mode']}/
                n_few_shot_{best_results_xss['validation_results']['examples_per_class']}/
                temperature_{best_results_xss['validation_results']['temperature']}/
                seed_156/run_0/""".split())
    folder_synth = "".join(f"""synthetic_results/task_detect_sqli_extended/
                {top_k_results['folder'].split('task_detect_xss_simple_prompt')[2]}/results.json""".split())
    exps_f2s = {}

    #remove all the spaces and \n from the folder
    exps = list(map(lambda x: x.split('/')[-1],get_exp_subfolders(folder_basic)))
    for exp in exps:
        exp_folder = os.path.join(folder_basic, exp, "results.json")
        exp_results = json.load(open(exp_folder))
        if not exp_results["failed"]:
            precision = exp_results[f"results"]["precision"]
            recall = exp_results[f"results"]["recall"]
            f2 =  (((1 + 4) * precision * recall) / (4 * precision + recall)) if precision!=0 and recall!=0 else 0
            exps_f2s[exp] = f2
    folder = os.path.join(folder_basic, folder_synth)

    exp_f2s_synths = {}
    for exp in exps:
        if exp not in exps_f2s.keys():
            continue
        exp_folder = os.path.join(folder_basic, exp, folder_synth)
        exp_results = json.load(open(exp_folder))
        precision = exp_results[f"results"]["precision"]
        recall = exp_results[f"results"]["recall"]
        f2 = (((1 + 4) * precision * recall) / (4 * precision + recall)) if precision!=0 and recall!=0 else 0
        exp_f2s_synths[exp] = f2

    top_ks_exps = sorted(exp_f2s_synths, key=exp_f2s_synths.get, reverse=True)[:k]
    f2 = 0
    top_ks_f2s = sum([exps_f2s[exp] for exp in top_ks_exps])/len(top_ks_exps) if len(top_ks_exps) > 0 else 0
    top_ks_f2s_synth = sum([exp_f2s_synths[exp] for exp in top_ks_exps])/len(top_ks_exps) if len(top_ks_exps) > 0 else 0
    for exp in top_ks_exps:
        exp_r = json.load(open(os.path.join(folder_basic, exp, "test_results.json")))
        if exp_r["results"]["precision"] != 0 and  exp_r["results"]["recall"] != 0:
            f2 += (((1 + 4) * exp_r["results"]["precision"] * exp_r["results"]["recall"]) / (4 * exp_r["results"]["precision"] + exp_r["results"]["recall"]))
    
    f2 = f2/len(top_ks_exps)
    #original_acc = best_results_xss["test_results_synth"]["top_k_metrics"][f"top_{k}_accuracy"]
    original_f2 = best_results_xss["test_results_synth"]["top_k_metrics"][f"top_{k}_f2"]
    exp_results = json.load(open(exp_folder))
    if not exp_results["failed"]:
        precision = exp_results[f"results"]["precision"]
        recall = exp_results[f"results"]["recall"]
        f2 =  (((1 + 4) * precision * recall) / (4 * precision + recall)) if precision!=0 and recall!=0 else 0
        exps_f2s[exp] = f2
    folder = os.path.join(folder_basic, folder_synth)

    exp_f2s_synths = {}
    for exp in exps:
        if exp not in exps_f2s.keys():
            continue
        exp_folder = os.path.join(folder_basic, exp, folder_synth)
        exp_results = json.load(open(exp_folder))
        precision = exp_results[f"results"]["precision"]
        recall = exp_results[f"results"]["recall"]
        f2 = (((1 + 4) * precision * recall) / (4 * precision + recall)) if precision!=0 and recall!=0 else 0
        exp_f2s_synths[exp] = f2

    top_ks_exps = sorted(exp_f2s_synths, key=exp_f2s_synths.get, reverse=True)[:k]
    f2 = 0
    top_ks_f2s = sum([exps_f2s[exp] for exp in top_ks_exps])/len(top_ks_exps) if len(top_ks_exps) > 0 else 0
    top_ks_f2s_synth = sum([exp_f2s_synths[exp] for exp in top_ks_exps])/len(top_ks_exps) if len(top_ks_exps) > 0 else 0
    for exp in top_ks_exps:
        exp_r = json.load(open(os.path.join(folder_basic, exp, "test_results.json")))
        if exp_r["results"]["precision"] != 0 and  exp_r["results"]["recall"] != 0:
            f2 += (((1 + 4) * exp_r["results"]["precision"] * exp_r["results"]["recall"]) / (4 * exp_r["results"]["precision"] + exp_r["results"]["recall"]))
    
    f2 = f2/len(top_ks_exps)
    #original_acc = best_results_xss["test_results_synth"]["top_k_metrics"][f"top_{k}_accuracy"]
    original_f2 = best_results_xss["test_results_synth"]["top_k_metrics"][f"top_{k}_f2"]
    from_xss_to_sqli = {
        "model":best_results_xss['validation_results']['model_name'],
        "generation_mode":best_results_xss['validation_results']['generation_mode'],
        "examples_per_class":best_results_xss['validation_results']['examples_per_class'],
        "temperature":best_results_xss['validation_results']['temperature'],
        "folder_synth":folder_synth,
        #"transfer_acc":acc,
        #"original_acc":original_acc,
        "transfer_f2":f2,
        "original_f2":original_f2

    }

    



    top_k_results = best_results_sqli["validation_results"]["top_k_results_filtered"][f"top_{k}"]["results"]
    folder_basic = "".join(f"""experiments/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty/
                model_{best_results_sqli['validation_results']['model_name']}/
                generation_mode_{best_results_sqli['validation_results']['generation_mode']}/
                n_few_shot_{best_results_sqli['validation_results']['examples_per_class']}/
                temperature_{best_results_sqli['validation_results']['temperature']}/
                seed_156/run_0/""".split())
    folder_synth = "".join(f"""synthetic_results/task_detect_xss_simple_prompt/
                {top_k_results['folder'].split('task_detect_sqli_extended')[2]}/results.json""".split())
    exps_f2s = {}

    #remove all the spaces and \n from the folder
    exps = list(map(lambda x: x.split('/')[-1],get_exp_subfolders(folder_basic)))
    for exp in exps:
        exp_folder = os.path.join(folder_basic, exp, "results.json")
        exp_results = json.load(open(exp_folder))
        if not exp_results["failed"]:
            precision = exp_results[f"results"]["precision"]
            recall = exp_results[f"results"]["recall"]
            f2 =  (((1 + 4) * precision * recall) / (4 * precision + recall)) if precision!=0 and recall!=0 else 0
            exps_f2s[exp] = f2
    folder = os.path.join(folder_basic, folder_synth)

    exp_f2s_synths = {}
    for exp in exps:
        if exp not in exps_f2s.keys():
            continue
        exp_folder = os.path.join(folder_basic, exp, folder_synth)
        exp_results = json.load(open(exp_folder))
        precision = exp_results[f"results"]["precision"]
        recall = exp_results[f"results"]["recall"]
        f2 = (((1 + 4) * precision * recall) / (4 * precision + recall)) if precision!=0 and recall!=0 else 0
        exp_f2s_synths[exp] = f2

    top_ks_exps = sorted(exp_f2s_synths, key=exp_f2s_synths.get, reverse=True)[:k]
    f2 = 0
    top_ks_f2s = sum([exps_f2s[exp] for exp in top_ks_exps])/len(top_ks_exps) if len(top_ks_exps) > 0 else 0
    top_ks_f2s_synth = sum([exp_f2s_synths[exp] for exp in top_ks_exps])/len(top_ks_exps) if len(top_ks_exps) > 0 else 0
    for exp in top_ks_exps:
        exp_r = json.load(open(os.path.join(folder_basic, exp, "test_results.json")))
        if exp_r["results"]["precision"] != 0 and  exp_r["results"]["recall"] != 0:
            f2 += (((1 + 4) * exp_r["results"]["precision"] * exp_r["results"]["recall"]) / (4 * exp_r["results"]["precision"] + exp_r["results"]["recall"]))
    
    f2 = f2/len(top_ks_exps)
    #original_acc = best_results_xss["test_results_synth"]["top_k_metrics"][f"top_{k}_accuracy"]
    original_f2 = best_results_sqli["test_results_synth"]["top_k_metrics"][f"top_{k}_f2"]

    sqli_to_xss = {
        "model":best_results_sqli['validation_results']['model_name'],
        "generation_mode":best_results_sqli['validation_results']['generation_mode'],
        "examples_per_class":best_results_sqli['validation_results']['examples_per_class'],
        "temperature":best_results_sqli['validation_results']['temperature'],
        "folder_synth":folder_synth,
        #"transfer_acc":acc,
        #"original_acc":original_acc,
        "transfer_f2":f2,
        "original_f2":original_f2

    }
    

    trasfer_results[f"top_{k}"]["xss_to_sqli"] = from_xss_to_sqli
    trasfer_results[f"top_{k}"]["sqli_to_xss"] = sqli_to_xss

    #open rq2_xss_file and rq2_sqli_file
    df_xss = pd.read_csv(rq2_xss_file)
    df_sqli = pd.read_csv(rq2_sqli_file)

    avg_results = {}

    for k in top_ks:
        avg_results[f"top_{k}"] = dict()
        avg_results[f"top_{k}"]["xss"] = df_xss[f"top_{k}_f2"].mean()
        avg_results[f"top_{k}"]["sqli"] = df_sqli[f"top_{k}_f2"].mean()

    trasfer_results["average"] = avg_results




#save trasfer_results in json file named rq4.json
with open("rq4_f2.json", 'w') as f:
    json.dump(trasfer_results, f, indent=4, ensure_ascii=False)



