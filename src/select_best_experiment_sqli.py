from utils.utils import NpEncoder, init_argument_parser
import pandas as pd
import os
import json
import numpy as np

def select_best_experiment(opt):
    df = pd.concat([pd.read_csv(opt.summary_file), pd.read_csv(opt.summary_file_sap)])

    #reset index
    df.reset_index(drop=False, inplace=True)

    #drop llama3 with temperature 0.0
    df = df.drop(df[(df["model_name"] == "llama3-70b-instruct") & (df["temperature"] == 0.0)].index)

    #drop gcp−chat−bison−001 with temperature 0.0 and 0.5
    df = df.drop(df[(df["model_name"] == "gcp-chat-bison-001") & (df["temperature"] == 0.0)].index)
    df = df.drop(df[(df["model_name"] == "gcp-chat-bison-001") & (df["temperature"] == 0.5)].index)

    #drop anthropic−claude−3−sonnet with temperature 0.0
    df = df.drop(df[(df["model_name"] == "anthropic-claude-3-sonnet") & (df["temperature"] == 0.0)].index)

    #drop mixtral−8x7b−instruct−v01 with temperature 0.0
    df = df.drop(df[(df["model_name"] == "mixtral-8x7b-instruct-v01") & (df["temperature"] == 0.0)].index)

    #drop anthropic-claude-3-haiku
    df = df.drop(df[(df["model_name"] == "anthropic-claude-3-haiku")].index)

    #drop gpt-3.5-turbo-0125
    df = df.drop(df[(df["model_name"] == "gpt-3.5-turbo-0125")].index)


    #reset index
    df.reset_index(drop=False, inplace=True)
    #get the row with maximum accuracy
    df_best = df.loc[df['accuracy'].idxmax()]
    top_k_results_val_set = {}
    results_file = os.path.join(df_best["folder"], opt.results_file_name)
    results = json.load(open(results_file))
    for k in opt.top_k:
        top_k_results_val_set[k] = {
            "exp" : results[f"top_{k}"]["experiments"],
            "accuracy" : results[f"top_{k}"]["accuracy"],

        }

    validation_results  = {
        "folder": str(df_best["folder"]),
        "model_name": str(df_best["model_name"]),
        "temperature": float(df_best["temperature"]),
        "generation_mode": str(df_best["generation_mode"]),
        "examples_per_class": int(df_best["examples_per_class"]),
        "accuracy": float(df_best["accuracy"]),
        "top_k_results_val_set" : top_k_results_val_set
    }

    #get all folders recursively into df_best folder
    synthetic_folder = []
    for root, dirs, files in os.walk(os.path.join(str(df_best["folder"]),opt.synthetic_results_folder)):
        for file in files:
            if file == opt.results_file_name:
                synthetic_folder.append(root)
    
    top_k_results = {}
    for k in opt.top_k:
        all_k_results = []
        #create a dict that, as key has a root in synthetic folder, and as value the content of the results file
        for folder in synthetic_folder:
            with open(os.path.join(folder, opt.results_file_name)) as f:
                results = json.load(f)
                single_accs = []
                for single_dataset_k, single_dataset_v in results["single_dataset_results"].items():
                    exps = single_dataset_v[str(k)]["experiments"]
                    avg_acc = sum([single_dataset_v["accuracies"][exp] for exp in exps])/len(exps) if len(exps) > 0 else 0
                    single_accs.append(avg_acc)
                #get the std of the accuracies
                acc_stdr = np.std(single_accs)
                all_k_results.append({
                    "folder": folder,
                    "synth_acc":results['top_k_metrics'][f"top_{k}"]["accuracy"],
                    "val_acc":results["validation_dataset"][f"top_{k}"]["accuracy"],
                    "acc_diff":results['top_k_metrics'][f"top_{k}"]["accuracy_diff"],
                    "exps":results['top_k_metrics'][f"top_{k}"]["experiments"],
                    "acc_std":acc_stdr
                    }) 
                
        #get lowest acc_diff in all_k_results
        best_k_result = [v for k,v in enumerate(all_k_results) if v["acc_diff"] == min([x["acc_diff"] for x in all_k_results])]
        #keep the one with the lowest std
        best_k_result = [v for k,v in enumerate(best_k_result) if v["acc_std"] == min([x["acc_std"] for x in best_k_result])][0]
    
                
        folder_data = os.path.join(opt.synthetic_dataset_folder, best_k_result["folder"].split(f"{opt.synthetic_results_folder}/")[-1], "run_0", opt.parameters_file_name)
        try:
            parameters = json.load(open(folder_data))
        except:
            folder_data = os.path.join(opt.synthetic_dataset_folder_sap, best_k_result["folder"].split(f"{opt.synthetic_results_folder}/")[-1], "run_0", opt.parameters_file_name)
            parameters = json.load(open(folder_data))

        top_k_results[f"top_{k}"] = {
            "results": best_k_result,
            "model_name": parameters["model_name"],
            "temperature": parameters["temperature"],
            "generation_mode": parameters["generation_mode"],
            "examples_per_class": parameters["examples_per_class"],
        }
    validation_results["top_k_results"] = top_k_results




    top_k_results_filtered = {}
    valid_datasets = []
    for root, dirs, files in os.walk(opt.cross_task_folder):
        for file in files:
            if file == opt.results_file_name:
                valid_datasets.append(root)
    #keep only the valid_datasets that, in the last split of /, contain seed_
    valid_datasets = list(filter(lambda x:  "seed" in x.split("/")[-1].split("_")[0], valid_datasets))
    valid_datasets = list(map(lambda x: x.split("template_create_synthetic_dataset")[1], valid_datasets))
    synthetic_folder = list(filter(lambda x: x.split("template_create_synthetic_dataset")[1] in valid_datasets, synthetic_folder))
    for k in opt.top_k:
        all_k_results = []
        #create a dict that, as key has a root in synthetic folder, and as value the content of the results file
        for folder in synthetic_folder:
            with open(os.path.join(folder, opt.results_file_name)) as f:
                results = json.load(f)
                single_accs = []
                for single_dataset_k, single_dataset_v in results["single_dataset_results"].items():
                    exps = single_dataset_v[str(k)]["experiments"]
                    avg_acc = sum([single_dataset_v["accuracies"][exp] for exp in exps])/len(exps) if len(exps) > 0 else 0
                    single_accs.append(avg_acc)
                #get the std of the accuracies
                acc_stdr = np.std(single_accs)
                all_k_results.append({
                    "folder": folder,
                    "synth_acc":results['top_k_metrics'][f"top_{k}"]["accuracy"],
                    "val_acc":results["validation_dataset"][f"top_{k}"]["accuracy"],
                    "acc_diff":results['top_k_metrics'][f"top_{k}"]["accuracy_diff"],
                    "exps":results['top_k_metrics'][f"top_{k}"]["experiments"],
                    "acc_std":acc_stdr
                    }) 
                
        #get lowest acc_diff in all_k_results
        best_k_result = [v for k,v in enumerate(all_k_results) if v["acc_diff"] == min([x["acc_diff"] for x in all_k_results])]
        #keep the one with the lowest std
        best_k_result = [v for k,v in enumerate(best_k_result) if v["acc_std"] == min([x["acc_std"] for x in best_k_result])][0]
    
                
        folder_data = os.path.join(opt.synthetic_dataset_folder, best_k_result["folder"].split(f"{opt.synthetic_results_folder}/")[-1], "run_0", opt.parameters_file_name)
        try:
            parameters = json.load(open(folder_data))
        except:
            folder_data = os.path.join(opt.synthetic_dataset_folder_sap, best_k_result["folder"].split(f"{opt.synthetic_results_folder}/")[-1], "run_0", opt.parameters_file_name)
            parameters = json.load(open(folder_data))

        top_k_results_filtered[f"top_{k}"] = {
            "results": best_k_result,
            "model_name": parameters["model_name"],
            "temperature": parameters["temperature"],
            "generation_mode": parameters["generation_mode"],
            "examples_per_class": parameters["examples_per_class"],
        }
    validation_results["top_k_results_filtered"] = top_k_results_filtered

    test_results_dict = {}
    test_results = json.load(open(os.path.join(validation_results["folder"], opt.test_results_file_name)))
    test_results_dict["accuracy"] = test_results["accuracy"]
    test_results_dict["top_k_metrics"] = {}
    for k in opt.top_k:
        acc = 0
        experiments = validation_results["top_k_results"][f"top_{k}"]["results"]["exps"]
        for exp in experiments:
            exp_r = json.load(open(os.path.join(validation_results["folder"], exp,  opt.test_results_file_name)))
            acc += exp_r["results"]["accuracy"]
        acc = acc/len(experiments)

        #avg the accuracies
        test_results_dict["top_k_metrics"][f"top_{k}_accuracy"] = acc
    test_results_val_dict = {}
    for k in opt.top_k:
        acc = 0
        experiments = validation_results["top_k_results_val_set"][k]["exp"]
        for exp in experiments:
            exp_r = json.load(open(os.path.join(validation_results["folder"], exp,  opt.test_results_file_name)))
            acc += exp_r["results"]["accuracy"]
        acc = acc/len(experiments)
        test_results_val_dict[f"top_{k}_accuracy"] = acc
    
    final_results = {
        "validation_results": validation_results,
        "test_results_synth": test_results_dict,
        "test_results_val": test_results_val_dict

    }
    #save the results
    with open(opt.output_file, 'w') as f:
        json.dump(final_results, f, indent=4,ensure_ascii=False, cls=NpEncoder)



def add_parse_arguments(parser):

    parser.add_argument('--summary_file', type=str, default='experiments/task_detect_sqli_extended/template_create_function_readable/experiments_summary.csv', help='summary file')
    parser.add_argument('--summary_file_sap', type=str, default='new_experiments_sap_sqli/task_detect_sqli_extended/template_create_function_readable/experiments_summary.csv', help='summary file')

    parser.add_argument('--synthetic_results_folder', type=str, default='synthetic_results', help='folder with synthetic results')
    parser.add_argument('--synthetic_dataset_folder', type=str, default='data/synthetic_datasets', help='folder with synthetic datasets')
    parser.add_argument('--synthetic_dataset_folder_sap', type=str, default='data/synthetic_datasets_sap_sqli', help='folder with synthetic datasets')
    parser.add_argument('--cross_task_folder', type=str, default='experiments/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty/model_gpt-4-0125-preview/generation_mode_rag/temperature_0.0/seed_156/run_0', help='folder with a run of the cross task')

    parser.add_argument('--parameters_file_name', type=str, default='parameters.json', help='name of the file containing the parameters')
    parser.add_argument('--results_file_name', type=str, default='results.json', help='name of the file containing the results')
    parser.add_argument('--test_results_file_name', type=str, default='test_results.json', help='name of the file containing the test_results')

    parser.add_argument('--top_k', type=int, action='append', help='top_k value to be considered for the top_k_metric, you can append more than one')

    parser.add_argument('--output_file', type=str, default='experiments/task_detect_sqli_extended/template_create_function_readable/prompt_parameters_empty/best_experiment_sqli.json', help='name of the file containing the summary of the experiments')

    return parser

def main():
    opt = init_argument_parser(add_parse_arguments)
    select_best_experiment(opt)

if __name__ == '__main__':
    main()