from utils.utils import NpEncoder, init_argument_parser
import pandas as pd
import os
import json
import numpy as np

def select_best_experiment(opt):
    df = pd.read_csv(opt.summary_file)
    #get the row with maximum accuracy
    df_best = df.loc[df['accuracy'].idxmax()]

    validation_results  = {
        "folder": str(df_best["folder"]),
        "model_name": str(df_best["model_name"]),
        "temperature": int(df_best["temperature"]),
        "generation_mode": str(df_best["generation_mode"]),
        "examples_per_class": int(df_best["examples_per_class"]),
        "accuracy": float(df_best["accuracy"]),
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
                    avg_acc = sum([single_dataset_v["accuracies"][exp] for exp in exps])/len(exps)
                    single_accs.append(avg_acc)
                #get the std of the accuracies
                acc_stdr = np.std(single_accs)
                all_k_results.append({
                    "folder": folder,
                    "synth_acc":results['top_k_metrics'][f"top_{k}"]["accuracy"],
                    "val_acc":results["validation_dataset"][f"top_{k}"]["accuracy"],
                    "acc_diff":results['top_k_metrics'][f"top_{k}"]["accuracy_diff"],
                    "acc_std":acc_stdr
                    }) 
                
        #get lowest acc_diff in all_k_results
        best_k_result = [v for k,v in enumerate(all_k_results) if v["acc_diff"] == min([x["acc_diff"] for x in all_k_results])]
        #keep the one with the lowest std
        best_k_result = [v for k,v in enumerate(best_k_result) if v["acc_std"] == min([x["acc_std"] for x in best_k_result])][0]
    
                
        folder_data = os.path.join(opt.synthetic_dataset_folder, best_k_result["folder"].split(f"{opt.synthetic_results_folder}/")[-1], "run_0", opt.parameters_file_name)
        parameters = json.load(open(folder_data))
        top_k_results[f"top_{k}"] = {
            "results": best_k_result,
            "model_name": parameters["model_name"],
            "temperature": parameters["temperature"],
            "generation_mode": parameters["generation_mode"],
            "examples_per_class": parameters["examples_per_class"],
        }
    validation_results["top_k_results"] = top_k_results

    test_results_dict = {}
    test_results = json.load(open(os.path.join(validation_results["folder"], opt.test_results_file_name)))
    test_results_dict["accuracy"] = test_results["accuracy"]
    test_results_dict["top_k_metrics"] = {}
    for k in opt.top_k:
        test_results = pd.read_csv(os.path.join(validation_results["top_k_results"][f"top_{k}"]["results"]["folder"], f"{opt.test_results_file_name.split('.')[0]}.csv"))
        #filter the results keeping only the values with top_k = k
        test_results = test_results[test_results["top_k"] == k]
        #avg the accuracies
        test_results_dict["top_k_metrics"][f"top_{k}_accuracy"] = test_results["accuracy"].mean()
    
    final_results = {
        "validation_results": validation_results,
        "test_results": test_results_dict
    }
    #save the results
    with open(opt.output_file, 'w') as f:
        json.dump(final_results, f, indent=4,ensure_ascii=False, cls=NpEncoder)



def add_parse_arguments(parser):

    parser.add_argument('--summary_file', type=str, default='experiments/task_detect_xss_simple_prompt/experiments_summary.csv', help='summary file')
    parser.add_argument('--synthetic_results_folder', type=str, default='synthetic_results', help='folder with synthetic results')
    parser.add_argument('--synthetic_dataset_folder', type=str, default='data/synthetic_datasets', help='folder with synthetic datasets')

    parser.add_argument('--parameters_file_name', type=str, default='parameters.json', help='name of the file containing the parameters')
    parser.add_argument('--results_file_name', type=str, default='results.json', help='name of the file containing the results')
    parser.add_argument('--test_results_file_name', type=str, default='test_results.json', help='name of the file containing the test_results')

    parser.add_argument('--top_k', type=int, action='append', help='top_k value to be considered for the top_k_metric, you can append more than one')

    parser.add_argument('--output_file', type=str, default='experiments/task_detect_xss_simple_prompt/template_create_function_readable/prompt_parameters_empty/best_experiment_xss.json', help='name of the file containing the summary of the experiments')

    return parser

def main():
    opt = init_argument_parser(add_parse_arguments)
    select_best_experiment(opt)

if __name__ == '__main__':
    main()