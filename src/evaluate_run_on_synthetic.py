from utils.utils import init_argument_parser, NpEncoder
from argparse import Namespace
import json
from utils.path_utils import get_experiment_folder, from_file_to_name, get_subfolders
import os
from evaluate_run import evaluate_run
from utils.evaluation_utils import summarize_results, get_results_from_synthetic

def evaluate_synth_run(opt):
    #open parameters file
    with open(f"{opt.dataset_folder}/{opt.parameters_file_name}") as f:
        parameters = json.load(f)
    evaluation_namespace = Namespace(**vars(opt))
    n_datasets = parameters["experiments"]
    exp_folder = get_experiment_folder(opt.evaluation_folder,
                                        from_file_to_name(parameters["task"]),
                                            from_file_to_name(parameters["template"]),
                                            from_file_to_name(parameters["prompt_parameters"]),
                                            parameters["model_name"],
                                            parameters["generation_mode"],
                                            parameters["examples_per_class"],
                                            parameters["temperature"],
                                            parameters["seed"])
    for i in range(n_datasets):
        evaluation_namespace.data = f"{opt.dataset_folder}/exp_{i}.csv"
        exp_folder_single_dataset = os.path.join(exp_folder, f"exp_{i}")
        evaluation_namespace.result_file_name = os.path.join(exp_folder_single_dataset, opt.result_file_name)
        evaluation_namespace.create_confusion_matrix = False
        evaluation_namespace.summarize_results = False
        evaluate_run(evaluation_namespace)
    subfolders = get_subfolders(opt.run)
    for i, subfolder in enumerate(subfolders):
        exp_folder_in_subfolder = os.path.join(subfolder, exp_folder)
        single_results = list(map(lambda x: json.load(open(os.path.join(exp_folder_in_subfolder, os.path.join(f"exp_{x}",opt.result_file_name)))), range(n_datasets)))
        summarized_results = dict()
        summarized_results['results'] = summarize_results(single_results, opt.top_k_metric, [])   
        summarized_results['failed'] = True if summarized_results["results"]["successes"] == 0 else False
        summarized_results['experiment'] = subfolder.split('/')[-1]
        with open(os.path.join(exp_folder_in_subfolder, opt.result_file_name), 'w') as f:
            json.dump(summarized_results, f,ensure_ascii=False,indent=4, cls=NpEncoder)

    summarized_results = dict()

    single_results_syn = list(map(lambda x: json.load(open(os.path.join(x, os.path.join(exp_folder,opt.result_file_name)))), subfolders))
    summarized_results_syn = summarize_results(single_results_syn, opt.top_k_metric, opt.top_k)            
    summarized_results["synthetic_dataset"] = summarized_results_syn

    single_results_val = list(map(lambda x: json.load(open(os.path.join(x, opt.result_file_name))), subfolders))
    summarized_results_val = summarize_results(single_results_val, opt.top_k_metric, opt.top_k)  
    summarized_results["validation_dataset"] = summarized_results_val

    summarized_results["top_k_metrics"] = get_results_from_synthetic(single_results_syn, 
                                                                     single_results_val, 
                                                                     opt.top_k_metric, 
                                                                     opt.top_k)
    
    summary_dir = os.path.join(opt.run, exp_folder)
    filename = os.path.join(summary_dir, opt.result_file_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(summarized_results, f,ensure_ascii=False,indent=4, cls=NpEncoder)

def add_parse_arguments(parser):
    #run parameters
    parser.add_argument('--run', type=str, required=True, help='Run to be evaluated')
    parser.add_argument('--dataset_folder', type=str, default='data/synthetic_datasets/run_0', help='synthetic dataset folder')
    parser.add_argument('--parameters_file_name', type=str, default='parameters.json', help='parameters file name of the synthetic dataset')
    parser.add_argument('--function_name', type=str, default='detect_xss', help='name of the generated function to be executed for evaluation')
    parser.add_argument('--evaluation_folder', type=str, default='synthetic_results', help='synthetic dataset folder')
   

    #evaluation parameters
    parser.add_argument('--isolated_execution', type=bool, default=False, help='if true, the evaluation will be executed in a separate docker environment')
    parser.add_argument('--result_file_name', type=str, default='results.json', help='name of the results file')
    parser.add_argument('--top_k_metric', type=str, default='accuracy', help='metric used to select the best experiments in the run')
    parser.add_argument('--top_k', type=int, action='append', help='top_k value to be considered for the top_k_metric, you can append more than one')

    return parser

def main():
    opt = init_argument_parser(add_parse_arguments)
    evaluate_synth_run(opt)

if __name__ == '__main__':
    main()