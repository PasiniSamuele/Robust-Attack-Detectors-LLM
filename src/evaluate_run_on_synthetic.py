
from utils.utils import init_argument_parser
from argparse import Namespace
import json
from utils.path_utils import get_experiment_folder, from_file_to_name, get_exp_subfolders, get_subfolders
import os
from utils.evaluation_utils import summarize_synth_results
from evaluate_run import evaluate_run
from utils.plot_utils import experiment_std_synth_plot, experiment_acc_diff_synth_plot
from utils.synthetic_dataset_utils import get_df_metrics
from ruamel.yaml import YAML

def synth_eval_run(dataset:int,
                   exp_folder:str, 
                   opt:Namespace,
                   result_file_name:str,
                   subset:int=None)->None:
        opt.data = os.path.join(opt.dataset_folder, subset or "", f"exp_{dataset}.csv")

        exp_folder_single_dataset = os.path.join(exp_folder, subset or "",f"exp_{dataset}")
        opt.result_file_name = os.path.join(exp_folder_single_dataset, result_file_name)

        opt.create_confusion_matrix = False
        opt.summarize_results = False
        evaluate_run(opt)

def evalaute_synth_dataset(n_datasets:str,
                           exp_folder:str,
                           opt:Namespace)->None:
    result_file_name = opt.result_file_name
    for i in range(n_datasets):
        synth_eval_run(i, exp_folder, opt, result_file_name)
        #find all subfolders in opt.dataset_folders
        subset_folders = get_subfolders(opt.dataset_folder)
        #keep only last folder name
        subset_folders = [f.split("/")[-1] for f in subset_folders]
        for subfolder in subset_folders:
            synth_eval_run(i, exp_folder, opt, result_file_name, subfolder)

def evaluate_synth_run(opt):
    #open parameters file
    with open(f"{opt.dataset_folder}/{opt.parameters_file_name}") as f:
        parameters = json.load(f)
    prompt_parameters = parameters["prompt_parameters"]
    with open(prompt_parameters) as f:
        yaml = YAML(typ="safe")
        params = yaml.load(f) 
    dataset_size = params["rows"]
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
    evalaute_synth_dataset(n_datasets, exp_folder, evaluation_namespace)

    subfolders = get_exp_subfolders(opt.run)
    summarized_results = summarize_synth_results(subfolders, n_datasets, exp_folder, opt)

    summary_dir = os.path.join(opt.run, exp_folder)

    if opt.plot:
        df_metrics = get_df_metrics(summarized_results, opt.top_k, dataset_size)
        experiment_std_synth_plot(df_metrics, summary_dir)
        experiment_acc_diff_synth_plot(df_metrics, summary_dir)

def add_parse_arguments(parser):
    #run parameters
    parser.add_argument('--run', type=str, required=True, help='Run to be evaluated')
    parser.add_argument('--dataset_folder', type=str, default='data/synthetic_datasets/task_detect_xss_simple_prompt/template_create_synthetic_dataset/prompt_parameters_medium_dataset/model_gpt-4-0125-preview/generation_mode_few_shot/n_few_shot_5/temperature_1.0/seed_156/run_0', help='synthetic dataset folder')
    parser.add_argument('--parameters_file_name', type=str, default='parameters.json', help='parameters file name of the synthetic dataset')
    parser.add_argument('--function_name', type=str, default='detect_xss', help='name of the generated function to be executed for evaluation')
    parser.add_argument('--evaluation_folder', type=str, default='synthetic_results', help='synthetic dataset folder')
   

    #evaluation parameters
    parser.add_argument('--isolated_execution', type=bool, default=False, help='if true, the evaluation will be executed in a separate docker environment')
    parser.add_argument('--result_file_name', type=str, default='results.json', help='name of the results file')
    parser.add_argument('--top_k_metric', type=str, default='accuracy', help='metric used to select the best experiments in the run')
    parser.add_argument('--top_k', type=int, action='append', help='top_k value to be considered for the top_k_metric, you can append more than one')

    #plot parameters
    parser.add_argument('--plot', type=bool, default=True, help='if true, the plots will be generated')
    return parser

def main():
    opt = init_argument_parser(add_parse_arguments)
    evaluate_synth_run(opt)

if __name__ == '__main__':
    main()