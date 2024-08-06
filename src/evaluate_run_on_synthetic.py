
from utils.utils import init_argument_parser
from argparse import Namespace
import json
from utils.path_utils import get_experiment_folder, from_file_to_name
import os
from evaluate_run import evaluate_run
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

def evaluate_synth_run(opt):
    #open parameters file
    with open(f"{opt.dataset_folder}/{opt.parameters_file_name}") as f:
        parameters = json.load(f)
    prompt_parameters = parameters["prompt_parameters"]
    with open(prompt_parameters) as f:
        yaml = YAML(typ="safe")
        params = yaml.load(f) 
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
    print(opt.run, exp_folder)

    evalaute_synth_dataset(n_datasets, exp_folder, evaluation_namespace)


def add_parse_arguments(parser):
    #run parameters
    parser.add_argument('--run', type=str, required=True, help='Run to be evaluated')
    parser.add_argument('--dataset_folder', type=str, default='synthetic_datasets/task_detect_xss_simple_prompt/template_create_synthetic_dataset/prompt_parameters_medium_dataset/model_anthropic-claude-3-haiku/generation_mode_few_shot/n_few_shot_1/temperature_0.5/seed_156/run_0', help='synthetic dataset folder')
    parser.add_argument('--parameters_file_name', type=str, default='parameters.json', help='parameters file name of the synthetic dataset')
    parser.add_argument('--function_name', type=str, default='detect_xss', help='name of the generated function to be executed for evaluation')
    parser.add_argument('--evaluation_folder', type=str, default='synthetic_results', help='synthetic dataset folder')
   

    #evaluation parameters
    parser.add_argument('--result_file_name', type=str, default='results.json', help='name of the results file')
    parser.add_argument('--test_results_file_name', type=str, default='test_results.json', help='name of the results file')

    return parser

def main():
    opt = init_argument_parser(add_parse_arguments)
    evaluate_synth_run(opt)

if __name__ == '__main__':
    main()