import os
import json
from evaluate_run_on_synthetic import evaluate_synth_run
from argparse import Namespace
from utils.utils import init_argument_parser


def evaluate_runs(opt):
#find all folders named run_0 recursively inside experiments_root
    runs = []

    for root, dirs, files in os.walk(opt.experiments_root):
        if opt.leaf_folder_name in dirs:
                runs.append(os.path.join(root, opt.leaf_folder_name))



    datasets = []
    for root, dirs, files in os.walk(opt.datasets_root):
        if opt.leaf_folder_name in dirs:
            results_json = os.path.join(root, opt.leaf_folder_name, opt.dataset_generation_results_file_name)
            with open(results_json, 'r') as f:
                results = json.load(f)
                if results["success"] == True:
                    datasets.append(os.path.join(root, opt.leaf_folder_name))

    for run in runs: 
        for dataset in datasets:

            opt = Namespace(run=run, 
                            dataset_folder=dataset, 
                            plot = True,
                            parameters_file_name = opt.parameters_file_name,
                            function_name = opt.function_name,
                            evaluation_folder = opt.evaluation_folder,
                            result_file_name = opt.result_file_name,
                            test_results_file_name = opt.dataset_generation_results_file_name)
            evaluate_synth_run(opt)
        



def add_parse_arguments(parser):
    #run parameters
    parser.add_argument('--experiments_root', type=str, default="generated_function_runs/task_detect_xss_simple_prompt/template_create_function_readable", help='Folder containing the generated function runs to be tested')
    parser.add_argument('--datasets_root', type=str, default='synthetic_datasets/task_detect_xss_simple_prompt/template_create_synthetic_dataset/prompt_parameters_medium_dataset/', help='Folder containing the synthetic datasets')
    parser.add_argument('--parameters_file_name', type=str, default='parameters.json', help='name of the parameters file')
    parser.add_argument('--function_name', type=str, default='detect_xss', help='name of the generated function to be executed for evaluation')
    parser.add_argument('--evaluation_folder', type=str, default='synthetic_results', help='name of the evaluation folder to store the results')
    parser.add_argument('--result_file_name', type=str, default='val_results.json', help='name of the results file for val set')
    parser.add_argument('--dataset_generation_results_file_name', type=str, default='results.json', help='name of the results file for the generation of the synthetic dataset')
    parser.add_argument('--test_results_file_name', type=str, default='test_results.json', help='name of the results file for test set')
    parser.add_argument('--leaf_folder_name', type=str, default='run_0', help='name of the leaf folder containing the generated data')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    evaluate_runs(opt)

if __name__ == '__main__':
    main()
