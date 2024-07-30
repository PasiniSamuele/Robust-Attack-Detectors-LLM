from argparse import Namespace
from evaluate_run import evaluate_run
from utils.path_utils import get_experiment_folder, from_file_to_name, get_last_run_number
import json
import os
from utils.evaluation_utils import summarize_results
from utils.utils import init_argument_parser
def test(opt):
    
    evaluation_namespace = Namespace(**vars(opt))

    evaluation_namespace.create_confusion_matrix = False
    evaluation_namespace.summarize_results = True
    evaluation_namespace.run = None
    evaluation_namespace.top_k = opt.top_k
    evaluate_run(evaluation_namespace)

    exp_folder = get_experiment_folder(opt.experiments_folder,
                                        from_file_to_name(opt.task),
                                            from_file_to_name(opt.template),
                                            from_file_to_name(opt.prompt_parameters),
                                            opt.model_name,
                                            opt.generation_mode,
                                            opt.examples_per_class,
                                            opt.temperature,
                                            opt.seed)
    run = f"run_{get_last_run_number(exp_folder)}"
    #read json file of test_results
    with open(f"{exp_folder}/{run}/{opt.result_file_name}") as f:
        results = json.load(f)
    #get synthetic dataset folder
    synth_folder = get_experiment_folder(os.path.join(exp_folder,run,opt.synthetic_folder),
                                        from_file_to_name(opt.synth_dataset_task),
                                            from_file_to_name(opt.synth_dataset_template),
                                            from_file_to_name(opt.synth_dataset_prompt_parameters),
                                            opt.synth_dataset_model_name,
                                            opt.synth_dataset_generation_mode,
                                            opt.synth_dataset_examples_per_class,
                                            opt.synth_dataset_temperature,
                                            opt.synth_dataset_seed)

    #read json file of synthetic test_results
    with open(f"{synth_folder}/{opt.synth_results_file_name}") as f:
        synth_results = json.load(f)
    
    top_k_experiments = dict()
    for k in opt.top_k:
        top_k_experiments[k] = dict()
        top_k_experiments[k]["experiments"] = synth_results["synthetic_dataset"][f"top_{k}"]["experiments"]

        results_files = list(map(lambda x: f"{exp_folder}/{run}/{x}/{opt.result_file_name}", top_k_experiments[k]["experiments"]))
        results_json = list(map(lambda x: json.load(open(x)), results_files))

        #summarize the results of the experiments
        top_k_experiments[k]["summarized_results"] = summarize_results(results_json, opt.top_k_metric, [])
        best_result = results[f"top_{k}"]["accuracy"]

        #remove top_k_Experiment[top_k] from the dict
        results.pop(f"top_{k}")
        top_k_experiments[k]["summarized_results"]["accuracy_diff"] = best_result - top_k_experiments[k]["summarized_results"]["accuracy"]
    #add top_k experiments to the results
    results["top_k_experiments"] = top_k_experiments
    #save results
    with open(f"{exp_folder}/{run}/{opt.result_file_name}", 'w') as f:
        json.dump(results, f,ensure_ascii=False,indent=4)
    
    
    



def add_parse_arguments(parser):
    #general parameters
    parser.add_argument('--experiments_folder', type=str, default="experiments", help='Folder containing experiments')
    parser.add_argument('--task', type=str, default="data/tasks/detect_xss_simple_prompt.txt", help='Task file')
    parser.add_argument('--template', type=str, default="data/templates/create_function_readable.yaml", help='Template file')
    parser.add_argument('--prompt_parameters', type=str, default="data/prompt_parameters/empty.yaml", help='Prompt parameters file')
    parser.add_argument('--test_folder', type=str, default='test', help='Folder to save test results')
    parser.add_argument('--model_name', type=str, default='gpt-4-0125-preview', help='Model')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature')
    parser.add_argument('--generation_mode', type=str, default="rag_few_shot", help='Generation Mode')
    parser.add_argument('--examples_per_class', type=int, default=5, help='Number of examples per class')
    parser.add_argument('--seed', type=int, default=156, help='Seed')
    parser.add_argument('--data', type=str, default='data/test.csv', help='Testing Dataset')
    parser.add_argument('--evaluation_folder', type=str, default='test', help='test evaluation folder')
    parser.add_argument('--function_name', type=str, default='detect_xss', help='name of the generated function to be executed for evaluation')

    #synthetic dataset
    parser.add_argument('--synthetic_folder', default = "synthetic_results",type=str, help='folder containing synthetic results')
    parser.add_argument('--synth_dataset_task', type=str, default="data/tasks/detect_xss_simple_prompt.txt", help='Synthetic dataset task file')
    parser.add_argument('--synth_dataset_template', type=str, default="data/templates/create_synthetic_dataset.yaml", help='Synthetic dataset template file')
    parser.add_argument('--synth_dataset_prompt_parameters', type=str, default="data/prompt_parameters/medium_dataset.yaml", help='Synthetic dataset prompt parameters file')
    parser.add_argument('--synth_dataset_model_name', type=str, default='gpt-4-0125-preview', help='Synthetic dataset model')
    parser.add_argument('--synth_dataset_temperature', type=float, default=1.0, help='Synthetic dataset temperature')
    parser.add_argument('--synth_dataset_generation_mode', type=str, default="few_shot", help='Synthetic dataset generation Mode')
    parser.add_argument('--synth_dataset_examples_per_class', type=int, default=5, help='Synthetic dataset number of examples per class')
    parser.add_argument('--synth_dataset_seed', type=int, default=156, help='Synthetic dataset seed')
    parser.add_argument('--synth_results_file_name', type=str, default='results.json', help='synthetic dataset file_name')


    parser.add_argument('--isolated_execution', type=bool, default=False, help='if true, the evaluation will be executed in a separate docker environment')
    parser.add_argument('--result_file_name', type=str, default='test_results.json', help='name of the results file')
    parser.add_argument('--top_k_metric', type=str, default='accuracy', help='metric used to select the best experiments in the run')
    parser.add_argument('--top_k', type=int, action='append', help='top_k value to be considered for the top_k_metric, you can append more than one')
    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    test(opt)

if __name__ == '__main__':
    main()