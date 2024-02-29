from utils.utils import init_argument_parser, NpEncoder
from utils.path_utils import get_last_run, get_subfolders
from utils.evaluation_utils import create_confusion_matrix, save_confusion_matrix, get_results_from_cm, summarize_results
import pandas as pd
import os
import shutil
import json
import importlib
import importlib.util 


def evaluate_run(opt):
    run_path = opt.run if opt.run is not None else get_last_run(opt)
    run_name = run_path.split('/')[-1]
    print(f"Evaluating {run_name}")
    val_set = pd.read_csv(opt.data)
    val_set['label'] = val_set.Class.map({'Malicious': 1, 'Benign': 0})

    if opt.isolated_execution:
        raise NotImplementedError('Isolated execution is not implemented yet')
    else:
        subfolders = get_subfolders(run_path)
        for i, subfolder in enumerate(subfolders):
            experiment_results = {
                "failed": False,
            }
            exp_test_set = val_set.copy()
            subfolder_name = subfolder.split('/')[-1]
            print(f"Evaluating  {subfolder_name}")
            file = os.path.join(subfolder, 'generated.py')
            experiment_results["experiment"] = subfolder_name

            try:
            
                spec = importlib.util.spec_from_file_location(
                name=f"generation_{i}",
                location=file,
                )
                generation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(generation_module)

                exp_test_set['prediction'] = exp_test_set["Payloads"].apply(lambda x: int(generation_module.__dict__[opt.function_name](x)))
                #print(exp_test_set.tail())
                cm = create_confusion_matrix(exp_test_set)
                if opt.create_confusion_matrix:
                    save_confusion_matrix(cm, subfolder)
                experiment_results["results"] = get_results_from_cm(cm)

                #write into the results the number of the executed experiment
                filename = os.path.join(subfolder, opt.result_file_name)
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                with open(filename, 'w') as f:
                    json.dump(experiment_results, f,ensure_ascii=False,indent=4, cls=NpEncoder)
                
            except Exception as e:
                print(f"Experiment {subfolder_name} failed to execute")
                experiment_results["failed"] = True
                filename = os.path.join(subfolder, opt.result_file_name)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(experiment_results, f,ensure_ascii=False,indent=4)
                continue
        #delete temp folder
        #shutil.rmtree(exec_dir)

        if opt.summarize_results:
            single_results = list(map(lambda x: json.load(open(os.path.join(x, opt.result_file_name))), subfolders))
            summarized_results = summarize_results(single_results, opt.top_k_metric, opt.top_k)            
            with open(os.path.join(run_path, opt.result_file_name), 'w') as f:
                    json.dump(summarized_results, f,ensure_ascii=False,indent=4, cls=NpEncoder)




def add_parse_arguments(parser):
    #run parameters
    parser.add_argument('--run', type=str, default=None, help='Run to be evaluated, if it is None, the last run given model parameters will be evaluated')
    parser.add_argument('--data', type=str, default='data/val.csv', help='validation dataset')
    parser.add_argument('--function_name', type=str, default='detect_xss', help='name of the generated function to be executed for evaluation')

    #evaluation parameters
    parser.add_argument('--isolated_execution', type=bool, default=False, help='if true, the evaluation will be executed in a separate docker environment')
    parser.add_argument('--summarize_results', type=bool, default=True, help='if true, the results for every experiment in the run will be summarized in a file')
    parser.add_argument('--result_file_name', type=str, default='results.json', help='name of the results file')
    parser.add_argument('--create_confusion_matrix', type=bool, default=True, help='if true, for every experiment it generates a confusion matrix')
    parser.add_argument('--top_k_metric', type=str, default='accuracy', help='metric used to select the best experiments in the run')
    parser.add_argument('--top_k', type=int, action='append', help='top_k value to be considered for the top_k_metric, you can append more than one')




    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    evaluate_run(opt)

if __name__ == '__main__':
    main()