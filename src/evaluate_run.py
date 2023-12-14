from re import sub
from utils.utils import init_argument_parser, NpEncoder
from utils.path_utils import get_last_run, get_subfolders
from utils.evaluation_utils import create_confusion_matrix, save_confusion_matrix, get_results_from_cm
import pandas as pd
import os
import shutil
import json
import seaborn as sn


def evaluate_run(opt):
    run_path = opt.run if opt.run is not None else get_last_run(opt)
    run_name = run_path.split('/')[-1]
    print(f"Evaluating {run_name}")
    test_set = pd.read_csv(opt.data)
    test_set['label'] = test_set.Class.map({'Malicious': 1, 'Benign': 0})

    if opt.isolated_execution:
        raise NotImplementedError('Isolated execution is not implemented yet')
    else:
        code_dir = os.path.dirname(os.path.realpath(__file__)).split('/')[-1]
        exec_dir = os.path.join(code_dir, opt.execution_dir)
        os.makedirs(exec_dir, exist_ok=True)
        subfolders = get_subfolders(run_path)
        for i, subfolder in enumerate(subfolders):
            experiment_results = {
                "failed": False,
            }
            exp_test_set = test_set.copy()
            subfolder_name = subfolder.split('/')[-1]
            print(f"Evaluating  {subfolder_name}")
            file = os.path.join(subfolder, 'generated.py')
           
            try:
                shutil.copy2(file, os.path.join(exec_dir,f"generated_{i}.py"))
                new_file = os.path.join(opt.execution_dir, f'generated_{i}')
                #replace / with .
                new_file = new_file.replace('/', '.')
                statement = f'from {new_file} import {opt.function_name} as fn_{i}'
                exec(statement, globals())
                exec(f"fn = fn_{i}", globals())

                exp_test_set['prediction'] = exp_test_set["Payloads"].apply(lambda x: int(fn(x)))
                #print(exp_test_set.tail())
                cm = create_confusion_matrix(exp_test_set)
                if opt.create_confusion_matrix:
                    save_confusion_matrix(cm, subfolder)
                experiment_results["results"] = get_results_from_cm(cm)
                with open(os.path.join(subfolder, opt.result_file_name), 'w') as f:
                    json.dump(experiment_results, f,ensure_ascii=False,indent=4, cls=NpEncoder)
                
            except Exception as e:
                #print(f"Experiment {subfolder_name} failed to execute")
                print(e)
                experiment_results["failed"] = True
                with open(os.path.join(subfolder, opt.result_file_name), 'w') as f:
                    json.dump(experiment_results, f,ensure_ascii=False,indent=4)
                continue
        #delete temp folder
        shutil.rmtree(exec_dir)
            
            





def add_parse_arguments(parser):
    #run parameters
    parser.add_argument('--run', type=str, default=None, help='Run to be evaluated, if it is None, the last run given model parameters will be evaluated')
    parser.add_argument('--data', type=str, default='data/test.csv', help='test dataset')
    parser.add_argument('--function_name', type=str, default='detect_xss', help='name of the generated function to be executed for evaluation')

    #evaluation parameters
    parser.add_argument('--isolated_execution', type=bool, default=False, help='if true, the evaluation will be executed in a separate docker environment')
    parser.add_argument('--summarize_results', type=bool, default=True, help='if true, the results for every experiment in the run will be summarized in a file')
    parser.add_argument('--result_file_name', type=str, default='results.json', help='name of the results file')
    parser.add_argument('--execution_dir', type=str, default='tmp', help='temporary directory for the execution of the generated code')
    parser.add_argument('--create_confusion_matrix', type=bool, default=True, help='if true, for every experiment it generates a confusion matrix')



    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    evaluate_run(opt)

if __name__ == '__main__':
    main()