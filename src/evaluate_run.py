from re import sub
from utils.utils import init_argument_parser, NpEncoder
from utils.path_utils import get_last_run, get_subfolders
import pandas as pd
import os
import shutil
import json

def evaluate_run(opt):
    run_path = opt.run if opt.run is not None else get_last_run(opt)
    run_name = run_path.split('/')[-1]
    print(f"Evaluating {run_name}")
    test_set = pd.read_csv(opt.data)
    test_set['label'] = test_set.Class.map({'Malicious': 1, 'Benign': 0})

    if opt.isolated_execution:
        raise NotImplementedError('Isolated execution is not implemented yet')
    else:
        subfolders = get_subfolders(run_path)
        for subfolder in subfolders:
            experiment_results = {
                "failed": False,
            }
            exp_test_set = test_set.copy()
            subfolder_name = subfolder.split('/')[-1]
            print(f"Evaluating  {subfolder_name}")
            file = os.path.join(subfolder, 'generated.py')
           
            try:
                code_dir = os.path.dirname(os.path.realpath(__file__)).split('/')[-1]
                exec_dir = os.path.join(code_dir, opt.execution_dir)
                os.makedirs(exec_dir, exist_ok=True)
                shutil.copy2(file, exec_dir)
                new_file = os.path.join(opt.execution_dir, 'generated')

                #replace / with .
                new_file = new_file.replace('/', '.')
                statement = f'from {new_file} import {opt.function_name} as fn'
                exec(statement, globals())
                print(fn)
                print(fn("test"))
                print(fn("<script>alert('test')</script>"))

                exp_test_set['prediction'] = exp_test_set["Payloads"].apply(lambda x: int(fn(x)))
                #print(exp_test_set.tail())
                grouped_results = exp_test_set.groupby(['label', 'prediction'])['Payloads'].count().reset_index(name="count")
                cm = grouped_results.pivot(index='label', columns='prediction', values='count').fillna(0)
                missing_cols = [col for col in cm.index if col not in cm.columns]
                for col in missing_cols:
                    cm[col] = 0
                cm = cm[cm.index.values]
                true_positives = cm[1][1]
                true_negatives = cm[0][0]
                false_positives = cm[0][1]
                false_negatives = cm[1][0]

                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / (true_positives + false_negatives)
                f1 = 2 * (precision * recall) / (precision + recall)
                accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

                experiment_results["true_positives"] = true_positives
                experiment_results["true_negatives"] = true_negatives
                experiment_results["false_positives"] = false_positives
                experiment_results["false_negatives"] = false_negatives
                experiment_results["total"] = true_positives + true_negatives + false_positives + false_negatives
                experiment_results["precision"] = precision
                experiment_results["recall"] = recall
                experiment_results["f1"] = f1
                experiment_results["accuracy"] = accuracy
                with open(os.path.join(subfolder, opt.result_file_name), 'w') as f:
                    json.dump(experiment_results, f,ensure_ascii=False,indent=4, cls=NpEncoder)
                
            except Exception as e:
                #print(f"Experiment {subfolder_name} failed to execute")
                print(e)
                experiment_results["failed"] = True
                with open(os.path.join(subfolder, opt.result_file_name), 'w') as f:
                    json.dump(experiment_results, f,ensure_ascii=False,indent=4)
                continue
            
            





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



    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    evaluate_run(opt)

if __name__ == '__main__':
    main()