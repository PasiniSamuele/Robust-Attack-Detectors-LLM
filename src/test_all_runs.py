import os
from argparse import Namespace
from evaluate_run import evaluate_run
import os
from utils.utils import init_argument_parser

def test_all_runs(opt):
    #find all folders named run_0 recursively inside experiments_root
    runs = []
    for root, dirs, files in os.walk(opt.experiments_root):
        #remove from dirs all the directories starting with exp_ or synthetic_
        for dir in dirs:
            if dir.startswith(opt.leaf_folder_name):
                if not os.path.exists(os.path.join(root, dir,opt.result_file_name)):
                    runs.append(os.path.join(root, dir))

    for run in runs:
        print(run)
        evaluation_namespace = Namespace()

        evaluation_namespace.create_confusion_matrix = False
        evaluation_namespace.summarize_results = True
        evaluation_namespace.run = run
        evaluation_namespace.data = opt.dataset
        evaluation_namespace.result_file_name = opt.result_file_name
        evaluation_namespace.function_name = opt.function_name


        evaluate_run(evaluation_namespace)

def add_parse_arguments(parser):
    #run parameters
    parser.add_argument('--experiments_root', type=str, default="generated_function_runs/task_detect_xss_simple_prompt/template_create_function_readable", help='Folder containing the generated function runs to be tested')
    parser.add_argument('--dataset', type=str, default='datasets/xss/test.csv', help='CSV file containing the dataset')
    parser.add_argument('--parameters_file_name', type=str, default='parameters.json', help='name of the parameters file')
    parser.add_argument('--function_name', type=str, default='detect_xss', help='name of the generated function to be executed for evaluation')
    parser.add_argument('--result_file_name', type=str, default='test_results.json', help='name of the results file to save results set')
    parser.add_argument('--leaf_folder_name', type=str, default='run_0', help='name of the leaf folder containing the generated data')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    test_all_runs(opt)

if __name__ == '__main__':
    main()
