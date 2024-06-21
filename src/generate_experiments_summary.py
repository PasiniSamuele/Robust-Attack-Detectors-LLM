from utils.utils import init_argument_parser
import os
import json
import pandas as pd

def generate_experiments_summary(opt):
    experiments_root_folder = opt.experiments_root_folder
    tail_folder = opt.tail_folder

    #get all the folders (recursively) inside experiments_root_folder
    results_dicts = []
    experiments_folders = []
    for root, dirs, files in os.walk(experiments_root_folder):
        for dir in dirs:
            if dir.startswith(tail_folder):
                experiments_folders.append(os.path.join(root, dir))
    for folder in experiments_folders:
        #load json of results
        results_file = os.path.join(folder, opt.results_file_name)
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results_dict = json.load(f)
        else:
            print(f"WARNING: results file {results_file} not found")
        #load json of parameters
        parameters_file = os.path.join(folder, opt.parameters_file_name)
        if os.path.exists(parameters_file):
            with open(parameters_file, 'r') as f:
                parameters_dict = json.load(f)
        else:
            print(f"WARNING: parameters file {parameters_file} not found")
        #merge the two dictionaries
        results_dict.update(parameters_dict)
        #if the run is None, set it to the folder name
        if 'run' not in results_dict.keys() or results_dict['run'] is None:
            results_dict['run'] = folder.split('/')[-1]
        results_dict['folder'] = folder
        to_remove_list = []
        temp_dict = dict()
        #flatten the dictionary
        for k, v in results_dict.items():
            if isinstance(v, dict):
                to_remove_list.append(k)
                for k2, v2 in v.items():
                    temp_dict[k+'_'+k2] = v2
        results_dict.update(temp_dict)
        for k in to_remove_list:
            del results_dict[k]
        results_dicts.append(results_dict)
    #convert the list of dictionaries to a dataframe
    df = pd.DataFrame(results_dicts)

    #filter the dataframe using template
    df = df[df['template'] == opt.template]
    #save the dataframe to a csv
    df.to_csv(os.path.join(opt.experiments_root_folder,opt.output_file))


def add_parse_arguments(parser):

    parser.add_argument('--experiments_root_folder', type=str, default='new_experiments_sap/task_detect_xss_simple_prompt/template_create_function_readable', help='root of the experiments')
    parser.add_argument('--tail_folder', type=str, default='run_', help='tail folder of the experiments')
    parser.add_argument('--parameters_file_name', type=str, default='parameters.json', help='name of the file containing the parameters')
    parser.add_argument('--results_file_name', type=str, default='results.json', help='name of the file containing the results')
    parser.add_argument('--template', type=str, default='data/templates/create_function_readable.yaml', help='template file')

    parser.add_argument('--output_file', type=str, default='experiments_summary.csv', help='name of the file containing the summary of the experiments')

    return parser

def main():
    opt = init_argument_parser(add_parse_arguments)
    generate_experiments_summary(opt)

if __name__ == '__main__':
    main()