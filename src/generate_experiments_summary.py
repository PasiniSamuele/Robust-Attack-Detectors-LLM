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
        if results_dict['run'] is None:
            results_dict['run'] = folder.split('/')[-1]
        results_dicts.append(results_dict)
    #convert the list of dictionaries to a dataframe
    df = pd.DataFrame(results_dicts)
    #save the dataframe to a csv
    df.to_csv(os.path.join(opt.experiments_root_folder,opt.output_file))


def add_parse_arguments(parser):

    parser.add_argument('--experiments_root_folder', type=str, default='experiments', help='root of the experiments')
    parser.add_argument('--tail_folder', type=str, default='run_', help='tail folder of the experiments')
    parser.add_argument('--parameters_file_name', type=str, default='parameters.json', help='name of the file containing the parameters')
    parser.add_argument('--results_file_name', type=str, default='results.json', help='name of the file containing the results')

    parser.add_argument('--output_file', type=str, default='experiments_summary.csv', help='name of the file containing the summary of the experiments')

    return parser
def main():
    opt = init_argument_parser(add_parse_arguments)
    generate_experiments_summary(opt)

if __name__ == '__main__':
    main()