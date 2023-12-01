import pandas as pd
from ruamel.yaml import YAML
import argparse


def filter_dataset(opt, params):
    df = pd.read_csv(opt.data, encoding = "ISO-8859-1")
    starting_strs = params["filter_dataset"]["starting_strs"]
    # filter df keeping only the rows where the payloads starts with any of the elements in starting_strs
    df_filtered = df[df.Payloads.str.startswith(tuple(starting_strs))]

    #drop duplicates in Payloads attribute
    df_filtered = df_filtered.drop_duplicates(subset=['Payloads'])
    df_filtered.to_csv(opt.dest, index=False)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dataset.csv', help='source')
    parser.add_argument('--dest', type=str, default='data/filtered_dataset.csv', help='destination')
    parser.add_argument('--params', type=str, default='params.yaml', help='params')  # file/folder, 0 for webcam

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_arguments()
    with open(opt.params) as f:
        yaml = YAML(typ="safe")
        params = yaml.load(f) 
    filter_dataset(opt, params)

    

if __name__ == '__main__':
    main()