from ruamel.yaml import YAML
import argparse
import pandas as pd

def balance_dataset(opt,params):
    class_column = params['balance_dataset']['class_column']
    #get all the values of the class column
    df = pd.read_csv(opt.data)
    #get the number of samples for each class
    samples = df[class_column].value_counts()
    #get the minimum number of samples
    max_samples = samples.min()

    new_df = pd.DataFrame(columns = df.columns)
    for i in samples.index:
        #get the samples of the class
        class_samples = df[df[class_column] == i]
        #get the first max_samples samples
        class_samples = class_samples.head(max_samples)
        #append the samples to the new dataframe
        new_df = pd.concat([new_df, class_samples])
    #save the new dataframe to the destination
    new_df.to_csv(opt.dest, index=False)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/unbalanced_dataset.csv', help='source folder')
    parser.add_argument('--dest', type=str, default='data/balanced_dataset.csv', help='destination')
    parser.add_argument('--params', type=str, default='params.yaml', help='params')

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_arguments()
    with open(opt.params) as f:
        yaml = YAML(typ="safe")
        params = yaml.load(f) 
    balance_dataset(opt, params)

if __name__ == '__main__':
    main()