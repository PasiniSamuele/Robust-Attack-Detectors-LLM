from random import uniform
import pandas as pd
from ruamel.yaml import YAML
import argparse
import numpy as np
np.random.seed(42)

def uniform_dataset(opt, params):
    df = pd.read_csv(opt.data, encoding = "ISO-8859-1")
    uniform_params = params[opt.uniform_arguments]
    #keep only payload and label columns
    df = df[[uniform_params["payload_column"], uniform_params["label_column"]]]
    #keep only rows with label that is malicious or benign
    df = df[df[uniform_params["label_column"]].isin([uniform_params["benign_label"], uniform_params["malicious_label"]])]
    #rename benign label to final_benign_label and malicious label to final_malicious_label
    mapping_labels ={
        uniform_params["benign_label"]: uniform_params["final_benign_label"],
        uniform_params["malicious_label"]: uniform_params["final_malicious_label"]
    }
    df[uniform_params["label_column"]] = df[uniform_params["label_column"]].map(mapping_labels)
    #uniform the number of benign and malicious samples
    benign_samples = df[df[uniform_params["label_column"]] == uniform_params["final_benign_label"]]
    malicious_samples = df[df[uniform_params["label_column"]] == uniform_params["final_malicious_label"]]
    if len(benign_samples) > len(malicious_samples):
        benign_samples = benign_samples.sample(n=len(malicious_samples))
    else:
        malicious_samples = malicious_samples.sample(n=len(benign_samples))
    df = pd.concat([benign_samples, malicious_samples])
    #rename label column to final_label_column and payload column to final_payload_column
    df.rename(columns={uniform_params["label_column"]: uniform_params["final_label_column"], uniform_params["payload_column"]: uniform_params["final_payload_column"]}, inplace=True)
    df.to_csv(opt.dest, index=False)

    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dataset.csv', help='source')
    parser.add_argument('--dest', type=str, default='data/uniformed_dataset.csv', help='destination')
    parser.add_argument('--params', type=str, default='params.yaml', help='params')  # file/folder, 0 for webcam
    parser.add_argument('--uniform_arguments', type=str, default='uniform_sqli', help='parameters containing uniform params')

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_arguments()
    with open(opt.params) as f:
        yaml = YAML(typ="safe")
        params = yaml.load(f) 
    uniform_dataset(opt, params)

    

if __name__ == '__main__':
    main()