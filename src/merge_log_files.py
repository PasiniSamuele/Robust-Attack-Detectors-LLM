from ruamel.yaml import YAML
import argparse
import os
import pandas as pd

def merge_logs(opt, params):
    #get all the .log files in the folder recursively
    logs = []
    for root, dirs, files in os.walk(opt.data):
        for file in files:
            if file.endswith(".log"):
                logs.append(os.path.join(root, file))
    params = params['merge_logs']
    kvs = [params['payload_column']['after'], params['class_column']['after']]
    payload_before = params['payload_column']['before']
    class_before = params['class_column']['before']
    payload_after = params['payload_column']['after']
    class_after = params['class_column']['after']
    dest = pd.DataFrame(columns = kvs, dtype = str)
    row = pd.Series(index = kvs, dtype = str)
    for log in logs:
        print(log)
        with open(log,errors='ignore') as f:
            lines = f.readlines()
            if len(lines) > 0:
                for line in lines:
                    #check if the line starts with the first values of the key-value pairs 
                    if line.startswith(payload_before):
                        #remove the key from the line
                        line = line.replace(payload_before, '')
                        #set the row value with the corresponding value from the line
                        row[payload_after] = line.strip()
                    elif line.startswith(class_before):
                        #remove the key from the line
                        line = line.replace(class_before, '')
                        #set the row value with the corresponding value from the line
                        row[class_after] = params["class_mapping"][line.strip()]
                        #append the row to the dataframe
                        dest.loc[len(dest)] = row
                        #reset the row
                        row = pd.Series(index = kvs, dtype = str)
                    else:
                        row[payload_after] = str(row[payload_after]) + ' ' + str(line.strip())

    #save the dataframe to the destination
    dest.to_csv(opt.dest, index=False)
                        

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='folder_with_logs', help='source folder')
    parser.add_argument('--dest', type=str, default='data/filtered_dataset.csv', help='destination')
    parser.add_argument('--params', type=str, default='params.yaml', help='params')  

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_arguments()
    with open(opt.params) as f:
        yaml = YAML(typ="safe")
        params = yaml.load(f) 
    merge_logs(opt, params)

if __name__ == '__main__':
    main()