import pandas as pd
from ruamel.yaml import YAML
import argparse

def wrap_row_with_quotes(row):
    #if the row starts and ends with "
    if row[0] == '"' and row[-1] == '"':
        return row[1:-1]
    elif row[0] == "'" and row[-1] == "'":
        return row[1:-1]
    elif '"' in row:
        return row
    elif "'" in row:
        #substitute ' with "
        row = row.replace("'", '"')
        return row
    else:
        return row

def fix_dataset_quotes(opt):
    df = pd.read_csv(opt.data, encoding = "ISO-8859-1")
    # keep only rows without both ' and " at the same time
    search_for = ['"', "'"]
    df_filtered = df[~df['payload'].str.contains('|'.join(search_for))]

    #drop duplicates in Payloads attribute
    df_filtered = df_filtered.drop_duplicates(subset=['payload'])

    #wrap rows containing " using ' as delimiter
    df_filtered['payload'] = df_filtered['payload'].apply(wrap_row_with_quotes)

    df_filtered.to_csv(opt.dest, index=False)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dataset.csv', help='source')
    parser.add_argument('--dest', type=str, default='data/filtered_dataset.csv', help='destination')

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_arguments()
    fix_dataset_quotes(opt)

    

if __name__ == '__main__':
    main()