import pandas as pd
import argparse
from ruamel.yaml import YAML
import re
import random
import ast
from dateutil.parser import parse

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
    
def replace_substrings(s, placeholders):
    s = s.replace("'\\'", "'/'")
    # Pattern to match substrings enclosed in single or double quotes
    pattern = r"\".*?\"|'.*?'"
    
    # Function to replace matched substrings with a random placeholder
    def replacer(match):
        #check in match is a number
        try:
            if str(ast.literal_eval(match.group(0))).isdigit():
                return f"'{str(ast.literal_eval(match.group(0)))}'"
            #check if match is a date
            elif is_date(match.group(0)):
                return f"'{str(ast.literal_eval(match.group(0)))}'"
            else:
                return f"'{random.choice(placeholders)}'"
        except:
            return f"'{random.choice(placeholders)}'"
    
    # Use re.sub with the replacer function to replace matched substrings
    result = re.sub(pattern, replacer, s)
    
    return result

def wrap_row_with_quotes(row):
    #if the row starts and ends with "
    row = row.strip()
    if row[0] == '"' and row[-1] == '"':
        row = row[1:-1]
    if row[0] == "'" and row[-1] == "'":
        row = row[1:-1]
    if '"' in row:
        #substitute "" with '
        row = row.replace('"', "'")
    return row

def fix_dataset_quotes(opt, params):
    df = pd.read_csv(opt.data, encoding = "ISO-8859-1")
    # keep only rows without both ' and " at the same time
    seed = params['fix_dataset_quotes']['seed']
    random.seed(seed)
    placeholders_file = opt.placeholders_file
    with open(placeholders_file) as f:
        placeholders = f.readlines()
    min_placeholder_len = params['fix_dataset_quotes']['min_placeholder_len']
    placeholders = [p.strip() for p in placeholders if len(p.strip()) >= min_placeholder_len]
    sample_size_placeholders = params['fix_dataset_quotes']['sample_size_placeholders']
    placeholders = random.sample(placeholders, sample_size_placeholders)
    new_df = pd.DataFrame(columns=df.columns)
    new_df = pd.concat([new_df, df[df['Class']=="Malicious"]], ignore_index=True)
    df['Payloads'] = df['Payloads'].apply(replace_substrings, placeholders=placeholders)
    new_df = pd.concat([new_df, df[df['Class']=="Benign"]], ignore_index=True)

    search_for = ['"', "'"]
    df_filtered = new_df[~new_df['Payloads'].str.contains('&'.join(search_for))]

    #drop duplicates in Payloads attribute
    df_filtered = df_filtered.drop_duplicates(subset=['Payloads'])

    #wrap rows containing " using ' as delimiter
    df_filtered['Payloads'] = df_filtered['Payloads'].apply(wrap_row_with_quotes)

    df_filtered.to_csv(opt.dest, index=False)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dataset.csv', help='source')
    parser.add_argument('--dest', type=str, default='data/filtered_dataset.csv', help='destination')
    parser.add_argument('--placeholders_file', type=str, default='data/placeholders_file', help='placeholders file')

    parser.add_argument('--params', type=str, default='params.yaml', help='params')


    opt = parser.parse_args()
    return opt

def main():
    opt = parse_arguments()
    with open(opt.params) as f:
        yaml = YAML(typ="safe")
        params = yaml.load(f) 
    fix_dataset_quotes(opt, params)

    

if __name__ == '__main__':
    main()