from ruamel.yaml import YAML
import argparse
import json
import numpy as np
from urllib.parse import urlparse

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
def load_yaml(path):
    with open(path) as f:
        yaml = YAML(typ="safe")
        return yaml.load(f)
    
def init_argument_parser(add_arguments_fn):
    parser = argparse.ArgumentParser()
    parser = add_arguments_fn(parser)
    opt = parser.parse_args()
    return opt

def sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]

def fill_default_parameters(prompt_parameters:dict, default_parameters:dict)->dict:
    for key in default_parameters.keys():
        if key not in prompt_parameters.keys():
            prompt_parameters[key] = default_parameters[key]
    return prompt_parameters

def save_parameters_file(file_path:str,
                         opt):
    with open(file_path, "w") as f:
        json.dump(vars(opt), f, cls=NpEncoder,ensure_ascii=False,indent=4)

def save_input_prompt_file(file_path:str,
                         input_prompt):
    with open(file_path, "w") as f:
        f.write(input_prompt)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)