from ruamel.yaml import YAML
import argparse
import json
import numpy as np
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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)