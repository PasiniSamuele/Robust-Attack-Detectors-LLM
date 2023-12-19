import re
import pandas as pd
from typing import List
from functools import partial

def sample_examples(df:pd.DataFrame,
                    samples_per_class:int = 2, 
                    label_column:str = "Class",
                    shuffle = True,
                    format_function = lambda x: x):
    #extract different labels using label_column
    labels = df[label_column].unique()
    examples = pd.DataFrame()
    for l in labels:
        #extract samples_per_class examples for each label
        sample = df[df[label_column] == l].sample(samples_per_class)

        #add the examples to the examples dataframe
        examples = pd.concat([examples, sample], ignore_index=True)
    if shuffle:
        examples = examples.sample(frac=1)
    
    #format examples
    examples = format_function(examples)

    return examples

def humaneval_style_format(examples:pd.Series,
                    template:str,
                    label_column:str = "Class",
                    payload_column:str = "Payloads",
                    mappig:dict = {"Benign": False, "Malicious": True})-> List[str]:
    formatted_examples = []
    for _ , row in examples.iterrows():
        formatted_examples.append(template.format(input = row[payload_column], output = mappig[row[label_column]]))
    return formatted_examples

def create_few_shot(prompt:str,
                    example_template_file:str,
                    positive_label:str = "Malicious",
                    negative_label:str = "Benign",
                    examples_per_class:int = 2,
                    examples_file:str = 'data/train.csv',):
    mapping_dict = {negative_label: False, positive_label: True}
    trainset = pd.read_csv(examples_file)
    with open(example_template_file) as f:
        example_template = f.read()
    partial_format_function = partial(humaneval_style_format, label_column="Class", payload_column="Payloads", template =  example_template, mappig = mapping_dict)
    examples = sample_examples(trainset, samples_per_class=examples_per_class, shuffle=True, format_function=partial_format_function)
    prompt += "\n"
    for example in examples:
        prompt += example
    return prompt