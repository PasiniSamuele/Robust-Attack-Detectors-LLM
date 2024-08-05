import json
from utils.openai_utils import is_openai_model, build_chat_model
from utils.hf_utils import create_hf_pipeline
from utils.utils import load_yaml, init_argument_parser, fill_default_parameters, save_parameters_file, save_input_prompt_file, is_valid_url
from utils.path_utils import create_folder_for_experiment, folder_exists_and_not_empty, get_last_run
from utils.synthetic_dataset_utils import XSS_dataset, SQLi_dataset, save_subset_of_df
from dotenv import dotenv_values
from langchain.prompts import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser

import os
import random
import numpy as np
from utils.few_shot_utils import create_few_shot
from utils.rag_utils import build_scientific_papers_loader, build_documents_retriever, build_web_page_loader, format_docs
import signal
from utils.synthetic_dataset_utils import FailureExpection, fill_df, TimeoutException
import multiprocessing 
def generate_synthetic_dataset(opt, env):
    
    #fix seed if it is not None
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)

    use_openai_api = is_openai_model(opt.model_name)

    #load model 
    model, embeddings = build_chat_model(opt, env) if use_openai_api else create_hf_pipeline(opt, env)
   
        
    # load template
    template = load_yaml(opt.template)
    # load parameters
    prompt_parameters = load_yaml(opt.prompt_parameters)

    #read txt containing the task
    with open(opt.task) as f:
        prompt_parameters["input"] = f.read()
    if opt.generation_mode == "few_shot" or opt.generation_mode == "rag_few_shot":
        prompt_parameters["input"] = create_few_shot(
            prompt_parameters["input"],
            opt.example_template,
            opt.example_positive_label,
            opt.example_negative_label,
            opt.examples_per_class,
            opt.examples_file
        )
    prompt_parameters = fill_default_parameters(prompt_parameters, template["default_parameters"])

    if opt.generation_mode == "rag" or opt.generation_mode == "rag_few_shot":
        with open(opt.rag_template_file) as f:
            template["input"] = template["input"] +"\n" + f.read()
        if not folder_exists_and_not_empty(opt.db_persist_path):
            print("Creating vectorstore")
            docs =  build_scientific_papers_loader(opt.papers_folder) if not is_valid_url(opt.rag_source) else build_web_page_loader(opt.rag_source)
        else: 
            docs = []
        retriever = build_documents_retriever(docs, db_persist_path=opt.db_persist_path, chunk_size=opt.chunk_size, chunk_overlap=opt.chunk_overlap, embeddings=embeddings)
        prompt_parameters['context'] = (retriever | format_docs).invoke(prompt_parameters['input'])

    #get experiment folder
    experiment_folder = create_folder_for_experiment(opt)
    if "xss" in opt.task:
        output_parser = PydanticOutputParser(pydantic_object=XSS_dataset)
    else:
        output_parser = PydanticOutputParser(pydantic_object=SQLi_dataset)

    #build prompt and chain
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(template['input']+ "\n{format_instructions}"),
            HumanMessagePromptTemplate.from_template("{input}")  
        ],
        input_variables=["input"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )
    chain = prompt | model | output_parser
    input_prompt = prompt.format(**prompt_parameters)
    print(input_prompt)
    save_input_prompt_file(os.path.join(experiment_folder, opt.input_prompt_file_name), input_prompt)
    save_parameters_file(os.path.join(experiment_folder, opt.parameters_file_name), opt)
    #run experiments
    i = 0
    failures = 0
    failed_subsets = 0
    

    while i < opt.experiments:
        print(f"Experiment {i}")
        try:
            
            df= fill_df(chain, prompt_parameters)
            save_file = os.path.join(experiment_folder, f"exp_{i}.csv")
            df.to_csv(save_file, index=False)
            i = i + 1
            failures = 0
        except TimeoutException:
            print("Experiment is over")
            signal.alarm(0)

            failed_subsets = failed_subsets + (opt.experiments - i)
            break

        except FailureExpection as e:
            #print(e)
            print("Experiment failed, try again")
            failures = failures + 1
            if failures > 10:
                print("Too many failures, moving to next experiment")
                i = i + 1
                failed_subsets = failed_subsets + 1
                continue
            continue
        
    
    results = {
        "experiments": opt.experiments,
        "failed_experiments": failed_subsets,
        "successful_experiments": opt.experiments - failed_subsets,
        "success": True if failed_subsets == 0 else False

    }
    #write json file
    json_file = os.path.join(experiment_folder, opt.results_file_name)
    with open(json_file, 'w') as f:
        json.dump(results, f)
        

def add_parse_arguments(parser):

    #model parameters
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-0613', help='name of the model')
    parser.add_argument('--temperature', type=float, default=1.0, help='model temperature')

    #task parameters
    parser.add_argument('--task', type=str, default='data/tasks/detect_xss_simple_prompt.txt', help='input task')
    parser.add_argument('--template', type=str, default='data/templates/create_synthetic_dataset.yaml', help='template for the prompt')
    parser.add_argument('--prompt_parameters', type=str, default='data/prompt_parameters/medium_dataset.yaml', help='parameters to format the prompt template')
    parser.add_argument('--generation_mode', type=str, default='zero_shot', help='Generation mode: zero_shot, few_shot, rag or react')


    #output
    parser.add_argument('--experiments_folder', type=str, default='data/synthetic_datasets', help='experiments folder')
    parser.add_argument('--experiments', type=int, default=5, help= 'number of experiments to run')
    parser.add_argument('--parameters_file_name', type=str, default='parameters.json', help='name of the parameters file')
    parser.add_argument('--results_file_name', type=str, default='results.json', help='name of the results file')

    parser.add_argument('--input_prompt_file_name', type=str, default='prompt.txt', help='name of the input prompt file')
    parser.add_argument('--subset', type=int, action = 'append', help='subset of the experiments to run')

    #hf parameters
    parser.add_argument('--hf_max_new_tokens', type=int, default=400, help='max new tokens for hf model')
    parser.add_argument('--hf_load_in_4bit', type=bool, default=True, help='load in 4 bit for hf model (qlora quantization)')

    #reproducibility
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')

    #few shot parameters
    parser.add_argument('--example_template', type=str, default='data/example_templates/detect_xss_simple_prompt.txt', help='template for the examples')
    parser.add_argument('--examples_per_class', type=int, default=0, help='number of examples for each class')
    parser.add_argument('--examples_file', type=str, default='data/train.csv', help='file containing the examples')
    parser.add_argument('--examples_payload_column', type=str, default='Payloads', help='column containing the payloads')
    parser.add_argument('--examples_label_column', type=str, default='Class', help='column containing the labels')
    parser.add_argument('--example_positive_label', type=str, default='Malicious', help='Label for positive examples')
    parser.add_argument('--example_negative_label', type=str, default='Benign', help='Label for negative examples')

    #rag parameters
    parser.add_argument('--rag_template_file', type=str, default='data/rag_templates/basic_rag_suffix.txt', help='template for the prompt with RAG')
    parser.add_argument('--rag_source', type=str, default='data/papers', help='folder with papers or url of a webpage')
    parser.add_argument('--db_persist_path', type=str, default='data/db/chroma', help='path to the db')
    parser.add_argument('--chunk_size', type=int, default=1000, help='chunk size')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='chunk overlap')

    parser.add_argument('--timeout', type=int, default=9000, help='timeout for the experiments in seconds')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    env = dotenv_values()
 
    p = multiprocessing.Process(target=generate_synthetic_dataset, args=(opt, env))
    p.start()

    p.join(opt.timeout)
    if p.is_alive():
        print("Timeout")
        p.terminate()
        p.join()
        results = {
        "experiments": opt.experiments,
        "success": False 

        }
        #write json file
        experiment_folder = get_last_run(opt)
        json_file = os.path.join(experiment_folder, opt.results_file_name)
        with open(json_file, 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    main()