from numpy import number
from utils.openai_utils import is_openai_model, build_chat_model
from utils.utils import load_yaml, init_argument_parser, sanitize_output, fill_default_parameters
from utils.path_utils import create_folder_for_experiment
from dotenv import dotenv_values
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
import os

def generate_code_snippets(opt, env):
    # load template
    template = load_yaml(opt.template)
    # load parameters
    prompt_parameters = load_yaml(opt.prompt_parameters)

    #read txt containing the task
    with open(opt.task) as f:
        prompt_parameters["input"] = f.read()
    prompt_parameters = fill_default_parameters(prompt_parameters, template["default_parameters"])
    use_openai_api = is_openai_model(opt.model_name)
    if use_openai_api:
        model = build_chat_model(opt, env)
    else:
        raise NotImplementedError('Only OpenAI models are supported at the moment')
    
    #get experiment folder
    experiment_folder = create_folder_for_experiment(opt)
    #build prompt and chain
    prompt = ChatPromptTemplate.from_messages([("system", template["input"]), ("human", "{input}")])
    chain = prompt | model | StrOutputParser() | sanitize_output

    #run experiments
    for i in range(opt.experiments):
        print(f"Experiment {i}")
        try:
            response = chain.invoke(prompt_parameters)
            save_dir = os.path.join(experiment_folder, f"exp_{i}")
            os.makedirs(save_dir, exist_ok=False)
            with open(os.path.join(save_dir, 'generated.py'), 'w') as f:
                f.write(response)
        except Exception as e:
            print("Experiment failed, try again")
        
    

def add_parse_arguments(parser):

    #model parameters
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo', help='name of the model')
    parser.add_argument('--temperature', type=float, default=1.0, help='model temperature')

    #task parameters
    parser.add_argument('--task', type=str, default='data/tasks/detect_xss_simple_prompt.txt', help='input task')
    parser.add_argument('--template', type=str, default='data/templates/complete_function.yaml', help='template for the prompt')
    parser.add_argument('--prompt_parameters', type=str, default='data/prompt_parameters/empty.yaml', help='parameters to format the prompt template')
    parser.add_argument('--generation_mode', type=str, default='one_shot', help='Generation mode: one_shot, few_shot, rag or react')


    #output
    parser.add_argument('--experiments_folder', type=str, default='experiments', help='experiments folder')
    parser.add_argument('--experiments', type=int, default=25, help= 'number of experiments to run')


    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    env = dotenv_values()
    generate_code_snippets(opt, env)

if __name__ == '__main__':
    main()