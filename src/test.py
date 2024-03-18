from utils.path_utils import get_experiment_folder

def test(opt):
    exp_folder = get_experiment_folder(opt.experiment_folder,
                          opt.task.split("/")[-1].split(".")[0],
                          opt.template.split("/")[-1].split(".")[0],
                          opt.prompt_parameters.split("/")[-1].split(".")[0],
                          opt.model,
                          opt.generation_mode,
                          opt.n_few_shot,
                          opt.temperature,
                          opt.seed)
    
    evaluation_namespace = Namespace(**vars(opt))



def add_parse_arguments(parser):
    #general parameters
    parser.add_argument('--experiment_folder', type=str, default="experiments", help='Folder containing experiments')
    parser.add_argument('--task', type=str, default="data/tasks/detect_xss_simple_prompt.txt", help='Summary file')
    parser.add_argument('--template', type=str, default="data/templates/detect_xss_simple_prompt.txt", help='Summary file')
    parser.add_argument('--prompt_parameters', type=str, default="data/prompt_parameters/detect_xss_simple_prompt.txt", help='Summary file')
    parser.add_argument('--test_folder', type=str, default='test', help='Folder to save test results')
    parser.add_argument('--model', type=str, default='gpt-4-0125-preview', help='Model')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature')
    parser.add_argument('--generation_mode', type=float, default="rag_few_shot", help='Generation Mode')
    parser.add_argument('--n_few_shot', type=int, default=5, help='Number of examples per class')
    parser.add_argument('--seed', type=int, default=156, help='Seed')
    parser.add_argument('--dataset', type=str, default='data/test.csv', help='Testing Dataset')
    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    test(opt)

if __name__ == '__main__':
    main()