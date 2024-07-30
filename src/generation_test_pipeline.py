import argparse
from generate_code_snippets import add_parse_arguments as add_generate_parse_arguments
from evaluate_run import add_parse_arguments as add_evaluate_parse_arguments, evaluate_run
from dotenv import dotenv_values

def run_pipeline(opt, env):
    #generate_code_snippets(opt, env)
    evaluate_run(opt)

def init_pipeline_parser():
    parser = argparse.ArgumentParser()
    parser = add_evaluate_parse_arguments(parser)
    parser = add_generate_parse_arguments(parser)
    opt = parser.parse_args()
    return opt

def main():
    opt = init_pipeline_parser()
    env = dotenv_values()
    run_pipeline(opt, env)

if __name__ == '__main__':
    main()