from utils.utils import init_argument_parser
import os
import json
import pandas as pd

def generate_analyzed_summaries(opt):
    df = pd.read_csv(opt.summary_file)
    parameters_to_keep = [
    "model_name",
    "temperature",
    "successes",
    ]
    top_n_parameters_to_keep = [
    "accuracy",
    "precision",
    "recall",
    "f1"
    ]

    top_n_values = [1,3,5,10,15]

    #keep only the rows with seed
    df = df[df["seed"] == opt.seed]
    df = df.round(3)
    df = df.sort_values(by=["model_name", "temperature"])

    #analyze zero shot
    df_zero_shot = df[df["generation_mode"] == "zero_shot"]
    #keep only the parameters we are interested in
    params = parameters_to_keep + top_n_parameters_to_keep
    df_zero_shot_total = df_zero_shot[params].copy()
    output = os.path.join(opt.output_folder, "experiments_zero_shot.csv")
    df_zero_shot_total.to_csv(output, index=False)

    #analyze zero shot top n values
    for n in top_n_values:
        output = os.path.join(opt.output_folder, f"experiments_zero_shot_top_{n}.csv")
        params = parameters_to_keep + list(map(lambda x: f"top_{n}_{x}",top_n_parameters_to_keep))
        df_zero_shot_top_n = df_zero_shot[params].copy()
        df_zero_shot_top_n.to_csv(output, index=False)

    #analyze few shot
    df_few_shot = df[df["generation_mode"] == "few_shot"]
    examples_values = list(df_few_shot["examples_per_class"].unique())
    for n_examples in examples_values:
        df_few_shot_examples = df_few_shot[df_few_shot["examples_per_class"] == n_examples].copy()
        params = parameters_to_keep + top_n_parameters_to_keep
        df_few_shot_total = df_few_shot_examples[params].copy()
        output = os.path.join(opt.output_folder, f"experiments_few_shot_{n_examples}_examples.csv")
        df_few_shot_total.to_csv(output, index=False)

        #analyze few shot top n values
        for n in top_n_values:
            output = os.path.join(opt.output_folder, f"experiments_few_shot_{n_examples}_examples_top_{n}.csv")
            params = parameters_to_keep + list(map(lambda x: f"top_{n}_{x}",top_n_parameters_to_keep))
            df_few_shot_top_n = df_few_shot_examples[params].copy()
            df_few_shot_top_n.to_csv(output, index=False)





def add_parse_arguments(parser):

    parser.add_argument('--summary_file', type=str, default='experiments/task_detect_xss_simple_prompt/experiments_summary.csv', help='summary_file')
    parser.add_argument('--output_folder', type=str, default='output', help='output_folder')
    parser.add_argument('--seed', type=int, default=156, help='output_folder')


    return parser
def main():
    opt = init_argument_parser(add_parse_arguments)
    generate_analyzed_summaries(opt)

if __name__ == '__main__':
    main()