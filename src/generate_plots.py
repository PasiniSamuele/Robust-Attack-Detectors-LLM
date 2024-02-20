import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from utils.path_utils import from_folder_to_accuracy_list, from_folder_to_success
from utils.utils import init_argument_parser
from utils.plot_utils import generate_plot, generate_plots_config


def generate_plots(opt): 
    sns.color_palette("hls", 8)
    df = pd.read_csv(opt.summary_file)

    df.loc[df['generation_mode'] == 'zero_shot', 'examples_per_class'] = 0
    df.loc[df['generation_mode'] == 'rag', 'examples_per_class'] = 0
    df['generation_mode'] = df['generation_mode'].replace('rag_few_shot', 'rag')
    df['generation_mode'] = df['generation_mode'].replace('zero_shot', 'no_rag')
    df['generation_mode'] = df['generation_mode'].replace('few_shot', 'no_rag')

    df = df[['model_name', 'temperature', 'generation_mode','examples_per_class', 'folder']]

    df['accuracy'] = df['folder'].map(from_folder_to_accuracy_list)
    df['success'] = df['folder'].map(from_folder_to_success)


    #explode the accuracy list
    df = df.explode('accuracy')
    df = df.reset_index(drop=False)
    df["model_temperature"] = df["model_name"] + "_" + df["temperature"].astype(str)

    plot_configs = generate_plots_config(opt)
    if opt.model_temperature_plots:
        for model_temperature in df["model_temperature"].unique():
            save_dir = os.path.join(opt.plots_folder, model_temperature)
            os.makedirs(save_dir, exist_ok=True)
            df_sub = df[df["model_temperature"] == model_temperature]
            hue_order = df.groupby('examples_per_class')["examples_per_class"].first().sort_values().index

            for plot_config in plot_configs:
                generate_plot(**plot_config.model_dump(exclude_none = True), 
                              df = df_sub, 
                              save_dir = save_dir, 
                              legend = "full", 
                              title = model_temperature, 
                              hue = "examples_per_class",
                              hue_order = hue_order)

    if opt.n_examples_plots:
        for n_examples in df["examples_per_class"].unique():
            save_dir = os.path.join(opt.plots_folder, f"{n_examples}_examples")
            os.makedirs(save_dir, exist_ok=True)
            df_sub = df[df["examples_per_class"] == n_examples]
            hue_order = df.groupby('model_temperature')["model_temperature"].first().sort_values().index
            for plot_config in plot_configs:
                generate_plot(**plot_config.model_dump(exclude_none = True), 
                              df = df_sub, 
                              save_dir = save_dir, 
                              legend = "brief", 
                              title = f"{n_examples}_examples", 
                              hue = "model_temperature",
                              hue_order = hue_order)

def add_parse_arguments(parser):
    #general parameters
    parser.add_argument('--summary_file', type=str, default="experiments/task_detect_xss_simple_prompt/experiments_summary.csv", help='Summary file')
    parser.add_argument('--plots_folder', type=str, default='plots', help='Folder to save plots')
    parser.add_argument('--minimum_success_rate', type=float, default=0.5, help='Minimum success rate to print the plot')

    #plot parameters
    parser.add_argument('--generate_countplot', type=bool, default=True, help='Generate countplot for successes')
    parser.add_argument('--generate_boxplot', type=bool, default=True, help='Generate boxplot for accuracy')
    parser.add_argument('--generate_violinplot', type=bool, default=True, help='Generate violinplot for accuracy')

    #group parameters
    parser.add_argument('--model_temperature_plots', type=bool, default=True, help='Generate plots grouped by model and temperature')
    parser.add_argument('--n_examples_plots', type=bool, default=True, help='Generate plots grouped by number of examples per class')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    generate_plots(opt)

if __name__ == '__main__':
    main()

