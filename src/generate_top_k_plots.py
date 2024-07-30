import pandas as pd
from utils.utils import init_argument_parser
import os
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_top_k_plots(opt):

    #rq1 - acc_diff plots
    df_test = pd.read_csv(opt.summary_file)
    df = pd.read_csv(opt.synth_summary_file)
    #print(df.columns)
    df['model_temperature'] = df['model_name'] + "_" + df['temperature'].astype(str)
    df.loc[df['generation_mode'] == 'zero_shot', 'examples_per_class'] = 0
    df.loc[df['generation_mode'] == 'rag', 'examples_per_class'] = 0
    df['generation_mode'] = df['generation_mode'].replace('rag_few_shot', 'rag')
    df['generation_mode'] = df['generation_mode'].replace('zero_shot', 'no_rag')
    df['generation_mode'] = df['generation_mode'].replace('few_shot', 'no_rag')

    df_test['model_temperature'] = df_test['model_name'] + "_" + df_test['temperature'].astype(str)
    df_test.loc[df_test['generation_mode'] == 'zero_shot', 'examples_per_class'] = 0
    df_test.loc[df_test['generation_mode'] == 'rag', 'examples_per_class'] = 0
    df_test['generation_mode'] = df_test['generation_mode'].replace('rag_few_shot', 'rag')
    df_test['generation_mode'] = df_test['generation_mode'].replace('zero_shot', 'no_rag')
    df_test['generation_mode'] = df_test['generation_mode'].replace('few_shot', 'no_rag')
    df_test['top_k'] = df_test['successes']

    if opt.rq2:
        for dataset in df["dataset"].unique():
            for model_temperature in df['model_temperature'].unique():
                print(dataset, model_temperature)
                dest_folder = os.path.join(opt.plots_folder, "rq2")

                dest_subfolder = os.path.join(dest_folder, f"dataset_{dataset}", f"model_{model_temperature}")
                os.makedirs(dest_subfolder, exist_ok=True)
                column_metric = "generation_mode"
                row_metric = "top_k"
                hue_metric = "examples_per_class"
                df_sub = df[(df["dataset"] == dataset) & (df["model_temperature"] == model_temperature)]
                #print(dataset, model_temperature)
                n_hues = len(df[hue_metric].unique())

                n_cols = len(df[column_metric].unique())

                #get the number of rows as the number of the unique values in top_k
                n_rows = len(df[row_metric].unique())
                #print(df_sub.generation_mode.unique())
                #sort dataframe by row and column metric
                df_sub = df_sub.sort_values(by = [row_metric, column_metric])
                #plt.figure(figsize=(20, 10))

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(50, 40))
                fig.suptitle(t=f'Dataset: {dataset}, model: {model_temperature}')
                

                for i, col_var in enumerate(df_sub[column_metric].unique()):
                    for j, row_val in enumerate(df_sub[row_metric].unique()):
                        ax = axes[j,i]
                        sns.boxplot(data = df_sub[(df_sub[column_metric] == col_var) & (df_sub[row_metric] == row_val)], hue = hue_metric, x = hue_metric, y = "accuracy_diff", ax = ax, palette=sns.color_palette("hls", n_hues) ).set(title=f'{column_metric}: {col_var}, {row_metric}: {row_val}')
                        #ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

                plt.savefig(os.path.join(dest_subfolder, f"boxplot.png"))
                plt.close(fig)
    if opt.rq3:
         for dataset in df["dataset"].unique():
            for model_temperature in df['model_temperature'].unique():
                dest_folder = os.path.join(opt.plots_folder, "rq3")

                print(dataset, model_temperature)
                dest_subfolder = os.path.join(dest_folder, f"dataset_{dataset}", f"model_{model_temperature}")
                os.makedirs(dest_subfolder, exist_ok=True)
                column_metric = "generation_mode"
                row_metric = "examples_per_class"
                hue_metric = "top_k"
                df_sub = df[(df["dataset"] == dataset) & (df["model_temperature"] == model_temperature)]
                #print(dataset, model_temperature)
                n_hues = len(df[hue_metric].unique())+1
                n_cols = len(df[column_metric].unique())

                #get the number of rows as the number of the unique values in top_k
                n_rows = len(df[row_metric].unique())
                #print(df_sub.generation_mode.unique())
                #sort dataframe by row and column metric
                df_sub = df_sub.sort_values(by = [row_metric, column_metric])
                #plt.figure(figsize=(20, 10))

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(50, 40))
                fig.suptitle(t=f'Dataset: {dataset}, model: {model_temperature}')
                

                for i, col_var in enumerate(df_sub[column_metric].unique()):
                    for j, row_val in enumerate(df_sub[row_metric].unique()):
                        ax = axes[j,i]
                        df_test_to_concat = df_test[ (df_test["model_temperature"] == model_temperature) & (df_test[column_metric] == col_var) & (df_test[row_metric] == row_val)]
                        df_sub_concat = pd.concat([
                            df_sub,
                            df_test_to_concat[["top_k", "accuracy", "generation_mode", "examples_per_class", "model_temperature", "temperature"]]
                        ])
                        sns.boxplot(data = df_sub_concat[(df_sub_concat[column_metric] == col_var) & (df_sub_concat[row_metric] == row_val)], hue = hue_metric, x = hue_metric, y = "accuracy", ax = ax, palette=sns.color_palette("hls", n_hues) ).set(title=f'{column_metric}: {col_var}, {row_metric}: {row_val}')
                        #ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

                plt.savefig(os.path.join(dest_subfolder, f"boxplot.png"))
                plt.close(fig)

def add_parse_arguments(parser):
    #general parameters
    parser.add_argument('--summary_file', type=str, default="experiments/task_detect_xss_simple_prompt/experiments_summary_test.csv", help='Summary file')
    parser.add_argument('--synth_summary_file', type=str, default="experiments/task_detect_xss_simple_prompt/test_results_synth.csv", help='Summary file')

    parser.add_argument('--plots_folder', type=str, default='plots_test/', help='Folder to save plots')
    parser.add_argument('--minimum_success_rate', type=float, default=0.1, help='Minimum success rate to print the plot')

    parser.add_argument('--rq2', type=bool, default=True, help='Generate rq2 plots')
    parser.add_argument('--rq3', type=bool, default=True, help='Generate rq3 plots')

    return parser

def main():
    opt = init_argument_parser(add_parse_arguments)
    generate_top_k_plots(opt)

if __name__ == '__main__':
    main()