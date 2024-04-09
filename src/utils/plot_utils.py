from pydantic import BaseModel
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt
import os
from functools import partial

class PlotConfig(BaseModel):
    plot_fn :object
    hue : Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None
    figure_name : Optional[str] = None
    legend : Optional[str] = None
    palette : Optional[str] = None
    rotation : Optional[int] = None
    minimum_success_rate : Optional[float] = None

def generate_plots_config(opt):
    plot_configs = []
    if opt.generate_countplot:
        fn = partial(sns.countplot, x="generation_mode")
        plot_configs.append(
            PlotConfig(plot_fn = fn,
                       xlabel = "generation_mode", 
                        ylabel = "successes",
                        palette = "Spectral",
                        figure_name = "countplot.png")
        )
    if opt.generate_boxplot:
        fn = partial(sns.boxplot, x="generation_mode", y="accuracy")
        plot_configs.append(
            PlotConfig(plot_fn = fn,
                       xlabel = "generation_mode", 
                        ylabel = "accuracy",
                        palette = "Spectral",
                        figure_name = "boxplot.png",
                        minimum_success_rate = opt.minimum_success_rate)
        )
    if opt.generate_violinplot:
        fn = partial(sns.violinplot, x="generation_mode", y="accuracy")
        plot_configs.append(
            PlotConfig(plot_fn = fn,
                       xlabel = "generation_mode", 
                        ylabel = "accuracy",
                        palette = "Spectral",
                        figure_name = "violinplot.png",
                        minimum_success_rate = opt.minimum_success_rate)
        )
        return plot_configs

def generate_plot(plot_fn, 
                  df,
                  xlabel,
                  ylabel,
                  title,
                  figure_name,
                  save_dir,
                  hue_order,
                  hue = "examples_per_class",
                  palette = "Spectral",
                  legend = "full",
                  minimum_success_rate = 0,
                  rotation = 30):
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 10))
    ax = plot_fn(data=df[df["success"] > minimum_success_rate], hue = hue, palette=palette, legend = legend, order = ['no_rag', 'rag'], hue_order = hue_order)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    #save the plot
    plt.savefig(os.path.join(save_dir, figure_name))
    #close and open new plot
    plt.close()

def model_temperature_plots(df, plots_folder, model_temperature,plot_configs):
    save_dir = os.path.join(plots_folder, model_temperature)
    os.makedirs(save_dir, exist_ok=True)
    df_sub = df[df["model_temperature"] == model_temperature]
    hue_order = df.groupby('examples_per_class')["examples_per_class"].first().sort_values().index
    df_sub.to_csv("df_sub.csv")
    for plot_config in plot_configs:
        generate_plot(**plot_config.model_dump(exclude_none = True), 
                        df = df_sub, 
                        save_dir = save_dir, 
                        legend = "full", 
                        title = model_temperature, 
                        hue = "examples_per_class",
                        hue_order = hue_order)

def n_examples_plots(df, plots_folder, n_examples, plot_configs):
    save_dir = os.path.join(plots_folder, f"{n_examples}_examples")
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
        
def experiment_synth_plot(df, plots_folder, title, file_name, y):
    sns.color_palette("hls", 8)
    hue_order = df.groupby('top_k')["top_k"].first().sort_values().index
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 10))
    #set subset as str
    df["subset"] = df["subset"].astype(int)
    #create a list with an increasing order of subset values
    x_order = df.groupby('subset')["subset"].first().sort_values().index
    #set values in x_order as str
    x_order = list(map(str, x_order))
    df["subset"] = df["subset"].astype(str)
    #sort dataset by subset using x_order
    df = df.sort_values(by = "subset", key = lambda x: x.map({v: i for i, v in enumerate(x_order)}))

    ax = sns.lineplot(data=df, x="subset", y=y, sort = True, hue = "top_k",  hue_order = hue_order, palette="Spectral", legend = "full")
    ax.set_title(title)

    plt.savefig(os.path.join(plots_folder,file_name))

def experiment_std_synth_plot(df, plots_folder):
    experiment_synth_plot(df, plots_folder, "Accuracy std", "accuracy_std.png", "accuracy_std")

def experiment_acc_diff_synth_plot(df, plots_folder):
    experiment_synth_plot(df, plots_folder, "Accuracy diff", "accuracy_diff.png", "accuracy_diff")

def experiments_synth_boxplots(df, plots_folder, model, column_metric, row_metric, metric, hue_order):

    os.makedirs(plots_folder, exist_ok=True)
    df_sub = df[(df["model"] == model)]
    #hue_order = df_sub.groupby('synth_dataset_name')["synth_dataset_name"].first().sort_values().index

    #get number of columns as the number of the unique values in temperature
    n_cols = len(df_sub[column_metric].unique())

    #get the number of rows as the number of the unique values in top_k
    n_rows = len(df_sub[row_metric].unique())

    #sort dataframe by row and column metric
    df_sub = df_sub.sort_values(by = [row_metric, column_metric, "synth_dataset_name"])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(50, 40))
    for i, col_var in enumerate(df_sub[column_metric].unique()):
        for j, row_val in enumerate(df_sub[row_metric].unique()):
            ax = axes[j,i]
            sns.boxplot(data = df_sub[(df_sub[column_metric] == col_var) & (df_sub[row_metric] == row_val)], x = "synth_dataset_name", y = metric, ax = ax, hue = "synth_dataset_name", hue_order = hue_order).set(title=f'{row_metric} {row_val} {column_metric} {col_var}')
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    plt.savefig(os.path.join(plots_folder, f"{model}.png"))

def experiments_synth_boxplots_by_model(df_sub, plots_folder, row_metric, metric, hue_order):

    os.makedirs(plots_folder, exist_ok=True)

    #get number of columns as the number of the unique values in temperature
    n_cols = len(df_sub["model"].unique())

    #get the number of rows as the number of the unique values in top_k
    n_rows = len(df_sub[row_metric].unique())

    #sort dataset by row_metric
    df_sub = df_sub.sort_values(by = [row_metric, "synth_dataset_name"])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(50, 30))
    for i, model in enumerate(df_sub["model"].unique()):
        for j, row_var in enumerate(df_sub[row_metric].unique()):
            ax = axes[j,i]
            sns.boxplot(data = df_sub[(df_sub["model"] == model) & (df_sub[row_metric] == row_var)], x = "synth_dataset_name", y = metric, ax = ax, hue = "synth_dataset_name", hue_order = hue_order).set(title=f'{row_metric} {row_var} Model {model}')
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    plt.savefig(os.path.join(plots_folder, f"{metric}.png"))