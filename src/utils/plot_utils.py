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