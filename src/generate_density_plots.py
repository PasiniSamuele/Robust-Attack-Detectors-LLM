import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plot_folder = "rq2_plots"
file = "plots_rq2.csv"

plots_df = pd.read_csv(file)

#keep only experiments of gpt4, bison, opus, sonnet, llama or mixtral
plots_df = plots_df[plots_df['experiment'].str.contains("gpt-4|bison|sonnet|opus|llama3|mixtral-8x7b")]
top_ks = [1,3,5]
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 10))

plots_df["use_rag"] = plots_df['dataset'].map(lambda x: "rag" in x)
plots_df["code_generated_with_rag"] = plots_df['experiment'].map(lambda x: "rag" in x)

for top_k in top_ks:
    output_plots_folder = os.path.join(plot_folder, f"top_{top_k}")
    plt.figure(figsize=(10, 10))
    os.makedirs(output_plots_folder, exist_ok=True)
    ax = sns.violinplot(data=plots_df, y=f"top_{top_k}_acc_diff", color = "orange",  alpha = 0.6, linewidth = 2, fill = True)
    # ax.figure.set_size_inches(7,8)
    ax.set_title(f"Top {top_k}", fontsize=28)
    ax.set_ylabel("Accuracy Difference", fontsize=28)
    ax.set_xticks(np.arange(-0.1,0.5, 0.05)) 
    # [single_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) for single_ax in ax.axes.flat]
    # [single_ax.set_xlim(-0.1,0.5) for single_ax in ax.axes.flat]
    # [single_ax.set_xticks(range(-0.1,0.5, 0.05)) for single_ax in ax.axes.flat]
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder,f"acc_diff.pdf"))
    plt.close()



    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 8))
    ax = sns.violinplot(data=plots_df, y=f"top_{top_k}_acc_improvement",  color = "orangered",linewidth = 2, fill = True,  alpha = 0.6)
    # ax.figure.set_size_inches(7,8)
    ax.set_title(f"Top {top_k}", fontsize=28) 
    ax.set_ylabel("Accuracy Improvement", fontsize=28)
    ax.set_yticks(np.arange(-0.4,0.5, 0.1)) 
    ax.set_ylim(-0.4,0.5)
    ax.tick_params(axis='y', labelsize=25)

    plt.tight_layout()

    plt.savefig(os.path.join(output_plots_folder,f"acc_improvement.pdf"))
    plt.close()

for top_k in top_ks:
    output_plots_folder = os.path.join(plot_folder, "rag",f"top_{top_k}")
    plt.figure(figsize=(10, 10))
    os.makedirs(output_plots_folder, exist_ok=True)
    ax = sns.violinplot(data=plots_df, y=f"top_{top_k}_acc_diff", palette = "hls",  hue="use_rag", linewidth = 2, fill = True)
    ax.figure.set_size_inches(10,8)
    ax.set_title(f"Top {top_k} Accuracy Difference")
    ax.set_ylabel("Accuracy Difference")
    ax.set_xticks(np.arange(-0.1,0.5, 0.05))
    #set legend
    # [single_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) for single_ax in ax.axes.flat]
    # [single_ax.set_xlim(-0.1,0.5) for single_ax in ax.axes.flat]
    # [single_ax.set_xticks(range(-0.1,0.5, 0.05)) for single_ax in ax.axes.flat]

    plt.savefig(os.path.join(output_plots_folder,f"acc_diff.pdf"))
    plt.close()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(50, 40))
    ax = sns.violinplot(data=plots_df, y=f"top_{top_k}_acc_improvement", palette = "hls", linewidth = 2, fill = True, hue="use_rag")
    ax.figure.set_size_inches(10,8)
    ax.set_title(f"Top {top_k} Accuracy Improvement") 
    ax.set_ylabel("Accuracy Improvement")
    ax.set_xticks(np.arange(-0.4,0.5, 0.1))

    plt.savefig(os.path.join(output_plots_folder,f"acc_improvement.pdf"))
    plt.close()

for top_k in top_ks:
    output_plots_folder = os.path.join(plot_folder, "experiment_rag",f"top_{top_k}")
    plt.figure(figsize=(10, 10))
    os.makedirs(output_plots_folder, exist_ok=True)
    ax = sns.violinplot(data=plots_df, y=f"top_{top_k}_acc_diff", palette = "hls",  hue="code_generated_with_rag", linewidth = 2, fill = True, alpha=0.6)
    ax.figure.set_size_inches(10,8)
    ax.set_title(f"Top {top_k} Accuracy Difference")
    ax.set_ylabel("Accuracy Difference")
    ax.set_xticks(np.arange(-0.1,0.5, 0.05))
    #set legend
    # [single_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) for single_ax in ax.axes.flat]
    # [single_ax.set_xlim(-0.1,0.5) for single_ax in ax.axes.flat]
    # [single_ax.set_xticks(range(-0.1,0.5, 0.05)) for single_ax in ax.axes.flat]

    plt.savefig(os.path.join(output_plots_folder,f"acc_diff.pdf"))
    plt.close()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(50, 40))
    ax = sns.violinplot(data=plots_df, y=f"top_{top_k}_acc_improvement", palette = "hls", linewidth = 2, fill = True, hue="use_rag")
    ax.figure.set_size_inches(10,8)
    ax.set_title(f"Top {top_k} Accuracy Improvement") 
    ax.set_ylabel("Accuracy Improvement")
    ax.set_xticks(np.arange(-0.4,0.5, 0.1))

    plt.savefig(os.path.join(output_plots_folder,f"acc_improvement.pdf"))
    plt.close()