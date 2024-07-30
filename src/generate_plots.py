import pandas as pd
import seaborn as sns
from utils.path_utils import from_folder_to_accuracy_list,from_folder_to_success,get_list_of_synth_results
from utils.utils import init_argument_parser
from utils.plot_utils import generate_plots_config, model_temperature_plots, n_examples_plots
import warnings

warnings.filterwarnings("ignore")

def generate_plots(opt): 
    sns.color_palette("hls", 8)
    df = pd.read_csv(opt.summary_file)

    df.loc[df['generation_mode'] == 'zero_shot', 'examples_per_class'] = 0
    df.loc[df['generation_mode'] == 'rag', 'examples_per_class'] = 0
    df['generation_mode'] = df['generation_mode'].replace('rag_few_shot', 'rag')
    df['generation_mode'] = df['generation_mode'].replace('zero_shot', 'no_rag')
    df['generation_mode'] = df['generation_mode'].replace('few_shot', 'no_rag')
    basic_columns = ['model_name', 'temperature', 'generation_mode','examples_per_class', 'folder']
    df = df[basic_columns]

    df["synthetic_dataset"] = df["folder"].map(lambda x: get_list_of_synth_results(x, opt.synthetic_datasets_folder))

    #obtain a set of all the synthetic datasets
    synthetic_datasets = set()
    for dataset in df["synthetic_dataset"]:
        synthetic_datasets.update(dataset)
    
    #create a column for each synthetic dataset that is True if this dataset is present in the list and False otherwise
    for dataset in synthetic_datasets:
        df[dataset] = df["synthetic_dataset"].map(lambda x: dataset in x)
    #drop the synthetic_dataset column
    df = df.drop(columns = ["synthetic_dataset"])
    #transform synthetic datasets in list
    synthetic_datasets = list(synthetic_datasets)
    df['accuracy'] = df['folder'].map(from_folder_to_accuracy_list)
    df['success'] = df['folder'].map(from_folder_to_success)


    #explode the accuracy list
    df = df.explode('accuracy')
    df = df.reset_index(drop=False)
    df["model_temperature"] = df["model_name"] + "_" + df["temperature"].astype(str)

    plot_configs = generate_plots_config(opt)
    if opt.model_temperature_plots:
        for model_temperature in df["model_temperature"].unique():
            print(f"Model: {model_temperature}")
            model_temperature_plots(df, opt.plots_folder, model_temperature, plot_configs)
            # for dataset in synthetic_datasets:
            #     print(f"Synthetic Dataset: {dataset}")

            #     df_dataset = df[df[dataset] == True]
            #     #drop duplicates considering model_temperature, generation_mode and examples_per_class
            #     df_dataset = df_dataset.drop_duplicates(subset = ["model_temperature", "generation_mode", "examples_per_class"])

            #     #open json file corresponding to the dataset in the folder and get top_k indexes, and top_k experiments
            #     df_dataset['top_k_metrics'] = df_dataset['folder'].map(lambda x: from_folder_to_top_k(x, dataset))
            #     top_ks_metrics = set()
            #     for metric in df_dataset["top_k_metrics"]:
            #         top_ks_metrics.update(metric)
            #     df_dataset = df_dataset.drop(columns = ["top_k_metrics"])

            #     for top_k_metric in top_ks_metrics:
            #         df_dataset[top_k_metric] = df_dataset["folder"].map(lambda x: from_folder_to_top_k_experiments(x, dataset, top_k_metric))
            #         print(f"Top K: {top_k_metric}")
            #         df_dataset_top_k = df_dataset.copy()
            #         df_dataset_top_k["accuracy"] = df_dataset_top_k.apply(lambda x: from_folder_to_accuracy_list(x["folder"], x[top_k_metric]) , axis = 1)
            #         df_dataset_top_k = df_dataset_top_k.explode('accuracy')
            #         df_dataset_top_k.to_csv("df_dataset_top_k.csv")
                    
            #         #df_dataset = df_dataset.reset_index(drop=False)
            #         #print(opt.plots_folder + os.path.join(dataset,top_k_metric))
            #         model_temperature_plots(df_dataset_top_k, opt.plots_folder + os.path.join(dataset,top_k_metric), model_temperature, plot_configs)
                # df_dataset['subsets'] = df_dataset['folder'].map(lambda x: from_folder_to_subsets(x, dataset))
                # subsets = set()
                # for subset in df_dataset["subsets"]:
                #     subsets.update(subset)
                # df_dataset = df_dataset.drop(columns = ["subsets"])
                # for subset in subsets:
                #     print(f"Subset: {subset}")
                #     for top_k_metric in top_ks_metrics:
                #         print(f"Top K: {top_k_metric}")
                #         df_dataset[top_k_metric] = df_dataset["folder"].map(lambda x: from_folder_to_top_k_subset_experiments(x, dataset, top_k_metric, subset))
                    
                #         df_dataset_top_k = df_dataset.copy()
                #         df_dataset_top_k["accuracy"] = df_dataset_top_k.apply(lambda x: from_folder_to_accuracy_list(x["folder"], x[top_k_metric]) , axis = 1)
                #         df_dataset_top_k = df_dataset_top_k.explode('accuracy')
                #         #df_dataset = df_dataset.reset_index(drop=False)
                #         #print(opt.plots_folder + os.path.join(dataset,top_k_metric))
                #         model_temperature_plots(df_dataset_top_k, opt.plots_folder + os.path.join(dataset,"subsets",subset,top_k_metric), model_temperature, plot_configs)



    if opt.n_examples_plots:
        for n_examples in df["examples_per_class"].unique():
            print(f"N_examples: {n_examples}")
            n_examples_plots(df, opt.plots_folder, n_examples, plot_configs)
            
            # for dataset in synthetic_datasets:
            #     print(f"Synthetic Dataset: {dataset}")

            #     df_dataset = df[df[dataset] == True]
            #     #drop duplicates considering model_temperature, generation_mode and examples_per_class
            #     df_dataset = df_dataset.drop_duplicates(subset = ["model_temperature", "generation_mode", "examples_per_class"])

            #     #open json file corresponding to the dataset in the folder and get top_k indexes, and top_k experiments
            #     df_dataset['top_k_metrics'] = df_dataset['folder'].map(lambda x: from_folder_to_top_k(x, dataset))
            #     top_ks_metrics = set()
            #     for metric in df_dataset["top_k_metrics"]:
            #         top_ks_metrics.update(metric)
            #     df_dataset = df_dataset.drop(columns = ["top_k_metrics"])
            #     for top_k_metric in top_ks_metrics:
            #         df_dataset[top_k_metric] = df_dataset["folder"].map(lambda x: from_folder_to_top_k_experiments(x, dataset, top_k_metric))
            #         print(f"Top K: {top_k_metric}")
            #         df_dataset_top_k = df_dataset.copy()
            #         df_dataset_top_k["accuracy"] = df_dataset_top_k.apply(lambda x: from_folder_to_accuracy_list(x["folder"], x[top_k_metric]) , axis = 1)
            #         df_dataset_top_k = df_dataset_top_k.explode('accuracy')
            #         df_dataset_top_k.to_csv("df_dataset_top_k.csv")
                    

            #         n_examples_plots(df_dataset_top_k, opt.plots_folder + os.path.join(dataset,top_k_metric), n_examples, plot_configs)
                
                # df_dataset['subsets'] = df_dataset['folder'].map(lambda x: from_folder_to_subsets(x, dataset))
                # subsets = set()
                # for subset in df_dataset["subsets"]:
                #     subsets.update(subset)
                # df_dataset = df_dataset.drop(columns = ["subsets"])
                # for subset in subsets:
                #     print(f"Subset: {subset}")
                #     for top_k_metric in top_ks_metrics:
                #         print(f"Top K: {top_k_metric}")
                #         df_dataset[top_k_metric] = df_dataset["folder"].map(lambda x: from_folder_to_top_k_subset_experiments(x, dataset, top_k_metric, subset))
                    
                #         df_dataset_top_k = df_dataset.copy()
                #         df_dataset_top_k["accuracy"] = df_dataset_top_k.apply(lambda x: from_folder_to_accuracy_list(x["folder"], x[top_k_metric]) , axis = 1)
                #         df_dataset_top_k = df_dataset_top_k.explode('accuracy')
                #         #df_dataset = df_dataset.reset_index(drop=False)
                #         #print(opt.plots_folder + os.path.join(dataset,top_k_metric))
                #         n_examples_plots(df_dataset_top_k, opt.plots_folder + os.path.join(dataset,"subsets",subset,top_k_metric), n_examples, plot_configs)


def add_parse_arguments(parser):
    #general parameters
    parser.add_argument('--summary_file', type=str, default="experiments/task_detect_xss_simple_prompt/experiments_summary_test.csv", help='Summary file')
    parser.add_argument('--plots_folder', type=str, default='plots_test/rq1/', help='Folder to save plots')
    parser.add_argument('--minimum_success_rate', type=float, default=0.1, help='Minimum success rate to print the plot')

    #plot parameters
    parser.add_argument('--generate_countplot', type=bool, default=True, help='Generate countplot for successes')
    parser.add_argument('--generate_boxplot', type=bool, default=True, help='Generate boxplot for accuracy')
    parser.add_argument('--generate_violinplot', type=bool, default=True, help='Generate violinplot for accuracy')

    #group parameters
    parser.add_argument('--model_temperature_plots', type=bool, default=True, help='Generate plots grouped by model and temperature')
    parser.add_argument('--n_examples_plots', type=bool, default=True, help='Generate plots grouped by number of examples per class')

    #synthetic datasets parameter
    parser.add_argument('--synthetic_datasets_folder', type=str, default='synthetic_results', help='Path with synthetic datasets')
    parser.add_argument('--synthetic_dataset_root', type=str, default='data/synthetic_datasets', help='Root containing synthetic dataset csv')
    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    generate_plots(opt)

if __name__ == '__main__':
    main()

