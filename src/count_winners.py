import pandas as pd
from utils.utils import init_argument_parser
import os
from utils.evaluation_utils import get_winner
import seaborn as sns
import matplotlib.pyplot as plt

def extract_dataset_parts(row):
   if row["dataset"].endswith("zero_shot") or row["dataset"].endswith("rag"):
      row["dataset"] = row["dataset"] +"_0"
   parts = row["dataset"].split("_")
   generation = "_".join(parts[2:-1])
   row["dataset_model"] = parts[0]
   row["dataset_temperature"] = parts[1]
   row["dataset_generation_mode"] = "no_rag" if generation == "zero_shot" or generation == "few_shot" else "rag"
   row["dataset_examples_per_class"] = parts[-1]
   return row

def create_experiment(row):
    row['experiment'] = row['model_name']+'_'+str(row['temperature'])+'_'+row['generation_mode']+'_'+str(row['examples_per_class'])
    return row


def create_plots_winners(df, save_folder, param):
    os.makedirs(save_folder, exist_ok=True)
    winners, winner_df = get_winner(df, param, query_on_functions = "model_name =='gpt-4-0125-preview' or model_name =='gpt-4-1106-preview'")
    winner_df = winner_df.sort_values(by=param)
    winners = pd.DataFrame(winners, index=[0])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 10))
    criterion_values = winner_df[param].unique()
    #check if all of them are digits
    if all([c.isdigit() for c in criterion_values]):
        criterion_values = [int(c) for c in criterion_values]

        #order them
        criterion_values.sort()
        winners = winners[list(map(lambda x:str(x),criterion_values))]

        ax = sns.countplot(data=winner_df, x=param, order = criterion_values, palette = "hls")
    elif(param == "temperature"):
        criterion_values = list(filter(lambda x: x in winner_df["temperature"].unique(),["0.0", "0.5", "1.0"]))
        winners = winners[list(map(lambda x:str(x),criterion_values))]

        ax = sns.countplot(data=winner_df, x=param, order = criterion_values, palette = "hls")
    else:
        ax = sns.countplot(data=winner_df, x=param, palette = "hls")
    plt.savefig(os.path.join(save_folder,f"winners_{param}.jpg"))
    winners.to_csv(os.path.join(save_folder, f"winners_{param}.csv"), index=False)


def count_winners(opt):
    df = pd.read_csv(opt.file)

    df = df.apply(extract_dataset_parts, axis=1)

    df['generation_mode'] = df['generation_mode'].replace('rag_few_shot', 'rag')
    df['generation_mode'] = df['generation_mode'].replace('zero_shot', 'no_rag')
    df['generation_mode'] = df['generation_mode'].replace('few_shot', 'no_rag')
    df = df.apply(create_experiment, axis=1)
    os.makedirs(opt.dest_folder, exist_ok=True)
    for param in opt.param:
        create_plots_winners(df, opt.dest_folder, param)
        for dataset_model in df['dataset_model'].unique():
            df_dataset = df[df['dataset_model'] == dataset_model]
            create_plots_winners(df_dataset, os.path.join(opt.dest_folder, dataset_model), param)
def add_parse_arguments(parser):
    #run parameters
    parser.add_argument('--file', type=str, default='test_results_synth.csv')
    parser.add_argument('--dest_folder', type=str, default='ranking_dataset_output')

    parser.add_argument('--param', type=str, action='append', help='paramters to be considered as criterion for the comparison')

    return parser

def main():
    opt = init_argument_parser(add_parse_arguments)
    count_winners(opt)

if __name__ == '__main__':
    main()