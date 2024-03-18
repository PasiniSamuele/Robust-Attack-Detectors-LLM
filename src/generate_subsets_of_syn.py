from utils.synthetic_dataset_utils import save_subset_of_df
import os

syn_folder = "/workspaces/RAG_secure_code_generation/data/synthetic_datasets/task_detect_xss_simple_prompt/template_create_synthetic_dataset/prompt_parameters_medium_dataset/model_gpt-4-0125-preview/generation_mode_rag_few_shot/n_few_shot_5/temperature_1.0/seed_156/run_0"
#list all csv files in the folder
csv_files = [f for f in os.listdir(syn_folder) if f.endswith('.csv')]
subsets = [10, 20, 30, 40]
for f in csv_files:
    print(f)
    for s in subsets:
        save_subset_of_df(os.path.join(syn_folder,f), s)