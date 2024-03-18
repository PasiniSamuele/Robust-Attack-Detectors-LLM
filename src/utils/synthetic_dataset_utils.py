import pandas as pd
from pydantic.v1 import Field ,BaseModel
from typing import List
import pandas as pd
from sqlalchemy import column
import os
import json

class XSS_row(BaseModel):
    Payloads: str = Field(description="a string representing an http get request with payload")
    Class: str = Field(description="a string representing the class of the http get request, it is Malicious if the http get request contains an xss attack, otherwise it is Benign")

    
class XSS_dataset(BaseModel):
    dataset: List[XSS_row]

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([row.dict() for row in self.dataset])
    

def fill_df(chain, prompt_parameters):
    n_lines = prompt_parameters["rows"]
    failures = 0
    df = pd.DataFrame(columns=["Payloads", "Class"])
    malicious_rows = len(df[df["Class"] == "Malicious"])
    benign_rows = len(df[df["Class"] == "Benign"])
    while malicious_rows < n_lines or benign_rows < n_lines:
        try:
            print("Filled", malicious_rows, "malicious rows and", benign_rows, "benign rows, new generation")
            response = chain.invoke(prompt_parameters)
            new_df = response.to_df()
            #print(new_df)
            new_df_malicious_rows = len(new_df[new_df["Class"] == "Malicious"])
            new_df_benign_rows = len(new_df[new_df["Class"] == "Benign"])
            to_append_df = pd.DataFrame(columns=["Payloads", "Class"])
            if malicious_rows < n_lines:
                to_append_df = pd.concat([to_append_df, new_df[new_df["Class"] == "Malicious"].head(min((n_lines - malicious_rows), new_df_malicious_rows))])
            if benign_rows < n_lines:
                to_append_df = pd.concat([to_append_df, new_df[new_df["Class"] == "Benign"].head(min((n_lines - benign_rows), new_df_benign_rows))])
            df = pd.concat([df, to_append_df])
            #drop duplicates
            df = df.drop_duplicates()
            #print(df)
            
            malicious_rows = len(df[df["Class"] == "Malicious"])
            benign_rows = len(df[df["Class"] == "Benign"])
            failures = 0

        except Exception as e:
            print("Partial filling failed, try again")
            print(e)
            failures = failures + 1
            if failures > 10:
                print("Filling failed, moving to new generation")
                raise e
            continue
    return df

def save_subset_of_df(file, subset):
    #take values of class
    file_name = file.split("/")[-1]
    folder = file.split(file_name)[0]
    subset_folder = os.path.join(folder, str(subset))
    os.makedirs(subset_folder, exist_ok=True)
    df = pd.read_csv(file)
    classes = df["Class"].unique()
    #take top subset values of each class
    subset_df = pd.DataFrame(columns=["Payloads", "Class"])
    for c in classes:
        subset_df = pd.concat([subset_df, df[df["Class"] == c].head(subset)])
    subset_df.to_csv(os.path.join(subset_folder, file_name), index=False)

def get_df_metrics(summarized_results:dict,
                   top_ks:list,
                   dataset_size:int)->pd.DataFrame:
    df_metrics = pd.DataFrame(columns = ["top_k", "subset", "accuracy_std", "accuracy_diff"])
    
    for top_k in top_ks:
        accuracy_diff = summarized_results["top_k_metrics"][f"top_{top_k}"]["accuracy_diff"]
        df_metrics.loc[len(df_metrics.index)] = [top_k, dataset_size, summarized_results["synthetic_dataset"]["avg_std_accuracy"][f"top_{top_k}"]["avg_std_accuracy"],accuracy_diff]
        for subset in summarized_results["synthetic_dataset"]["subsets"].keys():
            accuracy_diff = summarized_results["top_k_metrics"]["subsets"][str(subset)][f"top_{top_k}"]["accuracy_diff"]
            df_metrics.loc[len(df_metrics.index)] = [top_k, subset, summarized_results["synthetic_dataset"]["subsets"][subset]["avg_std_accuracy"][f"top_{top_k}"]["avg_std_accuracy"],accuracy_diff]
    #drop duplicates in subset top_k, subset
    df_metrics = df_metrics.drop_duplicates(subset = ["top_k", "subset"])
    return df_metrics
    