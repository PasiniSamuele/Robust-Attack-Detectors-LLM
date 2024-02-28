import pandas as pd
from pydantic.v1 import Field ,BaseModel
from typing import List
import pandas as pd
from sqlalchemy import column

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
    while malicious_rows < n_lines and benign_rows < n_lines:
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
