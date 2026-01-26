from typing import Dict
import pandas as pd

result_dir = "new_results"
pd.set_option('display.max_columns',  None)
pd.set_option('display.expand_frame_repr', False)

def generate_df(path, tag, result_dir="", filename="10_5"):
    filepath = f"{result_dir}/{tag}/{path}/{filename}.csv"
    return pd.read_csv(filepath)

def generate_difference_df(df_1: pd.DataFrame, df_2: pd.DataFrame):
    keys = ["id", "sentence"]
    df_1_new = df_1.set_index(keys).sort_index()
    df_2_new = df_2.set_index(keys).sort_index()
    metrics = [c for c in df_1.columns if c not in keys]
    df_result = df_1_new[metrics].subtract(df_2_new[metrics])
    return df_result.reset_index()

def print_top_bottom(df: pd.DataFrame, label_top, label_bottom, key='V-score'):
    df.sort_values(by=key, ascending=False, inplace=True)
    print(label_top)
    print(df.head(15))
    df.sort_values(by=key, ascending=True, inplace=True)
    print(label_bottom)
    print(df.head(15))

paths = ["EM", "kmeans", "NHMM"]
filenames = ["10_5", "10_5", "10_1"]
upos_dfs = {}
xpos_dfs = {}
improvement_dfs: Dict[str, pd.DataFrame] = {} 
for p, name in zip(paths, filenames):
    upos_dfs[p] = generate_df(p, "UPOS", result_dir=result_dir, filename=name)
    xpos_dfs[p] = generate_df(p, "XPOS", result_dir=result_dir, filename=name)
    improvement_dfs[p] = generate_difference_df(upos_dfs[p], xpos_dfs[p])
em_vs_km = generate_difference_df(upos_dfs["EM"], upos_dfs["kmeans"])
em_vs_nhmm = generate_difference_df(upos_dfs["EM"], upos_dfs["NHMM"])





print_top_bottom(upos_dfs["EM"], "Examples where EM performed best", "Examples where EM performed worst")
print_top_bottom(upos_dfs["kmeans"], "Examples where K-means performed best", "Examples where K-means performed worst")
print_top_bottom(em_vs_km, "Examples where EM performed better than K-means", "Examples where EM performed worse than K-means")
print_top_bottom(em_vs_nhmm, "Examples where EM performed better than the NHMM", "Examples where EM performed worse than the NHMM")
for p in paths:
    improvement_dfs[p].sort_values(by='V-score', inplace=True, ascending=False)
    print(f"Examples where {p} performed much better using XPOS tags than UPOS tags")
    print(improvement_dfs[p].head(15))











