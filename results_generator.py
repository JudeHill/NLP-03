import pandas as pd

em_dir = "EM_10_5_DONOTOVERWRITE"

dirs = [em_dir, "sEM/10_5", "hardEM"] # NHMM
stats = ["V-score", "homogeneity", "completeness", "VI", "normalized-VI"]

names = {
    em_dir: "5_10"
}



prog_results = {}
final_results = {}
for d in dirs:
    name = "10_5"
    if d == em_dir:
        name = "5_10"
    prog_results[d] = {
        "NVI": [],
        "V-measure": [],
    }
    final_results[d] = {}
    for i in range(10):
        filepath = f"{d}/{name}.{i}.csv"
        df = pd.read_csv(filepath)
        summary_stats = df.iloc[0]
        for s in stats:
            prog_results[d][s] = float(summary_stats[s])
    filepath = f"{d}/{name}.csv"
    summary_stats = df.iloc[0]
    for s in stats:
        final_results[d][s] = float(summary_stats[s])

for d in dirs:
    print(d)
    print(final_results[d])





