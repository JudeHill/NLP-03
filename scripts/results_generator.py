import pandas as pd
import matplotlib.pyplot as plt


# setup
dirs = ["EM", "sEM/alpha_6","sEM/alpha_8", "sEM/alpha_10", "hardEM", "MLE", "kmeans", "NHMM"]
paths = [(d, "10_1" if d == "NHMM" else "10_5") for d in dirs]
stats = ["V-score", "homogeneity", "completeness", "VI", "normalized-VI"]
root_results_dir = "new_results"
figures_dir = "figures"



def compute_results(paths, stats, method_names, results_dir, tag):
    prog_data = []
    final_data = []
    
    for d, name in paths:
        # 1. Collect Progressive Results (indices 0-9)
        increment = 5
        if name == "10_1":
            increment = 1
        for i in range(10):
            filepath = f"{results_dir}/{tag}/{d}/{name}.{i}.csv"
            df = pd.read_csv(filepath)
            base_key = d.split("/")[-1] 
            pretty_name = method_names[base_key]
            # Create a row dictionary for this specific iteration
            row = {"method": pretty_name, "epochs": increment * (i+1)}
            for s in stats:
                row[s] = float(df.iloc[0][s])
            prog_data.append(row)
            
        # 2. Collect Final Results
        final_filepath = f"{results_dir}/{tag}/{d}/{name}.csv"
        df_final = pd.read_csv(final_filepath)
        
        final_row = {"method": pretty_name}
        for s in stats:
            final_row[s] = float(df_final.iloc[0][s])
        final_data.append(final_row)

    # Convert lists to DataFrames
    df_prog = pd.DataFrame(prog_data)
    df_final = pd.DataFrame(final_data)
    
    return df_prog, df_final


def plot_prog_results(df, stats, method_name, filename="prog_results.png", save_path="", label = ""):
    """
    Plots line graphs for specified metrics over training epochs for a single method.
    """
    method_df = df[df["method"] == method_name]
    epochs = method_df["epochs"]
    
    plt.figure(figsize=(6, 4)) 
    
    for metric in stats:
        plt.plot(epochs, method_df[metric], marker='o', label=metric)

    plt.title(label)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(f"{save_path}/{filename}")
    plt.close()


# Example call:
# plot_prog_results_df(progressive_results_df, ["V-score", "VI"], "sEM/alpha_6")
method_names = {
    "EM": "Batch EM",
    "alpha_6": "Stochastic EM (α=0.6)",
    "alpha_8": "Stochastic EM (α=0.8)",
    "alpha_10": "Stochastic EM (α=1.0)",
    "hardEM": "Hard EM",
    "MLE": "MLE",
    "kmeans": "K-Means",
    "NHMM": "Neural HMM"
}

prog_results, final_results = compute_results(
    paths, 
    stats, 
    method_names,
    results_dir=root_results_dir, 
    tag="UPOS")
print(final_results)


prog_results_xpos, final_results_xpos = compute_results(
    paths, 
    stats, 
    method_names,
    results_dir=root_results_dir, 
    tag="XPOS")




def plot_all_results(df, tag, save_path):
    unique_methods = df["method"].unique()
    for method in unique_methods:
        label = f"NVI and V-score convergence: {method} ({tag})"
        # Replace spaces and special characters for a safe filename
        safe_filename = method.replace(" ", "_").replace("α", "a").replace("=", "").replace("(", "").replace(")", "")
        filename = f"{safe_filename}_{tag}.png"
        
        plot_prog_results(df, ["V-score", "normalized-VI"], method, filename=filename, label=label, save_path=save_path)


plot_all_results(prog_results, "UPOS", save_path=figures_dir)
plot_all_results(prog_results_xpos, "XPOS", save_path=figures_dir)

sEM_short_results, _ = compute_results([("sEM/alpha_6", "10_1")], stats, method_names, results_dir=root_results_dir, tag="XPOS")
plot_prog_results(sEM_short_results, ["normalized-VI", "V-score"], method_names["alpha_6"], filename="sEM_short_xpos.png", save_path=figures_dir)







