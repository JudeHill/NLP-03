import pandas as pd

import matplotlib.pyplot as plt

em_dir = "EM_10_5_DONOTOVERWRITE"

dirs = [em_dir, "sEM/10_5", "hardEM", "mle", "kmeans"] # NHMM
stats = ["V-score", "homogeneity", "completeness", "VI", "normalized-VI"]




def compute_results(dirs, stats):
    prog_results = {}
    final_results = {}
    for d in dirs:
        name = "10_5"
        if d == em_dir:
            name = "5_10"
        elif d == "NHMM":
            name = "10_1"
        prog_results[d] = {s: [] for s in stats}
            
        final_results[d] = {}
        for i in range(10):
            filepath = f"{d}/{name}.{i}.csv"
            df = pd.read_csv(filepath)
            summary_stats = df.iloc[0]
            for s in stats:
                prog_results[d][s].append(float(summary_stats[s]))
        filepath = f"{d}/{name}.csv"
        df = pd.read_csv(filepath)
        summary_stats = df.iloc[0]
        for s in stats:
            final_results[d][s] = float(summary_stats[s])

    return prog_results, final_results


def plot_prog_results(stats, results, label="", filename="prog_results.png"):
    """
    Plots line graphs for specified metrics over training epochs.
    The ith element corresponds to 5*(i+1) epochs.
    """
    # Create the x-axis: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    epochs = [1 * (i + 1) for i in range(len(next(iter(results.values()))))]
    
    plt.figure(figsize=(6, 4)) 
    
    for metric in stats:
        if metric in results:
            plt.plot(epochs, results[metric], marker='o', label=metric)
        else:
            print(f"Warning: Metric '{metric}' not found in results.")

    plt.title(label)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(filename)






dirs_names = ["standard EM", "stochastic EM", "hard EM", "MLE"]

def plot_all_results():
    tag = "UPOS"
    for dir, name in zip(dirs, dirs_names):
        label = f"NVI and V-score convergence: {name} with {tag} tags"
        filename = f"{name}_{tag}_results.png"
        plot_prog_results(["V-score", "normalized-VI"], prog_results[dir], label=label, filename=filename)
    tag = "XPOS"
    for dir, name in zip(dirs_xpos, dirs_names):
        label = f"NVI and V-score convergence: {name} with {tag} tags"
        filename = f"{name}_{tag}_results.png"
        plot_prog_results(["V-score", "normalized-VI"], prog_results_xpos[dir], label=label, filename=filename)


def compute_results_sEM(filenames, tag, stats):
    prog_results = {}
    final_results = {}

    dir = "sEM"
    if tag == "XPOS":
        dir = f"XPOS/{dir}"
    for name in filenames:
        prog_results[name] = {s: [] for s in stats}
            
        final_results[name] = {}
        for i in range(10):
            filepath = f"{dir}/{name}.{i}.csv"
            df = pd.read_csv(filepath)
            summary_stats = df.iloc[0]
            for s in stats:
                prog_results[name][s].append(float(summary_stats[s]))
        filepath = f"{dir}/{name}.csv"
        summary_stats = df.iloc[0]
        for s in stats:
            final_results[name][s] = float(summary_stats[s])

    return prog_results, final_results

prog_results, final_results = compute_results_sEM(["alpha_6"], "XPOS", ["normalized-VI", "V-score"])
plot_prog_results(["normalized-VI", "V-score"], prog_results["alpha_6"], "Convergence of sEM using XPOS tags with alpha=0.6", "sEM_xpos_6.png")


