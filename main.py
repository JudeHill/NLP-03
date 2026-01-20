import random
from pathlib import Path

import torch
import os

import hmm_pipeline
import nhmm_pipeline
import kmeans_pipeline
from argparser import arg_parsing
from logging_nlp import set_logging_verbosity

if __name__ == "__main__":
    # 1. Determine the total logical cores available on the VM
    total_cores = os.cpu_count() or 1

    # 2. Set the cap (Total cores minus 2, with a minimum of 1)
    thread_cap = max(1, total_cores - 2)

    # 3. Apply the limits to PyTorch's internal parallel engines
    torch.set_num_threads(thread_cap)
    torch.set_num_interop_threads(thread_cap)

    print(f"--- CPU Resource Control ---")
    print(f"Total Cores detected: {total_cores}")
    print(f"PyTorch Thread Cap:   {thread_cap} (Leaving 2 for OS)")
    print(f"----------------------------")
    torch.manual_seed(42)
    random.seed(42)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    set_logging_verbosity("info")
    args = arg_parsing()
    if args["model"] == "kmeans":
        # Complete your code here
        if args["action"] == "train-test":
            kmeans_pipeline.train_and_test(
                args["tag"],
                args["subset"],
                args["max_epochs"],
                args["load_path"],
                args["save_path"],
                args["res_path"],
            )
        else:
            kmeans_pipeline.test(
                args["tag"],
                args["subset"],
                args["load_path"],
                args["res_path"],
            )
    elif args["model"] == "nhmm":
        if args["action"] == "train-test":
            nhmm_pipeline.train_and_test(
                args["tag"],
                args["subset"],
                args["max_epochs"],
                args["load_path"],
                args["save_path"],
                args["res_path"],
            )
        else:
            nhmm_pipeline.test(
                args["tag"],
                args["subset"],
                args["load_path"],
                args["res_path"],
            )
    else:
        method = args["model"].split("-")[-1]
        if args["action"] == "train-test":
            hmm_pipeline.train_and_test(
                method,
                args["tag"],
                args["subset"],
                args["max_epochs"],
                args["load_path"],
                args["save_path"],
                args["res_path"],
            )
        else:
            hmm_pipeline.test(
                args["tag"],
                args["subset"],
                args["load_path"],
                args["res_path"],
            )
