import random
from pathlib import Path

import torch

import hmm_pipeline
import nhmm_pipeline
import kmeans_pipeline
from argparser import arg_parsing
from logging_nlp import set_logging_verbosity

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    Path("./logs").mkdir(parents=True, exist_ok=True)
    set_logging_verbosity("info")
    args = arg_parsing()
    if args["model"] == "kmeans":
        # Complete your code here
        kmeans_pipeline.lloyd(
            args["subset"],
            args["save_path"],
            args["res_path"]
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
