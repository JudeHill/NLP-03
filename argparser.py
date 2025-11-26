import argparse
import os


def arg_parsing():
    argparser = argparse.ArgumentParser()
    ACTIONS = ["train-test", "test"]
    MODELS = ["hmm-mle", "hmm-EM", "hmm-sEM", "hmm-hardEM", "kmeans"]
    TAGS = ["upos", "xpos"]
    argparser.add_argument(
        "action",
        choices=ACTIONS,
        help=f"action to perform. One of {'(' + '|'.join(ACTIONS) + ')'}",
        metavar="ACTION",
    )
    argparser.add_argument(
        "model",
        choices=MODELS,
        help=f"model and method to run. One of {'(' + '|'.join(MODELS) + ')'}",
        metavar="MODEL",
    )
    argparser.add_argument(
        "tag",
        choices=TAGS,
        help=f"part-of-speech tag to use. One of {'(' + '|'.join(TAGS) + ')'}",
        metavar="TAG",
    )
    argparser.add_argument(
        "--subset",
        dest="subset",
        type=int,
        default=None,
        help="number of data rows to use. Default using all data",
        metavar="INT",
    )
    argparser.add_argument(
        "--max-epochs",
        dest="max_epochs",
        nargs="+",
        type=int,
        default=[50],
        help="""maximum training iterations; for HMM, provide two numbers if using validation
        for the number of validation and validation iteration interval. Default to 50""",
        metavar="INT",
    )
    argparser.add_argument(
        "--load-path",
        dest="load_path",
        type=_valid_dir_or_file_path,
        default=None,
        help="path to load the model checkpoint from for testing.",
        metavar="PATH",
    )
    argparser.add_argument(
        "--save-path",
        dest="save_path",
        default=None,
        help="path to save the model to, `.pt` for HMM and `.pkl` for K-means.",
        metavar="PATH",
    )
    argparser.add_argument(
        "--res-path",
        dest="res_path",
        default=None,
        help="csv path to save the results to.",
        metavar="PATH",
    )
    argparser.add_argument(
        "--word-embedding-path",
        dest="word_embedding_path",
        default=None,
        help="path to load or save the BERT outputs for K-means method, use `.pt`.",
        metavar="PATH",
    )
    args = argparser.parse_args()
    args = vars(args)
    return args


def _valid_dir_or_file_path(path: str):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("file or directory not found")
    return os.path.abspath(path)
