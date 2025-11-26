import csv
import logging

import torch
from datasets import DatasetDict
from tqdm import tqdm

from pos_tagging.hmm import HMMClassifier
from preprocess_dataset import *
from utils import calculate_v_measure, calculate_variation_of_information

logger = logging.getLogger()


def train_hmm(
    method: str,
    dataset_splits: DatasetDict,
    max_epochs,
    num_states: int,
    num_obs: int,
    save_path: str = None,
):
    logger.info("Training HMM")
    hmm = HMMClassifier(num_states=num_states, num_obs=num_obs)
    # Training
    hmm.train(dataset_splits["train"], max_epochs, method=method)

    if save_path is not None:
        # Save HMM parameters
        logger.info(f"Saving HMM model to {save_path}")
        torch.save(hmm, save_path)
    else:
        logger.warning("No save path provided. HMM model not saved")

    logger.info("HMM training done")

    return hmm


def train_hmm_stage(
    method: str,
    dataset_splits: DatasetDict,
    max_epochs,
    num_states: int,
    num_obs: int,
    save_path: str = None,
    res_path: str = None,
):
    logger.info("Training HMM by stages")
    hmm = HMMClassifier(num_states=num_states, num_obs=num_obs)
    # Training
    f = False
    N = max_epochs[0]
    for i in tqdm(range(N), "Outer train loop", N):
        hmm.train(
            dataset_splits["train"], max_epochs[1], method=method, continue_training=f
        )
        f = True

        if save_path is not None:
            # Save HMM parameters
            t_path = save_path.split(".")
            t_path.insert(-1, f"{i}")
            t_path = ".".join(t_path)
            logger.info(f"Saving HMM model to {t_path}")
            torch.save(hmm, t_path)
        else:
            logger.warning("No save path provided. HMM model not saved")

        t_path = res_path.split(".")
        t_path.insert(-1, f"{i}")
        t_path = ".".join(t_path)
        eval_hmm(
            dataset_splits["test"].select(
                range(round(len(dataset_splits["test"]) * 0.05))
            ),
            hmm=hmm,
            res_path=t_path,
        )
    logger.info("HMM training done")

    return hmm


def eval_hmm(
    dataset_split: Dataset,
    hmm: HMMClassifier = None,
    load_path: str = None,
    res_path: str = "hmm_result.csv",
):
    if hmm is None:
        if load_path is None:
            raise ValueError(
                "At least one of HMM model and load_path should be provided"
            )
        # Load HMM parameters
        logger.info(f"Loading HMM model from {load_path}")
        hmm: HMMClassifier = torch.load(load_path)

    # Evaluate
    num_samples = len(dataset_split)
    results = []
    homo_sum = 0.0
    comp_sum = 0.0
    v_score_sum = 0.0
    vi_sum = 0.0
    normalized_vi_sum = 0.0
    true_labels = torch.tensor([])
    pred_labels = torch.tensor([])
    for i, example in enumerate(tqdm(dataset_split, "HMM testing", num_samples)):
        input_ids = example["input_ids"]
        forms = example["form"]
        true_tags = example["tags"]
        pred_tags = hmm.inference(input_ids)
        # TODO: what tokenizer is used to get ptb-train.conllu???
        sentence = " ".join(forms)

        # Compute per-example V-measure and VI
        homo_score, comp_score, v_score = calculate_v_measure(true_tags, pred_tags)
        vi, normalized_vi = calculate_variation_of_information(true_tags, pred_tags)

        homo_sum += homo_score
        comp_sum += comp_score
        v_score_sum += v_score
        vi_sum += vi
        normalized_vi_sum += normalized_vi
        results.append(
            [i + 1, sentence, vi, normalized_vi, homo_score, comp_score, v_score]
        )

        # Record true and predicted labels for computing whole-dataset V-measure and VI
        true_labels = torch.hstack([true_labels, torch.tensor(true_tags)])
        pred_labels = torch.hstack([pred_labels, torch.tensor(pred_tags)])

    # Compute whole-dataset V-measure and VI
    logger.info("Computing whole-dataset V-measure")
    homo_score_whole, comp_score_whole, v_score_whole = calculate_v_measure(
        true_labels.tolist(), pred_labels.tolist()
    )
    logger.info("Computing whole-dataset VI")
    vi_whole, normalized_vi_whole = calculate_variation_of_information(
        true_labels.tolist(), pred_labels.tolist()
    )

    print(
        f"| Homogeneity score: {homo_score_whole}\n"
        f"| Completeness score: {comp_score_whole}\n"
        f"| V-measure: {v_score_whole}\n"
        f"| Variation of information: {vi_whole}\n"
        f"| Normalized VI: {normalized_vi_whole}\n"
    )

    # Save results to CSV
    logger.info(f"Saving results to {res_path}")
    with open(res_path, "w+", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "sentence",
                "VI",
                "normalized-VI",
                "homogeneity",
                "completeness",
                "V-score",
            ]
        )
        # Save whole-dataset results as id==0
        writer.writerow(
            [
                0,
                "-",
                vi_whole,
                normalized_vi_whole,
                homo_score_whole,
                comp_score_whole,
                v_score_whole,
            ]
        )
        # Save per-example results
        writer.writerows(results)


# TODO: HMM inference for single sentence


def train_and_test(
    method,
    tag_name,
    subset,
    max_epochs,
    load_path,
    save_path,
    res_path,
):
    assert len(max_epochs) <= 2
    logger.warning(f"Using {tag_name} as tag")
    # Load and wrap PTB dataset
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set),
    }[tag_name]
    obs_mapping = create_obs_mapping(sentences)

    def map_tag_and_token(examples):
        input_ids = []
        for token in examples["form"]:
            input_ids.append(obs_mapping[token])
        examples["input_ids"] = input_ids
        # Using UPoS as tags
        tags = []
        for tag in examples[tag_name]:
            tags.append(tag_mapping[tag])
        examples["tags"] = tags
        return examples

    dataset = dataset.map(map_tag_and_token, desc="Mapping tokens and tags")

    dataset_splits = DatasetDict({"train": dataset, "test": dataset})

    with torch.no_grad():
        if len(max_epochs) == 1:
            hmm = train_hmm(
                method,
                dataset_splits,
                max_epochs[0],
                num_states=len(tag_mapping),
                num_obs=len(obs_mapping),
                save_path=save_path,
            )
        else:
            hmm = train_hmm_stage(
                method,
                dataset_splits,
                max_epochs,
                num_states=len(tag_mapping),
                num_obs=len(obs_mapping),
                save_path=save_path,
                res_path=res_path,
            )

        eval_hmm(
            dataset_splits["test"],
            hmm,
            load_path=load_path,
            res_path=res_path,
        )


def test(
    tag_name,
    subset,
    load_path,
    res_path,
):
    logger.warning(f"Using {tag_name} as tag")
    # Load and wrap PTB dataset
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set),
    }[tag_name]
    obs_mapping = create_obs_mapping(sentences)

    def map_tag_and_token(examples):
        input_ids = []
        for token in examples["form"]:
            input_ids.append(obs_mapping[token])
        examples["input_ids"] = input_ids
        # Using UPoS as tags
        tags = []
        for tag in examples[tag_name]:
            tags.append(tag_mapping[tag])
        examples["tags"] = tags
        return examples

    dataset = dataset.map(map_tag_and_token, desc="Mapping tokens and tags")

    dataset_splits = DatasetDict({"train": dataset, "test": dataset})

    with torch.no_grad():
        eval_hmm(
            dataset_splits["test"],
            load_path=load_path,
            res_path=res_path,
        )
