# nhmm_pipeline.py

import csv
import logging

import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from pos_tagging.neural_hmm import NeuralHMMClassifier
from preprocess_dataset import (
    load_ptb_dataset,
    wrap_dataset,
    create_tag_mapping,
    create_obs_mapping,
)
from utils import calculate_v_measure, calculate_variation_of_information

logger = logging.getLogger()


def train_nhmm(
    dataset_splits: DatasetDict,
    max_epochs: int,
    num_states: int,
    vocab_size: int,
    lr: float = 1e-3,
    device: str | torch.device = None,
    save_path: str | None = None,
) -> NeuralHMMClassifier:
    """
    Train Neural HMM (NeuralHMMClassifier) with gradient descent on unsupervised NLL.

    Args:
        dataset_splits: DatasetDict with "train" split; each example has "input_ids".
        max_epochs:     number of passes over the training data.
        num_states:     number of hidden states K.
        vocab_size:     vocabulary size V (number of distinct observations).
        lr:             learning rate for Adam.
        device:         "cuda" or "cpu". If None, auto-detect.
        save_path:      optional path to save the trained model (torch.save).

    Returns:
        nhmm: trained NeuralHMMClassifier instance.
    """
    logger.info("Training Neural HMM")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    nhmm = NeuralHMMClassifier(
        num_states=num_states,
        vocab_size=vocab_size,
    ).to(device)

    optimizer = torch.optim.Adam(nhmm.parameters(), lr=lr)

    train_split: Dataset = dataset_splits["train"]
    num_samples = len(train_split)

    for epoch in range(1, max_epochs + 1):
        nhmm.train()
        total_loss = 0.0

        for example in tqdm(train_split, desc=f"NHMM training epoch {epoch}", total=num_samples):
            input_ids = torch.tensor(example["input_ids"], dtype=torch.long, device=device)
            # Use batch size 1 to avoid padding / variable length complications
            batch_emissions = input_ids.unsqueeze(0)  # (1, T)

            optimizer.zero_grad()
            loss = nhmm(batch_emissions)  
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_samples
        logger.info(f"[Epoch {epoch}] avg NLL per sentence: {avg_loss:.4f}")

    if save_path is not None:
        logger.info(f"Saving Neural HMM model to {save_path}")
        torch.save(nhmm, save_path)
    else:
        logger.warning("No save path provided. Neural HMM model not saved")

    logger.info("Neural HMM training done")
    return nhmm


def train_nhmm_stage(
    dataset_splits: DatasetDict,
    max_epochs: list[int],
    num_states: int,
    vocab_size: int,
    lr: float = 1e-3,
    device: str | torch.device = None,
    save_path: str | None = None,
    res_path: str | None = None,
) -> NeuralHMMClassifier:
    """
    Train a Neural HMM classifier in multiple stages.

    This function performs stage-wise training analogous to `train_hmm_stage`.
    Training proceeds in an outer loop of stages, where each stage consists of
    several epochs of gradient-based optimization. After each stage, the model
    can be checkpointed and evaluated on a small subset of the test data.

    Args:
        dataset_splits (DatasetDict): Dataset dictionary containing `"train"`
            and `"test"` splits.
        max_epochs (list[int]): A list `[N_outer, inner_epochs]` where `N_outer`
            is the number of training stages and `inner_epochs` is the number
            of epochs per stage.
        num_states (int): Number of hidden states.
        vocab_size (int): Size of the observation vocabulary.
        lr (float, optional): Learning rate for the Adam optimizer.
            Defaults to 1e-3.
        device (str or torch.device, optional): Device on which to run training.
            If `None`, uses CUDA if available.
        save_path (str, optional): Base path for saving model checkpoints.
            The stage index is inserted before the file extension.
        res_path (str, optional): Base path for saving evaluation results.
            The stage index is inserted before the file extension.

    Returns:
        nhmm (NeuralHMMClassifier): The trained Neural HMM classifier after the
        final training stage.
    """
    assert len(max_epochs) == 2, "max_epochs should be [N_outer, inner_epochs]"

    logger.info("Training Neural HMM by stages")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    nhmm = NeuralHMMClassifier(
        num_states=num_states,
        vocab_size=vocab_size,
    ).to(device)

    optimizer = torch.optim.Adam(nhmm.parameters(), lr=lr)

    train_split: Dataset = dataset_splits["train"]
    num_samples = len(train_split)

    N_outer, inner_epochs = max_epochs
    for outer_idx in tqdm(range(N_outer), desc="Outer NHMM train loop", total=N_outer):
        for epoch in range(1, inner_epochs + 1):
            nhmm.train()
            total_loss = 0.0

            for example in tqdm(
                train_split,
                desc=f"NHMM training outer {outer_idx}, epoch {epoch}",
                total=num_samples,
                leave=False,
            ):
                input_ids = torch.tensor(example["input_ids"], dtype=torch.long, device=device)
                batch_emissions = input_ids.unsqueeze(0)  # (1, T)

                optimizer.zero_grad()
                loss = nhmm(batch_emissions)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / num_samples
            logger.info(
                f"[Outer {outer_idx}, epoch {epoch}] avg NLL per sentence: {avg_loss:.4f}"
            )

        # After each outer loop: save checkpoint & evaluate on a small test subset
        if save_path is not None:
            t_path = save_path.split(".")
            t_path.insert(-1, f"{outer_idx}")
            t_path = ".".join(t_path)
            logger.info(f"Saving Neural HMM model to {t_path}")
            torch.save(nhmm, t_path)
        else:
            logger.warning("No save path provided. Neural HMM model not saved")

        if res_path is not None:
            t_path = res_path.split(".")
            t_path.insert(-1, f"{outer_idx}")
            t_path = ".".join(t_path)
            # Use 5% of test set for quick evaluation
            sub_test = dataset_splits["test"].select(
                range(round(len(dataset_splits["test"]) * 0.05))
            )
            eval_nhmm(
                dataset_split=sub_test,
                nhmm=nhmm,
                res_path=t_path,
                device=device,
            )

    logger.info("Neural HMM staged training done")
    return nhmm


def eval_nhmm(
    dataset_split: Dataset,
    nhmm: NeuralHMMClassifier | None = None,
    load_path: str | None = None,
    res_path: str = "nhmm_result.csv",
    device: str | torch.device | None = None,
):
    """
    Evaluate a Neural HMM classifier on a dataset split.

    This function runs inference with a trained `NeuralHMMClassifier`,
    compares predicted state sequences to gold tags, and evaluates clustering
    quality using homogeneity, completeness, V-measure, and variation of
    information (VI). Per-example and whole-dataset results are saved to CSV.

    Args:
        dataset_split (Dataset): Dataset split to evaluate.
        nhmm (NeuralHMMClassifier, optional): Trained Neural HMM model. If not
            provided, `load_path` must be specified.
        load_path (str, optional): Path to a saved Neural HMM model to load.
        res_path (str, optional): Path to the CSV file where evaluation results
            will be written. Defaults to `"nhmm_result.csv"`.
        device (str or torch.device, optional): Device on which to run inference.
            If `None`, uses CUDA if available.

    Raises:
        ValueError: If neither `nhmm` nor `load_path` is provided.
    """
    if nhmm is None:
        if load_path is None:
            raise ValueError("At least one of nhmm model and load_path should be provided")
        logger.info(f"Loading Neural HMM model from {load_path}")
        nhmm: NeuralHMMClassifier = torch.load(load_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    nhmm.to(device)
    nhmm.eval()

    num_samples = len(dataset_split)
    results = []
    homo_sum = 0.0
    comp_sum = 0.0
    v_score_sum = 0.0
    vi_sum = 0.0
    normalized_vi_sum = 0.0
    true_labels = torch.tensor([], dtype=torch.long)
    pred_labels = torch.tensor([], dtype=torch.long)

    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset_split, desc="NHMM testing", total=num_samples)):
            input_ids_list = example["input_ids"]
            forms = example["form"]
            true_tags = example["tags"]  

            input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
            pred_tags = nhmm.inference(input_ids).cpu()  

            if isinstance(pred_tags, torch.Tensor):
                pred_tags_list = pred_tags.tolist()
            else:
                pred_tags_list = list(pred_tags)

            sentence = " ".join(forms)

            # Compute per-example V-measure and VI
            homo_score, comp_score, v_score = calculate_v_measure(true_tags, pred_tags_list)
            vi, normalized_vi = calculate_variation_of_information(true_tags, pred_tags_list)

            homo_sum += homo_score
            comp_sum += comp_score
            v_score_sum += v_score
            vi_sum += vi
            normalized_vi_sum += normalized_vi
            results.append(
                [i + 1, sentence, vi, normalized_vi, homo_score, comp_score, v_score]
            )

            # Record for whole-dataset V-measure and VI
            true_labels = torch.hstack([true_labels, torch.tensor(true_tags, dtype=torch.long)])
            pred_labels = torch.hstack([pred_labels, torch.tensor(pred_tags_list, dtype=torch.long)])

    logger.info("Computing whole-dataset V-measure (NHMM)")
    homo_score_whole, comp_score_whole, v_score_whole = calculate_v_measure(
        true_labels.tolist(), pred_labels.tolist()
    )

    logger.info("Computing whole-dataset VI (NHMM)")
    vi_whole, normalized_vi_whole = calculate_variation_of_information(
        true_labels.tolist(), pred_labels.tolist()
    )

    print(
        f"[NHMM]\n"
        f"| Homogeneity score: {homo_score_whole}\n"
        f"| Completeness score: {comp_score_whole}\n"
        f"| V-measure: {v_score_whole}\n"
        f"| Variation of information: {vi_whole}\n"
        f"| Normalized VI: {normalized_vi_whole}\n"
    )

    # Save results to CSV
    logger.info(f"Saving NHMM results to {res_path}")
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
        # Whole-dataset results as id==0
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
        # Per-example results
        writer.writerows(results)


def train_and_test(
    tag_name: str,
    subset: int | None,
    max_epochs: list[int],
    load_path: str | None,
    save_path: str | None,
    res_path: str,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
):
    """
    Train and evaluate a Neural HMM on the Penn Treebank dataset.

    This function mirrors the HMM and K-means training-and-evaluation pipelines,
    but uses a Neural HMM trained with gradient-based optimization. It loads and
    preprocesses the dataset, trains the model (single-stage or staged), and
    evaluates it using clustering-based metrics.

    Args:
        tag_name (str): Tag set to use for supervision and evaluation
            ("upos" or "xpos").
        subset (int or None): Number of sentences to load from the dataset.
            If `None`, the full dataset is used.
        max_epochs (list[int]): Either `[epochs]` for single-stage training or
            `[N_outer, inner_epochs]` for staged training.
        load_path (str or None): Path to a saved NHMM model to load for
            evaluation. If provided, the trained model is not used for eval.
        save_path (str or None): Path to save the trained NHMM model or base
            path for staged checkpoints.
        res_path (str): Path to save evaluation results.
        lr (float, optional): Learning rate for training. Defaults to 1e-3.
        device (str or torch.device, optional): Device on which to run training
            and evaluation. If `None`, uses CUDA if available.
    """
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
        input_ids = [obs_mapping[token] for token in examples["form"]]
        examples["input_ids"] = input_ids

        tags = [tag_mapping[tag] for tag in examples[tag_name]]
        examples["tags"] = tags
        return examples

    dataset = dataset.map(map_tag_and_token, desc="Mapping tokens and tags (NHMM)")
    dataset_splits = DatasetDict({"train": dataset, "test": dataset})

    num_states = len(tag_mapping)
    vocab_size = len(obs_mapping)

    if len(max_epochs) == 1:
        nhmm = train_nhmm(
            dataset_splits=dataset_splits,
            max_epochs=max_epochs[0],
            num_states=num_states,
            vocab_size=vocab_size,
            lr=lr,
            device=device,
            save_path=save_path,
        )
    else:
        nhmm = train_nhmm_stage(
            dataset_splits=dataset_splits,
            max_epochs=max_epochs,
            num_states=num_states,
            vocab_size=vocab_size,
            lr=lr,
            device=device,
            save_path=save_path,
            res_path=res_path,
        )

    # Full evaluation (no_grad)
    eval_nhmm(
        dataset_split=dataset_splits["test"],
        nhmm=nhmm if load_path is None else None,
        load_path=load_path,
        res_path=res_path,
        device=device,
    )


def test(
    tag_name: str,
    subset: int | None,
    load_path: str,
    res_path: str,
    device: str | torch.device | None = None,
):
    """
    Evaluate a pretrained Neural HMM on the Penn Treebank dataset.

    This is a test-only wrapper that loads a saved Neural HMM model and evaluates
    it on the dataset using clustering-based metrics.

    Args:
        tag_name (str): Tag set to use for evaluation ("upos" or "xpos").
        subset (int or None): Number of sentences to load from the dataset.
            If `None`, the full dataset is used.
        load_path (str): Path to a saved Neural HMM model.
        res_path (str): Path to save evaluation results.
        device (str or torch.device, optional): Device on which to run inference.
            If `None`, uses CUDA if available.
    """
    logger.warning(f"Using {tag_name} as tag")
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set),
    }[tag_name]
    obs_mapping = create_obs_mapping(sentences)

    def map_tag_and_token(examples):
        input_ids = [obs_mapping[token] for token in examples["form"]]
        examples["input_ids"] = input_ids

        tags = [tag_mapping[tag] for tag in examples[tag_name]]
        examples["tags"] = tags
        return examples

    dataset = dataset.map(map_tag_and_token, desc="Mapping tokens and tags (NHMM)")
    dataset_splits = DatasetDict({"train": dataset, "test": dataset})

    eval_nhmm(
        dataset_split=dataset_splits["test"],
        load_path=load_path,
        res_path=res_path,
        device=device,
    )
