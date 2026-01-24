# kmeans_pipeline.py
import csv
import logging
import os
import random
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from tqdm import tqdm

import pos_tagging.kmeans as kmeans
from preprocess_dataset import *  # load_ptb_dataset, wrap_dataset, create_tag_mapping
from utils import calculate_v_measure, calculate_variation_of_information

logger = logging.getLogger()


# -----------------------------
# Reproducibility helpers
# -----------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Centroid save/load helpers
# -----------------------------
def save_kmeans(
    clusterer: kmeans.KMeansPOSClusterer,
    save_path: str,
) -> None:
    if clusterer.centroids is None:
        raise ValueError("No centroids to save. Train k-means first.")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(clusterer.centroids.detach().cpu(), save_path)


def load_kmeans(
    clusterer: kmeans.KMeansPOSClusterer,
    load_path: str,
) -> None:
    centroids = torch.load(load_path, map_location=clusterer.device)
    if not isinstance(centroids, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in '{load_path}', got {type(centroids)}")
    clusterer.centroids = centroids.to(clusterer.device, dtype=torch.float32)


# -----------------------------
# Alignment safety
# -----------------------------
def _check_alignment(
    ds: Dataset,
    *,
    form_col: str = "form",
    tags_col: str = "tags",
    embeddings_col: str = "embeddings",
    split_name: str = "split",
) -> None:
    for i, ex in enumerate(ds):
        lf = len(ex[form_col])
        lt = len(ex[tags_col])
        le = len(ex[embeddings_col])
        if not (lf == lt == le):
            raise ValueError(
                f"[{split_name}] Length mismatch at row {i}: "
                f"len({form_col})={lf}, len({tags_col})={lt}, len({embeddings_col})={le}"
            )


# -----------------------------
# Training
# -----------------------------
def train_kmeans(
    dataset_splits: DatasetDict,
    *,
    K: int,
    num_iters: int,
    tol: float = 1e-4,
    save_path: Optional[str] = None,
    seed: int = 0,
    form_col: str = "form",
    tags_col: str = "tags",
    embeddings_col: str = "embeddings",
) -> kmeans.KMeansPOSClusterer:
    """
    Train a KMeansPOSClusterer on dataset_splits["train"] and optionally save centroids.
    Mirrors hmm_pipeline.train_hmm(...) pattern.
    """
    logger.info("Training k-means")
    _set_seed(seed)

    clusterer = kmeans.KMeansPOSClusterer()

    # Embed train split (and truncate tags to maintain alignment if truncation happens)
    train_ds = clusterer.embed_dataset(
        dataset_splits["train"],
        form_col=form_col,
        out_col=embeddings_col,
        truncate_other_token_cols=(tags_col,),
        show_progress=True,
    )
    _check_alignment(
        train_ds, form_col=form_col, tags_col=tags_col, embeddings_col=embeddings_col, split_name="train"
    )

    # Fit k-means (uses precomputed embeddings)
    clusterer.fit(
        train_ds,
        K=K,
        form_col=form_col,
        embeddings_col=embeddings_col,
        num_iters=num_iters,
        tol=tol,
        verbose=False,
        show_progress=True,
    )

    if save_path is not None:
        logger.info(f"Saving k-means centroids to {save_path}")
        save_kmeans(clusterer, save_path)
    else:
        logger.warning("No save path provided. K-means centroids not saved")

    logger.info("k-means training done")
    return clusterer


def train_kmeans_stage(
    dataset_splits: DatasetDict,
    *,
    K: int,
    max_epochs: Sequence[int],
    tol: float = 1e-4,
    save_path: Optional[str] = None,
    res_path: Optional[str] = None,
    seed: int = 0,
    form_col: str = "form",
    tags_col: str = "tags",
    embeddings_col: str = "embeddings",
) -> kmeans.KMeansPOSClusterer:
    """
    Stage-wise training for k-means, analogous to hmm_pipeline.train_hmm_stage(...),
    but with TRUE continuation across stages (warm-start from previous centroids).

    max_epochs:
      - [N, iters_per_stage]

    Each stage:
      - Stage 0: random init, run iters_per_stage Lloyd iterations
      - Stage i>0: warm-start from previous centroids, run iters_per_stage more iterations
      - Optionally save centroids with stage index inserted before file extension
      - Evaluate on a 5% test subset with stage-indexed CSV (like HMM)
    """
    if len(max_epochs) != 2:
        raise ValueError(
            "For staged k-means, max_epochs must be a sequence of length 2: [N, iters_per_stage]."
        )

    N = int(max_epochs[0])
    iters_per_stage = int(max_epochs[1])
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if iters_per_stage <= 0:
        raise ValueError(f"iters_per_stage must be positive, got {iters_per_stage}")

    logger.info("Training k-means by stages (continuation/warm-start)")
    _set_seed(seed)

    clusterer = kmeans.KMeansPOSClusterer()

    # Embed once and reuse across stages (fast + consistent)
    train_ds = clusterer.embed_dataset(
        dataset_splits["train"],
        form_col=form_col,
        out_col=embeddings_col,
        truncate_other_token_cols=(tags_col,),
        show_progress=True,
    )
    test_ds = clusterer.embed_dataset(
        dataset_splits["test"],
        form_col=form_col,
        out_col=embeddings_col,
        truncate_other_token_cols=(tags_col,),
        show_progress=True,
    )

    _check_alignment(
        train_ds,
        form_col=form_col,
        tags_col=tags_col,
        embeddings_col=embeddings_col,
        split_name="train",
    )
    _check_alignment(
        test_ds,
        form_col=form_col,
        tags_col=tags_col,
        embeddings_col=embeddings_col,
        split_name="test",
    )

    for i in tqdm(range(N), desc="Outer train loop", total=N):
        # Stage 0 uses random init; later stages warm-start from existing centroids.
        continue_training = i > 0

        # For deterministic behavior:
        # - stage 0 seeding affects random init
        # - later stages are deterministic given centroids + X, but we keep seeding anyway.
        _set_seed(seed + i)

        clusterer.fit(
            train_ds,
            K=K,
            form_col=form_col,
            embeddings_col=embeddings_col,
            num_iters=iters_per_stage,
            tol=tol,
            verbose=False,
            show_progress=True,
            continue_training=continue_training,
        )

        if save_path is not None:
            parts = save_path.split(".")
            parts.insert(-1, f"{i}")
            t_save_path = ".".join(parts)
            logger.info(f"Saving k-means centroids to {t_save_path}")
            save_kmeans(clusterer, t_save_path)
        else:
            logger.warning("No save path provided. K-means centroids not saved")

        if res_path is not None:
            parts = res_path.split(".")
            parts.insert(-1, f"{i}")
            t_res_path = ".".join(parts)

            # Match HMM: evaluate on 5% subset
            n_eval = max(1, round(len(test_ds) * 0.05)) if len(test_ds) > 0 else 0
            eval_split = test_ds.select(range(n_eval)) if n_eval > 0 else test_ds

            eval_kmeans(
                eval_split,
                kmeans_clusterer=clusterer,
                res_path=t_res_path,
            )

    logger.info("k-means staged training done")
    return clusterer



# -----------------------------
# Evaluation
# -----------------------------
def eval_kmeans(
    dataset_split: Dataset,
    kmeans_clusterer: kmeans.KMeansPOSClusterer = None,
    load_path: str = None,
    res_path: str = "kmeans_result.csv",
    *,
    form_col: str = "form",
    tags_col: str = "tags",
    embeddings_col: str = "embeddings",
):
    """
    Evaluate k-means clustering quality using V-measure and VI.
    Mirrors hmm_pipeline.eval_hmm(...) signature/pattern.

    Requires either:
      - kmeans_clusterer with centroids set
      - or load_path pointing to saved centroids
    """
    if kmeans_clusterer is None:
        if load_path is None:
            raise ValueError("At least one of kmeans_clusterer and load_path should be provided")
        kmeans_clusterer = kmeans.KMeansPOSClusterer()
        logger.info(f"Loading k-means centroids from {load_path}")
        load_kmeans(kmeans_clusterer, load_path)

    # If embeddings are missing, compute them (keeps eval robust)
    if embeddings_col not in dataset_split.column_names:
        logger.info("Embeddings column missing in eval split; embedding now")
        dataset_split = kmeans_clusterer.embed_dataset(
            dataset_split,
            form_col=form_col,
            out_col=embeddings_col,
            truncate_other_token_cols=(tags_col,),
            show_progress=True,
        )

    _check_alignment(dataset_split, form_col=form_col, tags_col=tags_col, embeddings_col=embeddings_col, split_name="eval")

    num_samples = len(dataset_split)
    results = []

    true_labels_all = []
    pred_labels_all = []

    for i, example in enumerate(tqdm(dataset_split, "KMeans testing", num_samples)):
        forms = example[form_col]
        true_tags = example[tags_col]  # list[int]
        embs_list = example[embeddings_col]  # nested list [T, D]

        # Create tensor on device
        embs = torch.tensor(embs_list, device=kmeans_clusterer.device, dtype=torch.float32)

        pred_clusters = kmeans_clusterer.predict_sentence(embs)  # [T] on CPU
        pred_tags = pred_clusters.tolist()

        sentence = " ".join(forms)

        homo, comp, v_score = calculate_v_measure(true_tags, pred_tags)
        vi, norm_vi = calculate_variation_of_information(true_tags, pred_tags)

        results.append([i + 1, sentence, vi, norm_vi, homo, comp, v_score])

        true_labels_all.extend(true_tags)
        pred_labels_all.extend(pred_tags)

    homo_whole, comp_whole, v_whole = calculate_v_measure(true_labels_all, pred_labels_all)
    vi_whole, norm_vi_whole = calculate_variation_of_information(true_labels_all, pred_labels_all)

    print(
        f"| Homogeneity score: {homo_whole}\n"
        f"| Completeness score: {comp_whole}\n"
        f"| V-measure: {v_whole}\n"
        f"| Variation of information: {vi_whole}\n"
        f"| Normalized VI: {norm_vi_whole}\n"
    )

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
        writer.writerow([0, "-", vi_whole, norm_vi_whole, homo_whole, comp_whole, v_whole])
        writer.writerows(results)


# -----------------------------
# Top-level wrappers (analogous to HMM)
# -----------------------------
def train_and_test(
    tag_name,
    subset,
    max_epochs,
    load_path,
    save_path,
    res_path,
    *,
    seed: int = 0,
    tol: float = 1e-4,
):
    """
    Wrapper analogous to hmm_pipeline.train_and_test(...).

    - method: accepted for signature compatibility (unused by k-means).
    - tag_name: "upos" or "xpos"
    - subset: passed to load_ptb_dataset(line_num=subset)
    - max_epochs: [iters] for single run OR [N, iters_per_stage] for staged runs
    - load_path: optional path to load centroids for evaluation (used if you want to eval a saved model)
    - save_path: where to save centroids (single run), or base path for staged saves
    - res_path: output CSV path (single run), or base path for staged eval outputs
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

    def map_tag(examples):
        examples["tags"] = [tag_mapping[t] for t in examples[tag_name]]
        return examples

    dataset = dataset.map(map_tag, desc="Mapping tags")
    dataset_splits = DatasetDict({"train": dataset, "test": dataset})

    # Mirror HMM choice: number of clusters = number of distinct gold tags
    K = len(tag_mapping)

    with torch.no_grad():
        if len(max_epochs) == 1:
            clusterer = train_kmeans(
                dataset_splits,
                K=K,
                num_iters=int(max_epochs[0]),
                tol=tol,
                save_path=save_path,
                seed=seed,
            )
        else:
            clusterer = train_kmeans_stage(
                dataset_splits,
                K=K,
                max_epochs=max_epochs,
                tol=tol,
                save_path=save_path,
                res_path=res_path,
                seed=seed,
            )

        # Always do a final eval on the full test set (like HMM does),
        # using the trained clusterer. If you want to evaluate a saved centroid file,
        # call `test(...)` instead.
        eval_kmeans(
            dataset_splits["test"],
            kmeans_clusterer=clusterer,
            load_path=load_path,
            res_path=res_path,
        )


def test(
    tag_name,
    subset,
    load_path,
    res_path,
    *,
    seed: int = 0,
):
    """
    Test-only wrapper analogous to hmm_pipeline.test(...).
    Loads centroids from load_path and evaluates on the test dataset.
    """
    logger.warning(f"Using {tag_name} as tag")

    if load_path is None:
        raise ValueError("load_path must be provided for k-means test().")

    _set_seed(seed)

    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set),
    }[tag_name]

    def map_tag(examples):
        examples["tags"] = [tag_mapping[t] for t in examples[tag_name]]
        return examples

    dataset = dataset.map(map_tag, desc="Mapping tags")
    dataset_splits = DatasetDict({"train": dataset, "test": dataset})

    with torch.no_grad():
        eval_kmeans(
            dataset_splits["test"],
            kmeans_clusterer=None,
            load_path=load_path,
            res_path=res_path,
        )
