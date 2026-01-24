import csv
from tqdm import tqdm
from utils import calculate_v_measure, calculate_variation_of_information
import torch
import pos_tagging.kmeans as kmeans
import logging
import os
from typing import Optional
from datasets import DatasetDict
from preprocess_dataset import *  # load_ptb_dataset, wrap_dataset, create_tag_mapping
from utils import calculate_v_measure, calculate_variation_of_information

logger = logging.getLogger()

def _save_centroids(clusterer: kmeans.KMeansPOSClusterer, save_path: str) -> None:
    if clusterer.centroids is None:
        raise ValueError("No centroids to save (clusterer.centroids is None).")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(clusterer.centroids.detach().cpu(), save_path)


def _load_centroids(clusterer: kmeans.KMeansPOSClusterer, load_path: str) -> None:
    centroids = torch.load(load_path, map_location=clusterer.device)
    if not isinstance(centroids, torch.Tensor):
        raise TypeError(f"Loaded centroids must be a torch.Tensor, got {type(centroids)}")
    clusterer.centroids = centroids.to(clusterer.device, dtype=torch.float32)


def train_and_test_old(
    K: int,
    tag_name: str,
    subset,
    load_path: Optional[str],
    save_path: Optional[str],
    res_path: str,
    *,
    num_iters: int = 20,
    tol: float = 1e-4,
):
    """
    KMeans pipeline mirroring hmm_pipeline.train_and_test(...) patterns.

    Args:
        K: number of clusters.
        tag_name: "upos" or "xpos" (same as HMM pipeline).
        subset: passed through to load_ptb_dataset(line_num=subset).
        load_path: optional path to load centroids instead of training.
        save_path: optional path to save centroids after training.
        res_path: CSV output path (same format style as eval_hmm).
        num_iters, tol: k-means hyperparameters.
    """
    logger.warning(f"Using {tag_name} as tag")

    # Load and wrap PTB dataset (same as HMM pipeline)
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set),
    }[tag_name]

    def map_tag(examples):
        # Using UPoS/XPoS as tags (mapped to ints)
        tags = []
        for tag in examples[tag_name]:
            tags.append(tag_mapping[tag])
        examples["tags"] = tags
        return examples

    dataset = dataset.map(map_tag, desc="Mapping tags")

    # Match your HMM pattern (train == test unless you change it later)
    dataset_splits = DatasetDict({"train": dataset, "test": dataset})

    # Create clusterer
    clusterer = kmeans.KMeansPOSClusterer()

    with torch.no_grad():
        # ---------------------------------------------------------------------
        # Embed train/test (adds "embeddings" and ensures token-level columns align)
        # ---------------------------------------------------------------------
        logger.info("Embedding train split for k-means")
        train_ds = clusterer.embed_dataset(
            dataset_splits["train"],
            form_col="form",
            out_col="embeddings",
            truncate_other_token_cols=("tags",),
            show_progress=True,
        )

        logger.info("Embedding test split for k-means")
        test_ds = clusterer.embed_dataset(
            dataset_splits["test"],
            form_col="form",
            out_col="embeddings",
            truncate_other_token_cols=("tags",),
            show_progress=True,
        )

        # Basic alignment sanity checks (critical for correctness)
        def _check_alignment(split, split_name: str):
            for i, ex in enumerate(split):
                lf = len(ex["form"])
                lt = len(ex["tags"])
                le = len(ex["embeddings"])
                if not (lf == lt == le):
                    raise ValueError(
                        f"[{split_name}] Length mismatch at row {i}: "
                        f"len(form)={lf}, len(tags)={lt}, len(embeddings)={le}"
                    )

        _check_alignment(train_ds, "train")
        _check_alignment(test_ds, "test")

        # ---------------------------------------------------------------------
        # Train or load centroids
        # ---------------------------------------------------------------------
        if load_path is not None:
            logger.info(f"Loading k-means centroids from {load_path}")
            _load_centroids(clusterer, load_path)
        else:
            logger.info("Training k-means centroids")
            clusterer.fit(
                train_ds,
                K=K,
                form_col="form",
                embeddings_col="embeddings",  # use cached embeddings we just computed
                num_iters=num_iters,
                tol=tol,
                verbose=False,
                show_progress=True,
            )

            if save_path is not None:
                logger.info(f"Saving k-means centroids to {save_path}")
                _save_centroids(clusterer, save_path)
            else:
                logger.warning("No save path provided. K-means centroids not saved")

        logger.info("Evaluating k-means on test split")
        eval_kmeans(
            test_ds,
            kmeans_clusterer=clusterer,
            res_path=res_path,
        )


def test(
    K: int,
    tag_name: str,
    subset,
    load_path: str,
    res_path: str,
):
    """
    KMeans test-only pipeline mirroring hmm_pipeline.test(...) patterns.
    Requires load_path for centroids.
    """
    logger.warning(f"Using {tag_name} as tag")

    if load_path is None:
        raise ValueError("load_path must be provided for k-means test().")

    # Load and wrap PTB dataset
    sentences, upos_set, xpos_set = load_ptb_dataset(line_num=subset)
    dataset = wrap_dataset(sentences)

    tag_mapping = {
        "upos": create_tag_mapping(upos_set),
        "xpos": create_tag_mapping(xpos_set),
    }[tag_name]

    def map_tag(examples):
        tags = []
        for tag in examples[tag_name]:
            tags.append(tag_mapping[tag])
        examples["tags"] = tags
        return examples

    dataset = dataset.map(map_tag, desc="Mapping tags")
    dataset_splits = DatasetDict({"train": dataset, "test": dataset})

    clusterer = kmeans.KMeansPOSClusterer()

    with torch.no_grad():
        # Embed test
        logger.info("Embedding test split for k-means")
        test_ds = clusterer.embed_dataset(
            dataset_splits["test"],
            form_col="form",
            out_col="embeddings",
            truncate_other_token_cols=("tags",),
            show_progress=True,
        )

        # Alignment check
        for i, ex in enumerate(test_ds):
            lf, lt, le = len(ex["form"]), len(ex["tags"]), len(ex["embeddings"])
            if not (lf == lt == le):
                raise ValueError(
                    f"[test] Length mismatch at row {i}: len(form)={lf}, len(tags)={lt}, len(embeddings)={le}"
                )

        # Load centroids + eval
        logger.info(f"Loading k-means centroids from {load_path}")
        _load_centroids(clusterer, load_path)

        logger.info("Evaluating k-means on test split")
        eval_kmeans(
            test_ds,
            kmeans_clusterer=clusterer,
            res_path=res_path,
        )

def eval_kmeans(
    dataset_split,
    kmeans_clusterer: kmeans.KMeansPOSClusterer,
    res_path: str = "kmeans_result.csv",
):
    device = kmeans_clusterer.device

    num_samples = len(dataset_split)
    results = []
    homo_sum = 0.0
    comp_sum = 0.0
    v_score_sum = 0.0
    vi_sum = 0.0
    normalized_vi_sum = 0.0
    true_labels_all = torch.tensor([], dtype=torch.long)
    pred_labels_all = torch.tensor([], dtype=torch.long)

    for i, example in enumerate(tqdm(dataset_split, "KMeans testing", num_samples)):
        forms = example["form"]
        true_tags = example["tags"]              # list[int], shape [T]
        embs = torch.tensor(example["embeddings"],
                            device=device,
                            dtype=torch.float32)  # [T, D]

        # Predict cluster per token
        pred_clusters = kmeans_clusterer.predict_sentence(embs)  # [T]
        pred_tags = pred_clusters.cpu().tolist()                 # cluster IDs

        sentence = " ".join(forms)

        # Per-example metrics
        homo, comp, v_score = calculate_v_measure(true_tags, pred_tags)
        vi, norm_vi = calculate_variation_of_information(true_tags, pred_tags)

        homo_sum += homo
        comp_sum += comp
        v_score_sum += v_score
        vi_sum += vi
        normalized_vi_sum += norm_vi

        results.append(
            [i + 1, sentence, vi, norm_vi, homo, comp, v_score]
        )

        # Aggregate labels
        true_labels_all = torch.hstack([true_labels_all, torch.tensor(true_tags)])
        pred_labels_all = torch.hstack([pred_labels_all, pred_clusters.cpu()])

    # Whole-dataset V-measure and VI
    homo_whole, comp_whole, v_whole = calculate_v_measure(
        true_labels_all.tolist(), pred_labels_all.tolist()
    )
    vi_whole, norm_vi_whole = calculate_variation_of_information(
        true_labels_all.tolist(), pred_labels_all.tolist()
    )

    print(
        f"| Homogeneity score: {homo_whole}\n"
        f"| Completeness score: {comp_whole}\n"
        f"| V-measure: {v_whole}\n"
        f"| Variation of information: {vi_whole}\n"
        f"| Normalized VI: {norm_vi_whole}\n"
    )

    # Save to CSV (same format as eval_hmm)
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
        writer.writerow(
            [
                0,
                "-",
                vi_whole,
                norm_vi_whole,
                homo_whole,
                comp_whole,
                v_whole,
            ]
        )
        writer.writerows(results)

def train_and_test_old(
    tag_name,
    subset,
    max_epochs,
    load_path,
    save_path,
    res_path,
):
    assert len(max_epochs) <= 2
    logger.warning(f"Using {tag_name} as tag")

    # -------------------------
    # Normalize max_epochs
    # -------------------------
    if len(max_epochs) == 1:
        N_stages = 1
        iters_per_stage = max_epochs[0]
    else:
        N_stages, iters_per_stage = max_epochs

    # -------------------------
    # Load and wrap dataset
    # -------------------------
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

    clusterer = kmeans.KMeansPOSClusterer()

    with torch.no_grad():
        # -------------------------
        # Embed once
        # -------------------------
        train_ds = clusterer.embed_dataset(
            dataset_splits["train"],
            out_col="embeddings",
            truncate_other_token_cols=("tags",),
        )
        test_ds = clusterer.embed_dataset(
            dataset_splits["test"],
            out_col="embeddings",
            truncate_other_token_cols=("tags",),
        )

        # -------------------------
        # Training stages
        # -------------------------
        for stage in range(N_stages):
            logger.info(
                f"KMeans stage {stage}: "
                f"K={len(tag_mapping)}, iters={iters_per_stage}"
            )

            clusterer.fit(
                train_ds,
                K=len(tag_mapping),
                embeddings_col="embeddings",
                num_iters=iters_per_stage,
            )

            # Save centroids
            if save_path is not None:
                parts = save_path.split(".")
                parts.insert(-1, str(stage))
                stage_save = ".".join(parts)
                torch.save(clusterer.centroids.cpu(), stage_save)

            # Eval
            if res_path is not None:
                parts = res_path.split(".")
                parts.insert(-1, str(stage))
                stage_res = ".".join(parts)
                eval_kmeans(test_ds, clusterer, stage_res)

