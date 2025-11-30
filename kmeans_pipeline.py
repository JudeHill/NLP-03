import csv
from tqdm import tqdm
from utils import calculate_v_measure, calculate_variation_of_information
import torch
import pos_tagging.kmeans as kmeans

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
