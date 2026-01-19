import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

IDENTIFICATION_DIR = "./../../../Output/Identification"
MATCH_SCORES_DIR = "./../../../Output/Identification/Match_scores"
FIGURES_DIR = "./../../../Output/Identification/Figures"

os.makedirs(MATCH_SCORES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def extract_csis(text):
    """
    Extract CSI strings wrapped in <CSI>...</CSI>.
    DOTALL supports multiline CSI content.
    """
    if pd.isna(text):
        return []
    return [csi.strip() for csi in re.findall(r"<CSI>(.*?)</CSI>", str(text), flags=re.DOTALL)]

def hungarian_match_similarities(gt_csis, model_csis):
    """
    Optimal 1-to-1 (Hungarian) matching between GT and predicted CSIs.
    Returns the list of similarity values for matched pairs.

    Unmatched items are handled by padding with dummy assignments (similarity = 0).
    """
    gt = [g.strip().lower() for g in gt_csis]
    md = [m.strip().lower() for m in model_csis]

    n_gt, n_md = len(gt), len(md)
    if n_gt == 0 or n_md == 0:
        return []

    # Similarity matrix (n_gt x n_md)
    sim = np.zeros((n_gt, n_md), dtype=float)
    for i, g in enumerate(gt):
        for j, m in enumerate(md):
            sim[i, j] = Levenshtein.normalized_similarity(g, m)

    # Pad to square to allow "unmatched" via dummy rows/cols => similarity 0.
    size = max(n_gt, n_md)
    cost = np.ones((size, size), dtype=float)  # cost = 1 - sim ; dummy entries => cost=1 (sim=0)
    cost[:n_gt, :n_md] = 1.0 - sim

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_sims = []
    for r, c in zip(row_ind, col_ind):
        if r < n_gt and c < n_md:
            matched_sims.append(sim[r, c])

    return matched_sims

def soft_prf_hungarian(gt_csis, model_csis):
    """
    Soft precision/recall/F1 using Hungarian 1-to-1 matching.
    Uses similarity values directly (no threshold).

    Edge cases:
      - if both empty => (1,1,1)
      - if only one empty => (0,0,0)
    """
    if not gt_csis and not model_csis:
        return 1.0, 1.0, 1.0
    if not gt_csis or not model_csis:
        return 0.0, 0.0, 0.0

    matched_sims = hungarian_match_similarities(gt_csis, model_csis)
    sim_sum = float(np.sum(matched_sims))

    precision = sim_sum / len(model_csis) if len(model_csis) else 0.0
    recall = sim_sum / len(gt_csis) if len(gt_csis) else 0.0
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    return precision, recall, f1

# -----------------------------
# Load GT once, extract once
# -----------------------------
df_ground_truth_base = pd.read_csv("./../../../Datasets/Culturemark_final - westvalue_df_final.csv")
df_ground_truth_base["ground_truth_csis"] = df_ground_truth_base["Sentence"].apply(extract_csis)

aggregated_results = []

for csv_file in os.listdir(IDENTIFICATION_DIR):
    if not csv_file.endswith(".csv"):
        continue

    model_name = csv_file.replace(".csv", "")
    print(f"\nProcessing {model_name}...")

    df_model = pd.read_csv(os.path.join(IDENTIFICATION_DIR, csv_file))
    df_model["model_csis"] = df_model["CSI Output"].apply(extract_csis)

    # Avoid silent truncation from zip()
    if len(df_model) != len(df_ground_truth_base):
        raise ValueError(
            f"Row mismatch for {model_name}: GT={len(df_ground_truth_base)} vs Model={len(df_model)}"
        )

    # Work on a copy so we don't mutate GT across models
    df_eval = df_ground_truth_base.copy()

    # Soft metrics per row
    match_scores = []  # Soft F1 per row (used as match_score)
    precisions = []
    recalls = []

    for gt_csis, model_csis in zip(df_eval["ground_truth_csis"], df_model["model_csis"]):
        p, r, f1 = soft_prf_hungarian(gt_csis, model_csis)
        precisions.append(p)
        recalls.append(r)
        match_scores.append(f1)

    # Overall (percentage)
    overall_f1 = (sum(match_scores) / len(match_scores)) * 100
    overall_precision = (sum(precisions) / len(precisions)) * 100
    overall_recall = (sum(recalls) / len(recalls)) * 100

    df_eval["CSI Category"] = df_model["CSI Category"]
    df_eval["match_score"] = match_scores
    df_eval["precision"] = precisions
    df_eval["recall"] = recalls

    category_f1 = df_eval.groupby("CSI Category")["match_score"].mean() * 100
    category_precision = df_eval.groupby("CSI Category")["precision"].mean() * 100
    category_recall = df_eval.groupby("CSI Category")["recall"].mean() * 100

    print(f"Overall Levenshtein-Hungarian Soft F1: {overall_f1:.2f}%")
    print(f"Overall Soft Precision: {overall_precision:.2f}% | Overall Soft Recall: {overall_recall:.2f}%")

    print("\nCategory-wise Scores (Soft F1 / Soft Precision / Soft Recall):")
    for category in category_f1.index:
        print(
            f"  {category}: "
            f"{category_f1[category]:.2f}% / "
            f"{category_precision[category]:.2f}% / "
            f"{category_recall[category]:.2f}%"
        )

    # Aggregated results (F1 only)
    aggregated_results.append({
        "Experiment": "Levenshtein-Hungarian-Soft",
        "Model": model_name,
        "Category": "Overall",
        "Score": overall_f1
    })

    for category in category_f1.index:
        aggregated_results.append({
            "Experiment": "Levenshtein-Hungarian-Soft",
            "Model": model_name,
            "Category": category,
            "Score": float(category_f1[category])
        })

    # Save category-wise CSV (expanded)
    category_scores_df = pd.DataFrame({
        "CSI Category": category_f1.index,
        "Soft F1 (%)": category_f1.values,
        "Soft Precision (%)": category_precision.reindex(category_f1.index).values,
        "Soft Recall (%)": category_recall.reindex(category_f1.index).values
    })
    category_scores_df.to_csv(
        f"{MATCH_SCORES_DIR}/levenshtein_hungarian_soft_{model_name}_category_scores.csv",
        index=False
    )

    # Histograms
    gt_csi_counts = df_eval["ground_truth_csis"].apply(len)
    plt.figure(figsize=(8, 6))
    plt.hist(gt_csi_counts, bins=range(max(gt_csi_counts.max(), 1) + 1),
             color="skyblue", edgecolor="black")
    plt.title("Ground Truth: Distribution of CSI Counts per Sentence")
    plt.xlabel("Number of CSIs")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIGURES_DIR}/levenshtein_hungarian_soft_{model_name}_ground_truth_distribution.png")
    plt.close()

    model_csi_counts = df_model["model_csis"].apply(len)
    plt.figure(figsize=(8, 6))
    plt.hist(model_csi_counts, bins=range(max(model_csi_counts.max(), 1) + 1),
             color="salmon", edgecolor="black")
    plt.title("Model Output: Distribution of CSI Counts per Sentence")
    plt.xlabel("Number of CSIs")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIGURES_DIR}/levenshtein_hungarian_soft_{model_name}_model_output_distribution.png")
    plt.close()

    # Save per-row detailed results
    df_eval["model_output"] = df_model["CSI Output"]
    df_eval["model_csis"] = df_model["model_csis"]
    df_eval.to_csv(
        f"{MATCH_SCORES_DIR}/levenshtein_hungarian_soft_{model_name}_CSI_match_scores.csv",
        index=False
    )

# Save aggregated results
df_aggregated = pd.DataFrame(aggregated_results)
df_aggregated.to_csv(f"{MATCH_SCORES_DIR}/levenshtein_hungarian_soft_aggregated_results.csv", index=False)

print("\nProcessing complete!")
print(f"Aggregated results saved to: {MATCH_SCORES_DIR}/levenshtein_hungarian_soft_aggregated_results.csv")
