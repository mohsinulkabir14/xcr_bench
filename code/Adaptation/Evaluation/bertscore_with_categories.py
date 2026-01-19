#!/usr/bin/env python3
"""
BERTScore Evaluation for Adaptation Task with Category-wise Results

This script evaluates cultural adaptation outputs using BERTScore F1 metrics.

Computes:
- Overall scores (like original evaluation)
- Category-wise scores (new!)
- Both Hindu and Muslim cultural contexts
- Both Intra-lingual (English) and Inter-lingual (Bengali) adaptations

Usage:
    python bertscore_with_categories.py --model gpt
    python bertscore_with_categories.py --model claude
    python bertscore_with_categories.py --all  # Evaluate all models
"""

import pandas as pd
from bert_score import BERTScorer, score
from transformers import AutoModel
import re
import numpy as np
from tqdm import tqdm
import os
import argparse
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = ROOT / "Output" / "Adaptation" / "Category_Scores"
GROUND_TRUTH_FILE = ROOT / "Datasets" / "Adaptation_Final_with_Categories.csv"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_all_csi(text):
    """
    Extract all content between <CSI> tags from the given text.

    Args:
        text: Input string or any other type

    Returns:
        list: List of found CSI items (empty list if none found or invalid input)
    """
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        return re.findall(r'<CSI>(.*?)</CSI>', text)
    except (TypeError, re.error):
        return []


def compute_intra_scores(output, ground_truth):
    """Compute BERTScore for Intra-lingual (English) adaptations."""
    gt_csi_list = extract_all_csi(ground_truth)
    out_csi_list = extract_all_csi(output)

    if len(gt_csi_list) != len(out_csi_list):
        return None, None

    # Full sentence score
    P_full, R_full, F1_full = score([output], [ground_truth], lang="en",
                                     model_type='bert-base-uncased', rescale_with_baseline=True)
    full_f1 = F1_full.mean().item()

    # CSI segment scores
    csi_scores = []
    for gt_csi, out_csi in zip(gt_csi_list, out_csi_list):
        P, R, F1 = score([out_csi], [gt_csi], lang="en",
                        model_type='bert-base-uncased', rescale_with_baseline=True)
        csi_scores.append(F1.mean().item())

    avg_csi_score = sum(csi_scores) / len(csi_scores) if csi_scores else 0.0
    return full_f1, avg_csi_score


def compute_inter_scores(output, ground_truth, scorer):
    """Compute BERTScore for Inter-lingual (Bengali) adaptations."""
    gt_csi_list = extract_all_csi(ground_truth)
    out_csi_list = extract_all_csi(output)

    if len(gt_csi_list) != len(out_csi_list):
        return None, None

    # Full sentence score
    P_full, R_full, F1_full = scorer.score([output], [ground_truth])
    full_f1 = F1_full.mean().item()

    # CSI segment scores
    csi_scores = []
    for gt_csi, out_csi in zip(gt_csi_list, out_csi_list):
        P, R, F1 = scorer.score([out_csi], [gt_csi])
        csi_scores.append(F1.mean().item())

    avg_csi_score = sum(csi_scores) / len(csi_scores) if csi_scores else 0.0
    return full_f1, avg_csi_score


def evaluate_model(model_name, ground_truth_df, scorer_bn):
    """Evaluate a single model and return results."""

    model_output_file = ROOT / "Output" / "Adaptation" / f"{model_name}.csv"

    if not model_output_file.exists():
        print(f"  ✗ Output file not found: {model_output_file}")
        return None

    # Load model outputs
    model_output_df = pd.read_csv(model_output_file)
    print(f"  Loaded: {len(model_output_df)} rows")

    # Check for required columns
    if 'Intra' not in model_output_df.columns or 'Inter' not in model_output_df.columns:
        print(f"  ✗ Missing 'Intra' or 'Inter' columns")
        return None

    # Lists to collect scores
    results = []

    print(f"  Processing sentences...")

    for i in tqdm(range(len(model_output_df)), desc=f"  {model_name}", leave=False):
        # Get outputs
        output_intra = model_output_df['Intra'].iloc[i]
        output_inter = model_output_df['Inter'].iloc[i]

        # Get ground truths
        gt_intra_hindu = ground_truth_df['Intra-lingual (Hindu)'].iloc[i]
        gt_intra_muslim = ground_truth_df['Intra-lingual (Muslim)'].iloc[i]
        gt_inter_hindu = ground_truth_df['Inter-lingual (Hindu)'].iloc[i]
        gt_inter_muslim = ground_truth_df['Inter-lingual (Muslim)'].iloc[i]

        # Get metadata
        category = ground_truth_df['CSI Category'].iloc[i] if 'CSI Category' in ground_truth_df.columns else None
        sentence = ground_truth_df['Sentence'].iloc[i]

        # Compute scores for Hindu context
        intra_hindu_full, intra_hindu_csi = compute_intra_scores(output_intra, gt_intra_hindu)
        inter_hindu_full, inter_hindu_csi = compute_inter_scores(output_inter, gt_inter_hindu, scorer_bn)

        # Compute scores for Muslim context
        intra_muslim_full, intra_muslim_csi = compute_intra_scores(output_intra, gt_intra_muslim)
        inter_muslim_full, inter_muslim_csi = compute_inter_scores(output_inter, gt_inter_muslim, scorer_bn)

        # Store results
        results.append({
            'Sentence': sentence,
            'CSI Category': category,
            'Intra_Hindu_Full_F1': intra_hindu_full,
            'Intra_Hindu_CSI_F1': intra_hindu_csi,
            'Intra_Muslim_Full_F1': intra_muslim_full,
            'Intra_Muslim_CSI_F1': intra_muslim_csi,
            'Inter_Hindu_Full_F1': inter_hindu_full,
            'Inter_Hindu_CSI_F1': inter_hindu_csi,
            'Inter_Muslim_Full_F1': inter_muslim_full,
            'Inter_Muslim_CSI_F1': inter_muslim_csi,
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save detailed results
    detailed_file = OUTPUT_DIR / f"{model_name}_detailed_scores.csv"
    results_df.to_csv(detailed_file, index=False)
    print(f"  ✓ Detailed scores saved: {detailed_file.name}")

    return results_df


def compute_overall_scores(results_df):
    """Compute overall scores (like original evaluation)."""
    overall_scores = {
        'Adaptation': ['Intra-lingual (Hindu)', 'Intra-lingual (Muslim)',
                      'Inter-lingual (Hindu)', 'Inter-lingual (Muslim)'],
        'Avg Full Sentence F1': [
            results_df['Intra_Hindu_Full_F1'].mean(),
            results_df['Intra_Muslim_Full_F1'].mean(),
            results_df['Inter_Hindu_Full_F1'].mean(),
            results_df['Inter_Muslim_Full_F1'].mean()
        ],
        'Avg Aggregate CSI F1': [
            results_df['Intra_Hindu_CSI_F1'].mean(),
            results_df['Intra_Muslim_CSI_F1'].mean(),
            results_df['Inter_Hindu_CSI_F1'].mean(),
            results_df['Inter_Muslim_CSI_F1'].mean()
        ]
    }

    return pd.DataFrame(overall_scores)


def compute_category_scores(results_df):
    """Compute category-wise scores."""
    if 'CSI Category' not in results_df.columns or results_df['CSI Category'].notna().sum() == 0:
        return None

    # Filter out rows without categories
    categorized_df = results_df[results_df['CSI Category'].notna()]

    # Compute category-wise averages
    category_scores = categorized_df.groupby('CSI Category').agg({
        'Intra_Hindu_Full_F1': 'mean',
        'Intra_Hindu_CSI_F1': 'mean',
        'Intra_Muslim_Full_F1': 'mean',
        'Intra_Muslim_CSI_F1': 'mean',
        'Inter_Hindu_Full_F1': 'mean',
        'Inter_Hindu_CSI_F1': 'mean',
        'Inter_Muslim_Full_F1': 'mean',
        'Inter_Muslim_CSI_F1': 'mean',
    }).round(4)

    # Add count column
    category_scores['Count'] = categorized_df.groupby('CSI Category').size()

    # Reorder columns
    category_scores = category_scores[['Count'] + [col for col in category_scores.columns if col != 'Count']]

    return category_scores


def print_results(model_name, results_df, overall_df, category_df):
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print(f"RESULTS FOR {model_name.upper()}")
    print("=" * 80)

    # Overall scores
    print("\nOVERALL SCORES:")
    print("-" * 80)
    print(overall_df.to_string(index=False))

    # Category-wise scores
    if category_df is not None:
        print("\n\nCATEGORY-WISE SCORES (Hindu Context):")
        print("-" * 80)
        simplified = category_df[['Count', 'Intra_Hindu_Full_F1', 'Intra_Hindu_CSI_F1',
                                  'Inter_Hindu_Full_F1', 'Inter_Hindu_CSI_F1']].copy()
        simplified.columns = ['Count', 'Intra_Full', 'Intra_CSI', 'Inter_Full', 'Inter_CSI']
        print(simplified.to_string())
    else:
        print("\n⚠ Category-wise scores not available (no CSI Category metadata)")

    # Summary stats
    print("\n\nSUMMARY:")
    print("-" * 80)
    total = len(results_df)
    valid_intra = results_df['Intra_Hindu_Full_F1'].notna().sum()
    valid_inter = results_df['Inter_Hindu_Full_F1'].notna().sum()
    with_categories = results_df['CSI Category'].notna().sum()

    print(f"Total sentences: {total}")
    print(f"Valid Intra-lingual: {valid_intra}/{total} ({valid_intra/total*100:.1f}%)")
    print(f"Valid Inter-lingual: {valid_inter}/{total} ({valid_inter/total*100:.1f}%)")
    print(f"With CSI categories: {with_categories}/{total} ({with_categories/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate adaptation task with category-wise BERTScore results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Model to evaluate (gpt, gemini, claude, deepseek, llama, qwen)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all available models"
    )

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Please specify --model or --all")

    print("=" * 80)
    print("BERTSCORE EVALUATION FOR ADAPTATION TASK")
    print("With Category-wise Results")
    print("=" * 80)

    # Check for ground truth file
    if not GROUND_TRUTH_FILE.exists():
        print(f"\n✗ Ground truth file not found: {GROUND_TRUTH_FILE}")
        print("\nPlease run:")
        print(f"  cd {ROOT / 'Datasets'}")
        print("  python3 merge_adaptation_metadata.py")
        return 1

    # Load ground truth
    print(f"\nLoading ground truth: {GROUND_TRUTH_FILE.name}")
    ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)
    print(f"  ✓ Loaded: {len(ground_truth_df)} rows")

    has_categories = 'CSI Category' in ground_truth_df.columns
    if has_categories:
        category_count = ground_truth_df['CSI Category'].notna().sum()
        print(f"  ✓ CSI Categories available: {category_count} sentences")
    else:
        print("  ⚠ No CSI Category metadata (only overall scores will be computed)")

    # Initialize Bengali BERTScore model
    print("\nInitializing BERTScore models...")
    model_name_bn = 'sagorsarker/bangla-bert-base'
    model_bn = AutoModel.from_pretrained(model_name_bn)
    scorer_bn = BERTScorer(model_type=model_name_bn, lang="bn",
                          num_layers=model_bn.config.num_hidden_layers)
    print(f"  ✓ Bengali model: {model_name_bn}")
    print(f"  ✓ English model: bert-base-uncased")

    # Determine models to evaluate
    if args.all:
        models = ['gpt', 'gemini', 'claude', 'deepseek', 'llama', 'qwen']
    else:
        models = [args.model]

    # Evaluate each model
    all_results = {}

    for model_name in models:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {model_name}")
        print('=' * 80)

        results_df = evaluate_model(model_name, ground_truth_df, scorer_bn)

        if results_df is None:
            print(f"  ✗ Skipping {model_name}")
            continue

        # Compute scores
        overall_df = compute_overall_scores(results_df)
        category_df = compute_category_scores(results_df)

        # Save scores
        overall_file = OUTPUT_DIR / f"{model_name}_overall_scores.csv"
        overall_df.to_csv(overall_file, index=False)
        print(f"  ✓ Overall scores saved: {overall_file.name}")

        if category_df is not None:
            category_file = OUTPUT_DIR / f"{model_name}_category_scores.csv"
            category_df.to_csv(category_file)
            print(f"  ✓ Category scores saved: {category_file.name}")

        # Print results
        print_results(model_name, results_df, overall_df, category_df)

        # Store for summary
        all_results[model_name] = {
            'overall': overall_df,
            'category': category_df
        }

    # Print final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nEvaluated {len(all_results)} model(s):")
    for model_name in all_results.keys():
        print(f"  ✓ {model_name}")

    return 0


if __name__ == "__main__":
    exit(main())
