#!/usr/bin/env python3
"""
BERTScore Evaluation for Chinese Cultural Adaptation Task with Category-wise Results

This script evaluates cultural adaptation outputs using BERTScore F1 metrics
for Chinese language adaptations.

Computes:
- Overall scores
- Category-wise scores (by CSI Category, CSI Sub-Category, Hall's Level)
- Both Intra-lingual (English) and Inter-lingual (Chinese) adaptations

Supported Chinese BERT Models:
- Chinese-BERT-wwm-ext (hfl/chinese-bert-wwm-ext) - RECOMMENDED (whole word masking)
- BERT-base-Chinese (bert-base-chinese) - Standard baseline
- Chinese-RoBERTa-wwm-ext (hfl/chinese-roberta-wwm-ext) - RoBERTa variant with WWM

Usage:
    python bertscore_chinese.py --model qwen
    python bertscore_chinese.py --model qwen --bert chinesebert
    python bertscore_chinese.py --all --bert chinesebert  # Evaluate all LLM models
    python bertscore_chinese.py --model qwen --all-bert  # Evaluate with all BERT models
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
import warnings

# Suppress FutureWarnings from HuggingFace
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONFIGURATION - Modify these paths according to your project structure
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = ROOT / "Output" / "Adaptation" / "Chinese" / "BERTScore_Results"
GROUND_TRUTH_FILE = ROOT / "Datasets" / "Chinese_Main.csv"
MODEL_OUTPUT_DIR = ROOT / "Output" / "Adaptation" / "Chinese"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CHINESE BERT MODEL CONFIGURATIONS
# ============================================================================

CHINESE_BERT_MODELS = {
    'chinesebert': {
        'name': 'Chinese-BERT-wwm-ext',
        'model_id': 'hfl/chinese-bert-wwm-ext',
        'description': 'RECOMMENDED - Whole word masking for better Chinese semantics, extended training corpus'
    },
    'bertbase': {
        'name': 'BERT-base-Chinese',
        'model_id': 'bert-base-chinese',
        'description': 'Standard baseline by Google, character-level masking'
    },
    'roberta': {
        'name': 'Chinese-RoBERTa-wwm-ext',
        'model_id': 'hfl/chinese-roberta-wwm-ext',
        'description': 'RoBERTa variant with whole word masking, often best performance'
    }
}

# English model for Intra-lingual evaluation
ENGLISH_MODEL = 'bert-base-uncased'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    """
    Compute BERTScore for Intra-lingual (English) adaptations.

    Args:
        output: Model's English adaptation output
        ground_truth: Ground truth English adaptation

    Returns:
        tuple: (full_sentence_f1, avg_csi_f1) or (None, None) if CSI count mismatch
    """
    gt_csi_list = extract_all_csi(ground_truth)
    out_csi_list = extract_all_csi(output)

    if len(gt_csi_list) != len(out_csi_list):
        return None, None

    # Full sentence score
    P_full, R_full, F1_full = score(
        [output], [ground_truth],
        lang="en",
        model_type=ENGLISH_MODEL,
        rescale_with_baseline=True
    )
    full_f1 = F1_full.mean().item()

    # CSI segment scores
    csi_scores = []
    for gt_csi, out_csi in zip(gt_csi_list, out_csi_list):
        P, R, F1 = score(
            [out_csi], [gt_csi],
            lang="en",
            model_type=ENGLISH_MODEL,
            rescale_with_baseline=True
        )
        csi_scores.append(F1.mean().item())

    avg_csi_score = sum(csi_scores) / len(csi_scores) if csi_scores else 0.0
    return full_f1, avg_csi_score


def compute_inter_scores(output, ground_truth, scorer):
    """
    Compute BERTScore for Inter-lingual (Chinese) adaptations.

    Args:
        output: Model's Chinese adaptation output
        ground_truth: Ground truth Chinese adaptation
        scorer: BERTScorer instance for Chinese

    Returns:
        tuple: (full_sentence_f1, avg_csi_f1) or (None, None) if CSI count mismatch
    """
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


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(llm_model_name, ground_truth_df, scorer_zh, bert_model_key):
    """
    Evaluate a single LLM model's outputs and return results.

    Args:
        llm_model_name: Name of the LLM (e.g., 'qwen', 'gpt', 'claude')
        ground_truth_df: DataFrame with ground truth adaptations
        scorer_zh: BERTScorer instance for Chinese
        bert_model_key: Key for the Chinese BERT model being used

    Returns:
        DataFrame with detailed per-sentence scores, or None if evaluation failed
    """
    model_output_file = MODEL_OUTPUT_DIR / f"{llm_model_name}.csv"

    if not model_output_file.exists():
        print(f"  ERROR: Output file not found: {model_output_file}")
        return None

    # Load model outputs
    model_output_df = pd.read_csv(model_output_file)
    print(f"  Loaded: {len(model_output_df)} rows")

    # Check for required columns
    if 'Intra' not in model_output_df.columns or 'Inter' not in model_output_df.columns:
        print(f"  ERROR: Missing 'Intra' or 'Inter' columns in model output")
        print(f"    Available columns: {list(model_output_df.columns)}")
        return None

    # Check for missing values and report
    missing_intra = model_output_df['Intra'].isna().sum()
    missing_inter = model_output_df['Inter'].isna().sum()
    if missing_intra > 0 or missing_inter > 0:
        print(f"  WARNING: Missing values detected: Intra={missing_intra}, Inter={missing_inter}")
        print(f"    These rows will be skipped during evaluation.")

    # Lists to collect scores
    results = []

    print(f"  Processing sentences...")

    for i in tqdm(range(len(model_output_df)), desc=f"  {llm_model_name}", leave=False):
        # Get model outputs
        output_intra = model_output_df['Intra'].iloc[i]
        output_inter = model_output_df['Inter'].iloc[i]

        # Get ground truths from Chinese_Main.csv
        gt_intra = ground_truth_df['Intra-lingual'].iloc[i]
        gt_inter = ground_truth_df['Inter-lingual'].iloc[i]

        # Get metadata
        sentence = ground_truth_df['Sentence'].iloc[i]
        csi_category = ground_truth_df['CSI Category'].iloc[i] if 'CSI Category' in ground_truth_df.columns else None
        csi_subcategory = ground_truth_df['CSI Sub-Category'].iloc[i] if 'CSI Sub-Category' in ground_truth_df.columns else None
        halls_level = ground_truth_df["Hall's Level"].iloc[i] if "Hall's Level" in ground_truth_df.columns else None

        # Check for missing/NaN values and skip if found
        intra_full, intra_csi = None, None
        inter_full, inter_csi = None, None

        # Compute Intra scores (only if both output and ground truth are valid strings)
        if pd.notna(output_intra) and pd.notna(gt_intra) and isinstance(output_intra, str) and isinstance(gt_intra, str):
            intra_full, intra_csi = compute_intra_scores(output_intra, gt_intra)

        # Compute Inter scores (only if both output and ground truth are valid strings)
        if pd.notna(output_inter) and pd.notna(gt_inter) and isinstance(output_inter, str) and isinstance(gt_inter, str):
            inter_full, inter_csi = compute_inter_scores(output_inter, gt_inter, scorer_zh)

        # Store results
        results.append({
            'Sentence': sentence,
            'CSI Category': csi_category,
            'CSI Sub-Category': csi_subcategory,
            "Hall's Level": halls_level,
            'Intra_Full_F1': intra_full,
            'Intra_CSI_F1': intra_csi,
            'Inter_Full_F1': inter_full,
            'Inter_CSI_F1': inter_csi,
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save detailed results
    detailed_file = OUTPUT_DIR / f"{llm_model_name}_{bert_model_key}_detailed_scores.csv"
    results_df.to_csv(detailed_file, index=False)
    print(f"  SAVED: Detailed scores saved: {detailed_file.name}")

    return results_df


def compute_overall_scores(results_df):
    """
    Compute overall average scores across all sentences.

    Args:
        results_df: DataFrame with per-sentence scores

    Returns:
        DataFrame with overall scores
    """
    overall_scores = {
        'Adaptation Type': ['Intra-lingual (English)', 'Inter-lingual (Chinese)'],
        'Avg Full Sentence F1': [
            results_df['Intra_Full_F1'].mean(),
            results_df['Inter_Full_F1'].mean()
        ],
        'Avg CSI F1': [
            results_df['Intra_CSI_F1'].mean(),
            results_df['Inter_CSI_F1'].mean()
        ],
        'Valid Count': [
            results_df['Intra_Full_F1'].notna().sum(),
            results_df['Inter_Full_F1'].notna().sum()
        ]
    }

    return pd.DataFrame(overall_scores)


def compute_category_scores(results_df, category_column='CSI Category'):
    """
    Compute category-wise average scores.

    Args:
        results_df: DataFrame with per-sentence scores
        category_column: Column name to group by

    Returns:
        DataFrame with category-wise scores, or None if column doesn't exist
    """
    if category_column not in results_df.columns or results_df[category_column].notna().sum() == 0:
        return None

    # Filter out rows without categories
    categorized_df = results_df[results_df[category_column].notna()]

    # Compute category-wise averages
    category_scores = categorized_df.groupby(category_column).agg({
        'Intra_Full_F1': 'mean',
        'Intra_CSI_F1': 'mean',
        'Inter_Full_F1': 'mean',
        'Inter_CSI_F1': 'mean',
    }).round(4)

    # Add count column
    category_scores['Count'] = categorized_df.groupby(category_column).size()

    # Reorder columns
    category_scores = category_scores[['Count'] + [col for col in category_scores.columns if col != 'Count']]

    return category_scores


def print_results(llm_model_name, bert_model_key, results_df, overall_df, category_dfs):
    """
    Print evaluation results to console.

    Args:
        llm_model_name: Name of the LLM being evaluated
        bert_model_key: Key for the Chinese BERT model
        results_df: DataFrame with per-sentence scores
        overall_df: DataFrame with overall scores
        category_dfs: Dict of category DataFrames
    """
    bert_info = CHINESE_BERT_MODELS[bert_model_key]

    print("\n" + "=" * 90)
    print(f"RESULTS FOR {llm_model_name.upper()} (using {bert_info['name']})")
    print("=" * 90)

    # Overall scores
    print("\nOVERALL SCORES:")
    print("-" * 90)
    print(overall_df.to_string(index=False))

    # Category-wise scores
    for category_name, category_df in category_dfs.items():
        if category_df is not None:
            print(f"\n\nSCORES BY {category_name.upper()}:")
            print("-" * 90)
            print(category_df.to_string())

    # Summary stats
    print("\n\nSUMMARY:")
    print("-" * 90)
    total = len(results_df)
    valid_intra = results_df['Intra_Full_F1'].notna().sum()
    valid_inter = results_df['Inter_Full_F1'].notna().sum()

    print(f"Total sentences: {total}")
    print(f"Valid Intra-lingual: {valid_intra}/{total} ({valid_intra/total*100:.1f}%)")
    print(f"Valid Inter-lingual: {valid_inter}/{total} ({valid_inter/total*100:.1f}%)")

    # Category coverage
    for col in ['CSI Category', 'CSI Sub-Category', "Hall's Level"]:
        if col in results_df.columns:
            with_cat = results_df[col].notna().sum()
            print(f"With {col}: {with_cat}/{total} ({with_cat/total*100:.1f}%)")


def initialize_chinese_scorer(bert_model_key):
    """
    Initialize BERTScorer with the specified Chinese BERT model.

    Args:
        bert_model_key: Key from CHINESE_BERT_MODELS dict

    Returns:
        BERTScorer instance
    """
    bert_config = CHINESE_BERT_MODELS[bert_model_key]
    model_id = bert_config['model_id']

    print(f"\nInitializing Chinese BERTScore model...")
    print(f"  Model: {bert_config['name']}")
    print(f"  ID: {model_id}")
    print(f"  Description: {bert_config['description']}")

    model = AutoModel.from_pretrained(model_id)
    scorer = BERTScorer(
        model_type=model_id,
        lang="zh",
        num_layers=model.config.num_hidden_layers
    )

    print(f"  LOADED: Chinese model loaded: {bert_config['name']}")
    print(f"  LOADED: English model: {ENGLISH_MODEL}")

    return scorer


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Chinese cultural adaptation using BERTScore with category-wise results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Chinese BERT Models Available:
  chinesebert  - Chinese-BERT-wwm-ext (RECOMMENDED - whole word masking)
  bertbase     - BERT-base-Chinese (standard baseline)
  roberta      - Chinese-RoBERTa-wwm-ext (RoBERTa variant)

Examples:
  python bertscore_chinese.py --model qwen
  python bertscore_chinese.py --model qwen --bert chinesebert
  python bertscore_chinese.py --all --bert chinesebert
  python bertscore_chinese.py --model gpt --all-bert
        """
    )

    parser.add_argument(
        "--model",
        help="LLM model to evaluate (e.g., gpt, gemini, claude, deepseek, llama, qwen)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all available LLM models"
    )
    parser.add_argument(
        "--bert",
        choices=list(CHINESE_BERT_MODELS.keys()),
        default='chinesebert',
        help="Chinese BERT model for evaluation (default: chinesebert)"
    )
    parser.add_argument(
        "--all-bert",
        action="store_true",
        help="Evaluate using all Chinese BERT models"
    )

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Please specify --model or --all")

    print("=" * 90)
    print("BERTSCORE EVALUATION FOR CHINESE CULTURAL ADAPTATION")
    print("With Category-wise Results")
    print("=" * 90)

    # Check for ground truth file
    if not GROUND_TRUTH_FILE.exists():
        print(f"\nERROR: Ground truth file not found: {GROUND_TRUTH_FILE}")
        print("\nExpected file: Chinese_Main.csv")
        print("Please ensure the file exists at the specified path.")
        return 1

    # Load ground truth
    print(f"\nLoading ground truth: {GROUND_TRUTH_FILE.name}")
    ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)
    print(f"  LOADED: {len(ground_truth_df)} rows")

    # Check available metadata columns
    metadata_cols = ['CSI Category', 'CSI Sub-Category', "Hall's Level"]
    for col in metadata_cols:
        if col in ground_truth_df.columns:
            count = ground_truth_df[col].notna().sum()
            unique = ground_truth_df[col].nunique()
            print(f"  FOUND: {col}: {count} sentences, {unique} unique values")

    # Determine BERT models to use
    if args.all_bert:
        bert_models = list(CHINESE_BERT_MODELS.keys())
    else:
        bert_models = [args.bert]

    # Determine LLM models to evaluate
    if args.all:
        llm_models = ['gpt', 'gemini', 'claude', 'deepseek', 'llama', 'qwen', 'olmo']
    else:
        llm_models = [args.model]

    # Track all results
    all_results = {}

    # Evaluate each BERT model
    for bert_model_key in bert_models:
        # Initialize Chinese scorer
        scorer_zh = initialize_chinese_scorer(bert_model_key)

        # Evaluate each LLM model
        for llm_model_name in llm_models:
            print(f"\n{'=' * 90}")
            print(f"Evaluating: {llm_model_name} (with {CHINESE_BERT_MODELS[bert_model_key]['name']})")
            print('=' * 90)

            results_df = evaluate_model(llm_model_name, ground_truth_df, scorer_zh, bert_model_key)

            if results_df is None:
                print(f"  SKIPPED: {llm_model_name}")
                continue

            # Compute overall scores
            overall_df = compute_overall_scores(results_df)

            # Compute category-wise scores for different groupings
            category_dfs = {
                'CSI Category': compute_category_scores(results_df, 'CSI Category'),
                'CSI Sub-Category': compute_category_scores(results_df, 'CSI Sub-Category'),
                "Hall's Level": compute_category_scores(results_df, "Hall's Level")
            }

            # Save scores
            overall_file = OUTPUT_DIR / f"{llm_model_name}_{bert_model_key}_overall_scores.csv"
            overall_df.to_csv(overall_file, index=False)
            print(f"  SAVED: Overall scores saved: {overall_file.name}")

            for category_name, category_df in category_dfs.items():
                if category_df is not None:
                    safe_name = category_name.replace("'", "").replace(" ", "_")
                    category_file = OUTPUT_DIR / f"{llm_model_name}_{bert_model_key}_{safe_name}_scores.csv"
                    category_df.to_csv(category_file)
                    print(f"  SAVED: {category_name} scores saved: {category_file.name}")

            # Print results
            print_results(llm_model_name, bert_model_key, results_df, overall_df, category_dfs)

            # Store for summary
            result_key = f"{llm_model_name}_{bert_model_key}"
            all_results[result_key] = {
                'overall': overall_df,
                'categories': category_dfs
            }

    # Print final summary
    print("\n" + "=" * 90)
    print("EVALUATION COMPLETE")
    print("=" * 90)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nEvaluated {len(all_results)} configuration(s):")
    for result_key in all_results.keys():
        print(f"  DONE: {result_key}")

    return 0


if __name__ == "__main__":
    exit(main())
