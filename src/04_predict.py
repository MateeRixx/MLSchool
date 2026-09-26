"""
Prediction / Inference for Entity Resolution.

Loads trained model, runs inference on test features,
applies thresholds, generates submission files.
"""

import polars as pl
import joblib
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time

from utils import (
    OUTPUT_DIR,
    SEED,
)

# Paths
FEATURES_TEST = OUTPUT_DIR / "features_test.parquet"
CANDIDATES_TEST = OUTPUT_DIR / "candidate_pairs_test.parquet"

MODEL_DIR = OUTPUT_DIR / "models"
CALIBRATED_MODEL_PATH = MODEL_DIR / "xgb_calibrated.pkl"
THRESHOLD_PATH = MODEL_DIR / "thresholds.json"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.json"

# Output submission files
MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"


def load_model_and_config():
    """Load model, thresholds, feature columns."""
    print("[Loading] Model and config...")
    
    model = joblib.load(CALIBRATED_MODEL_PATH)
    print(f"  Model loaded: {type(model).__name__}")
    
    with open(THRESHOLD_PATH) as f:
        thresholds = json.load(f)
    print(f"  Thresholds: {thresholds}")
    
    with open(FEATURE_COLS_PATH) as f:
        feat_cols = json.load(f)
    print(f"  Feature columns: {len(feat_cols)}")
    
    return model, thresholds, feat_cols


def run_inference(
    features_df: pl.DataFrame,
    model,
    feat_cols: List[str],
    batch_size: int = 100000,
) -> pl.DataFrame:
    """
    Run batched inference on test features.
    Returns DataFrame with probabilities added.
    """
    print(f"\n[Inference] Running on {len(features_df):,} rows...")
    start = time.time()
    
    # Extract feature matrix
    X = features_df.select(feat_cols).to_numpy()
    
    # Batch prediction
    probs = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_probs = model.predict_proba(batch)[:, 1]
        probs.extend(batch_probs)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(X)):,} / {len(X):,}")
    
    # Add probabilities
    features_df = features_df.with_columns(
        pl.Series("match_prob", probs)
    )
    
    print(f"  Inference complete in {time.time()-start:.1f}s")
    return features_df


def apply_thresholds(
    features_df: pl.DataFrame,
    thresholds: Dict[str, float],
) -> pl.DataFrame:
    """
    Apply threshold per country (or global).
    Returns DataFrame with binary prediction.
    """
    print("\n[Thresholding] Applying thresholds...")
    
    global_thresh = thresholds.get("global", 0.5)
    
    # For now use global threshold
    # TODO: Join country info for per-country thresholds
    features_df = features_df.with_columns(
        pl.when(pl.col("match_prob") >= global_thresh)
        .then(1)
        .otherwise(0)
        .alias("is_match")
    )
    
    match_count = features_df["is_match"].sum()
    print(f"  Matches predicted: {match_count:,} / {len(features_df):,}")
    
    return features_df


def generate_matching_results(
    predictions_df: pl.DataFrame,
    candidates_df: pl.DataFrame,
    output_path: Path,
) -> pl.DataFrame:
    """
    Generate matching_results.tsv from predictions.
    
    Format: source1_entity_id \t matched_entity_ids (comma-separated)
    """
    print(f"\n[Output] Generating {output_path}...")
    
    # Get matched pairs
    matched = predictions_df.filter(pl.col("is_match") == 1)
    
    # Group by S1 entity
    matched_groups = matched.group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").alias("matched_ids")
    )
    
    # Create comma-separated strings
    matched_groups = matched_groups.with_columns(
        pl.col("matched_ids").list.join(",").alias("matched_entity_ids")
    )
    
    # Get ALL S1 entities from candidates (including those with no candidates)
    all_s1 = candidates_df.select("source1_entity_id").unique()
    
    # Left join to ensure all S1 entities present
    results = all_s1.join(matched_groups, on="source1_entity_id", how="left")
    
    # Fill nulls with empty string
    results = results.with_columns(
        pl.col("matched_entity_ids").fill_null("").alias("matched_entity_ids")
    )
    
    # Select and order columns
    results = results.select(["source1_entity_id", "matched_entity_ids"])
    
    # Write TSV
    results.write_csv(output_path, separator="\t", include_header=True)
    
    print(f"  Rows written: {len(results):,}")
    non_empty = results.filter(pl.col("matched_entity_ids") != "").height
    print(f"  Non-empty: {non_empty:,} | Empty (singletons): {len(results)-non_empty:,}")
    
    return results


def generate_candidate_pairs(
    candidates_df: pl.DataFrame,
    output_path: Path,
):
    """
    Generate candidate_pairs.tsv from candidates.
    
    Format: source1_entity_id \t candidate_entity_ids (comma-separated)
    """
    print(f"\n[Output] Generating {output_path}...")
    
    # Filter non-empty candidates
    non_empty = candidates_df.filter(pl.col("candidate_entity_id") != "")
    
    # Group by S1 entity
    cand_groups = non_empty.group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").alias("candidate_ids")
    )
    
    # Create comma-separated strings
    cand_groups = cand_groups.with_columns(
        pl.col("candidate_ids").list.join(",").alias("candidate_entity_ids")
    )
    
    # Get ALL S1 entities
    all_s1 = candidates_df.select("source1_entity_id").unique()
    
    # Left join
    results = all_s1.join(cand_groups, on="source1_entity_id", how="left")
    
    # Fill nulls with empty string
    results = results.with_columns(
        pl.col("candidate_entity_ids").fill_null("").alias("candidate_entity_ids")
    )
    
    # Select and order columns
    results = results.select(["source1_entity_id", "candidate_entity_ids"])
    
    # Write TSV
    results.write_csv(output_path, separator="\t", include_header=True)
    
    print(f"  Rows written: {len(results):,}")
    non_empty = results.filter(pl.col("candidate_entity_ids") != "").height
    print(f"  Non-empty: {non_empty:,} | Empty: {len(results)-non_empty:,}")


def validate_submission_locally():
    """Run the validation script on generated outputs."""
    print("\n[Validation] Running submission validator...")
    import subprocess
    result = subprocess.run([
        "python", "utils/validate_submission.py",
        "--matching", str(MATCHING_RESULTS),
        "--candidate", str(CANDIDATE_PAIRS),
        "--test-dir", "dataset/test"
    ], capture_output=True, text=True, cwd=".")
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0


def main():
    """Run full prediction pipeline."""
    print("="*60)
    print("PREDICTION / INFERENCE PIPELINE")
    print("="*60)
    
    # Load model and config
    model, thresholds, feat_cols = load_model_and_config()
    
    # Load test features
    print(f"\n[Loading] Test features from {FEATURES_TEST}...")
    features_df = pl.read_parquet(FEATURES_TEST)
    print(f"  Rows: {len(features_df):,}")
    
    # Run inference
    features_df = run_inference(features_df, model, feat_cols)
    
    # Apply thresholds
    features_df = apply_thresholds(features_df, thresholds)
    
    # Load candidates for padding
    print(f"\n[Loading] Candidates from {CANDIDATES_TEST}...")
    candidates_df = pl.read_parquet(CANDIDATES_TEST)
    print(f"  Rows: {len(candidates_df):,}")
    
    # Generate matching_results.tsv
    generate_matching_results(features_df, candidates_df, MATCHING_RESULTS)
    
    # Generate candidate_pairs.tsv
    generate_candidate_pairs(candidates_df, CANDIDATE_PAIRS)
    
    # Validate
    print("\n" + "="*60)
    print("VALIDATING SUBMISSION")
    print("="*60)
    valid = validate_submission_locally()
    
    if valid:
        print("\n✓ VALIDATION PASSED - Ready for submission!")
    else:
        print("\n✗ VALIDATION FAILED - Fix issues above")
    
    print("\nPREDICTION COMPLETE")
    print(f"  matching_results.tsv: {MATCHING_RESULTS}")
    print(f"  candidate_pairs.tsv:  {CANDIDATE_PAIRS}")


if __name__ == "__main__":
    main()