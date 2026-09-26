"""
Model Training for Entity Resolution.

Trains XGBoost classifier on pairwise features.
Optimizes threshold for macro F0.5 on validation split.
"""

import polars as pl
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import json

from utils import (
    load_ground_truth,
    parse_matched_ids,
    OUTPUT_DIR,
    SEED,
)

# Paths
FEATURES_TRAIN = OUTPUT_DIR / "features_train.parquet"
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
CALIBRATED_MODEL_PATH = MODEL_DIR / "xgb_calibrated.pkl"
THRESHOLD_PATH = MODEL_DIR / "thresholds.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Feature columns (exclude identifiers and label)
ID_COLS = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
LABEL_COL = "label"


def load_features(path: Path) -> pl.DataFrame:
    """Load features parquet."""
    return pl.read_parquet(path)


def prepare_train_data(
    features_df: pl.DataFrame,
    val_split: float = 0.1,
    seed: int = SEED,
) -> Tuple:
    """
    Prepare train/validation split by S1 entity (not by row!).
    This prevents data leakage.
    """
    print("[Data Prep] Splitting by S1 entity...")
    
    # Get unique S1 entities
    s1_entities = features_df["source1_entity_id"].unique().to_list()
    print(f"  Unique S1 entities: {len(s1_entities):,}")
    
    # Split entities
    train_entities, val_entities = train_test_split(
        s1_entities, test_size=val_split, random_state=seed
    )
    print(f"  Train entities: {len(train_entities):,} | Val entities: {len(val_entities):,}")
    
    # Filter rows
    train_df = features_df.filter(pl.col("source1_entity_id").is_in(train_entities))
    val_df = features_df.filter(pl.col("source1_entity_id").is_in(val_entities))
    
    # Feature columns
    feat_cols = [c for c in features_df.columns if c not in ID_COLS + [LABEL_COL]]
    print(f"  Feature columns: {len(feat_cols)}")
    
    # Convert to numpy
    X_train = train_df.select(feat_cols).to_numpy()
    y_train = train_df[LABEL_COL].to_numpy()
    X_val = val_df.select(feat_cols).to_numpy()
    y_val = val_df[LABEL_COL].to_numpy()
    
    print(f"  Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,}")
    print(f"  Train pos: {y_train.sum():,} | Val pos: {y_val.sum():,}")
    
    return X_train, y_train, X_val, y_val, val_df, feat_cols


def compute_macro_f05(y_true: np.ndarray, y_pred: np.ndarray, s1_ids: np.ndarray) -> float:
    """
    Compute macro-averaged F0.5 score.
    F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    unique_s1 = np.unique(s1_ids)
    f05_scores = []
    
    for s1 in unique_s1:
        mask = s1_ids == s1
        yt = y_true[mask]
        yp = y_pred[mask]
        
        if yt.sum() == 0:
            # Singleton: correct if no predictions
            f05 = 1.0 if yp.sum() == 0 else 0.0
        else:
            tp = ((yt == 1) & (yp == 1)).sum()
            fp = ((yt == 0) & (yp == 1)).sum()
            fn = ((yt == 1) & (yp == 0)).sum()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if prec + rec > 0:
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            else:
                f05 = 0.0
        
        f05_scores.append(f05)
    
    return np.mean(f05_scores)


def find_optimal_threshold(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_s1_ids: np.ndarray,
    thresholds: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Find threshold maximizing macro F0.5 on validation set.
    
    Returns (best_threshold, best_f05)
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.02)
    
    # Get predicted probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    
    best_thresh = 0.5
    best_f05 = 0.0
    
    print("\n[Threshold Search]")
    for thresh in thresholds:
        y_pred = (val_probs >= thresh).astype(int)
        f05 = compute_macro_f05(y_val, y_pred, val_s1_ids)
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
        
        print(f"  thresh={thresh:.2f}: F0.5={f05:.4f}")
    
    print(f"  BEST: thresh={best_thresh:.2f}, F0.5={best_f05:.4f}")
    return best_thresh, best_f05


def find_per_country_thresholds(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_df: pl.DataFrame,
    feat_cols: List[str],
    thresholds: np.ndarray = None,
) -> Dict[str, float]:
    """
    Find optimal threshold per country.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.02)
    
    val_probs = model.predict_proba(X_val)[:, 1]
    val_s1_ids = val_df["source1_entity_id"].to_numpy()
    
    # Need country info - join back
    # For now, use global threshold
    # TODO: Add country to features or join
    
    best_thresh, best_f05 = find_optimal_threshold(
        model, X_val, y_val, val_s1_ids, thresholds
    )
    
    return {"global": best_thresh, "US": best_thresh, "India": best_thresh, "France": best_thresh}


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_s1_ids: np.ndarray,
) -> Tuple:
    """
    Train XGBoost with early stopping.
    Returns (model, calibrated_model, best_threshold)
    """
    print("\n[Training] XGBoost...")
    
    # Calculate scale_pos_weight for imbalance
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"  Class balance: pos={pos:,}, neg={neg:,}, scale_pos_weight={scale_pos_weight:.1f}")
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=30,
    )
    
    # Train with validation for early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best score: {model.best_score:.4f}")
    
    # Calibrate probabilities
    print("\n[Calibration] Fitting CalibratedClassifierCV...")
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    
    # Find optimal threshold on calibrated probabilities
    print("\n[Threshold Optimization]")
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_thresh, best_f05 = find_optimal_threshold(
        calibrated, X_val, y_val, val_s1_ids, thresholds
    )
    
    return model, calibrated, best_thresh


def evaluate_model(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_s1_ids: np.ndarray,
    threshold: float,
) -> Dict:
    """Evaluate model with given threshold."""
    probs = model.predict_proba(X_val)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    
    # Per-entity metrics
    f05 = compute_macro_f05(y_val, y_pred, val_s1_ids)
    
    # Overall metrics
    tp = ((y_val == 1) & (y_pred == 1)).sum()
    fp = ((y_val == 0) & (y_pred == 1)).sum()
    fn = ((y_val == 1) & (y_pred == 0)).sum()
    tn = ((y_val == 0) & (y_pred == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Singleton stats
    unique_s1 = np.unique(val_s1_ids)
    singleton_count = 0
    singleton_correct = 0
    for s1 in unique_s1:
        mask = val_s1_ids == s1
        if y_val[mask].sum() == 0:
            singleton_count += 1
            if y_pred[mask].sum() == 0:
                singleton_correct += 1
    
    singleton_acc = singleton_correct / singleton_count if singleton_count > 0 else 0
    
    metrics = {
        "macro_f05": float(f05),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "singleton_accuracy": float(singleton_acc),
        "singleton_count": int(singleton_count),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    
    return metrics


def main():
    """Train model and save artifacts."""
    print("="*60)
    print("MODEL TRAINING - XGBOOST")
    print("="*60)
    
    # Load features
    print(f"\n[Loading] Features from {FEATURES_TRAIN}...")
    features_df = load_features(FEATURES_TRAIN)
    print(f"  Rows: {len(features_df):,} | Cols: {len(features_df.columns)}")
    
    # Prepare train/val split
    X_train, y_train, X_val, y_val, val_df, feat_cols = prepare_train_data(features_df)
    
    # Train
    model, calibrated, best_thresh = train_xgboost(
        X_train, y_train, X_val, y_val, 
        val_df["source1_entity_id"].to_numpy()
    )
    
    # Evaluate
    metrics = evaluate_model(calibrated, X_val, y_val, 
                            val_df["source1_entity_id"].to_numpy(), best_thresh)
    
    print(f"\n[Results]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Save model artifacts
    print(f"\n[Saving] Model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    
    print(f"[Saving] Calibrated model to {CALIBRATED_MODEL_PATH}")
    joblib.dump(calibrated, CALIBRATED_MODEL_PATH)
    
    print(f"[Saving] Thresholds to {THRESHOLD_PATH}")
    thresholds = {"global": best_thresh, "US": best_thresh, "India": best_thresh, "France": best_thresh}
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)
    
    print(f"[Saving] Metrics to {METRICS_PATH}")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[Saving] Feature columns to {MODEL_DIR / 'feature_cols.json'}")
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f)
    
    print("\nTRAINING COMPLETE")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Calibrated: {CALIBRATED_MODEL_PATH}")
    print(f"  Threshold: {best_thresh:.3f}")
    print(f"  Val Macro F0.5: {metrics['macro_f05']:.4f}")


if __name__ == "__main__":
    main()