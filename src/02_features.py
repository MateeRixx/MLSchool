"""
Feature Extraction for Entity Resolution.

Extracts pairwise features for (S1, candidate) pairs.
Features include name similarity, address similarity, and meta features.
"""

import polars as pl
import rapidfuzz
from rapidfuzz import fuzz, process
import jellyfish
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import numpy as np

from utils import (
    load_source_tsv,
    load_ground_truth,
    parse_matched_ids,
    normalize_source_lf,
    OUTPUT_DIR,
)

# Paths
TRAIN_DIR = Path("dataset/train")
TEST_DIR = Path("dataset/test")

CANDIDATES_TRAIN = OUTPUT_DIR / "candidate_pairs_train.parquet"
CANDIDATES_TEST = OUTPUT_DIR / "candidate_pairs_test.parquet"

FEATURES_TRAIN_OUT = OUTPUT_DIR / "features_train.parquet"
FEATURES_TEST_OUT = OUTPUT_DIR / "features_test.parquet"


def jaccard_tokens(s1: str, s2: str) -> float:
    """Jaccard similarity on word tokens."""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (0-1)."""
    if not s1 or not s2:
        return 0.0
    return rapidfuzz.fuzz.ratio(s1, s2) / 100.0


def token_overlap_ratio(s1: str, s2: str) -> float:
    """Token overlap ratio."""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def prefix_match(s1: str, s2: str) -> int:
    """First token exact match."""
    if not s1 or not s2:
        return 0
    t1 = s1.split()[0] if s1.split() else ""
    t2 = s2.split()[0] if s2.split() else ""
    return int(t1 == t2)


def suffix_match(s1: str, s2: str) -> int:
    """Last token exact match."""
    if not s1 or not s2:
        return 0
    t1 = s1.split()[-1] if s1.split() else ""
    t2 = s2.split()[-1] if s2.split() else ""
    return int(t1 == t2)


def tfidf_cosine_similarity(s1: str, s2: str, vocab: dict = None) -> float:
    """
    Simplified TF-IDF cosine using word overlap with IDF weighting.
    In production, use fitted TF-IDF vectorizer.
    """
    if not s1 or not s2:
        return 0.0
    # Placeholder - returns token overlap as proxy
    return token_overlap_ratio(s1, s2)


def phonetic_match(s1: str, s2: str) -> int:
    """Soundex match on first token."""
    if not s1 or not s2:
        return 0
    t1 = s1.split()[0] if s1.split() else ""
    t2 = s2.split()[0] if s2.split() else ""
    try:
        return int(jellyfish.soundex(t1) == jellyfish.soundex(t2))
    except:
        return int(t1[:4] == t2[:4])


def extract_address_tokens(address: str) -> set:
    """Extract meaningful tokens from address."""
    if not address:
        return set()
    # Remove punctuation, split, filter
    tokens = address.lower().replace(",", " ").replace(".", " ").split()
    # Keep alphanumeric tokens > 2 chars
    return {t for t in tokens if t.isalnum() and len(t) > 2}


def address_token_overlap(addr1: str, addr2: str) -> float:
    """Jaccard on address tokens."""
    set1 = extract_address_tokens(addr1)
    set2 = extract_address_tokens(addr2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def street_number_match(addr1: str, addr2: str) -> int:
    """Check if street numbers match."""
    if not addr1 or not addr2:
        return 0
    import re
    nums1 = re.findall(r"\b\d+\b", addr1)
    nums2 = re.findall(r"\b\d+\b", addr2)
    return int(bool(set(nums1) & set(nums2)))


def zip_pin_match(zip1: str, zip2: str) -> int:
    """Exact ZIP/PIN match."""
    if not zip1 or not zip2 or zip1 == "null" or zip2 == "null":
        return 0
    return int(zip1 == zip2)


def zip_prefix_match(zip1: str, zip2: str, prefix_len: int = 3) -> int:
    """ZIP/PIN prefix match."""
    if not zip1 or not zip2 or zip1 == "null" or zip2 == "null":
        return 0
    if len(zip1) >= prefix_len and len(zip2) >= prefix_len:
        return int(zip1[:prefix_len] == zip2[:prefix_len])
    return 0


def city_match(addr1: str, addr2: str) -> int:
    """Simple city name match (heuristic)."""
    if not addr1 or not addr2:
        return 0
    # Extract potential city tokens (capitalized words)
    import re
    words1 = set(re.findall(r"\b[A-Z][a-z]+\b", addr1))
    words2 = set(re.findall(r"\b[A-Z][a-z]+\b", addr2))
    # Filter common non-city words
    stopwords = {"Road", "Street", "Avenue", "Drive", "Lane", "Boulevard", "Court", "Place",
                 "North", "South", "East", "West", "Near", "Opposite", "Behind"}
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    return int(bool(words1 & words2))


def compute_pair_features(
    s1_row: Dict,
    cand_row: Dict,
) -> Dict[str, float]:
    """
    Compute all features for a single (S1, candidate) pair.
    
    Args:
        s1_row: Dict with keys [entity_id, name_norm, business_address, country, zip_pin, name_first_2_tokens, name_soundex]
        cand_row: Same structure
    
    Returns:
        Dict of feature_name -> value
    """
    s1_name = s1_row.get("name_norm", "") or ""
    cand_name = cand_row.get("name_norm", "") or ""
    s1_addr = s1_row.get("business_address", "") or ""
    cand_addr = cand_row.get("business_address", "") or ""
    s1_zip = s1_row.get("zip_pin", "") or ""
    cand_zip = cand_row.get("zip_pin", "") or ""
    s1_country = s1_row.get("country", "") or ""
    cand_country = cand_row.get("country", "") or ""
    
    features = {}
    
    # Name features
    features["name_jaccard"] = jaccard_tokens(s1_name, cand_name)
    features["name_levenshtein"] = levenshtein_ratio(s1_name, cand_name)
    features["name_token_overlap"] = token_overlap_ratio(s1_name, cand_name)
    features["name_prefix_match"] = prefix_match(s1_name, cand_name)
    features["name_suffix_match"] = suffix_match(s1_name, cand_name)
    features["name_phonetic_match"] = phonetic_match(s1_name, cand_name)
    
    # Address features
    features["addr_token_overlap"] = address_token_overlap(s1_addr, cand_addr)
    features["street_num_match"] = street_number_match(s1_addr, cand_addr)
    features["zip_pin_match"] = zip_pin_match(s1_zip, cand_zip)
    features["zip_prefix_match"] = zip_prefix_match(s1_zip, cand_zip)
    features["city_match"] = city_match(s1_addr, cand_addr)
    features["addr_null"] = int(not cand_addr or cand_addr == "null")
    
    # Meta features
    features["country_match"] = int(s1_country == cand_country)
    features["source_pair_S1S2"] = int(cand_row.get("candidate_source", "") == "S2")
    features["source_pair_S1S3"] = int(cand_row.get("candidate_source", "") == "S3")
    
    # Length ratios
    s1_name_len = len(s1_name)
    cand_name_len = len(cand_name)
    features["name_len_ratio"] = min(s1_name_len, cand_name_len) / max(s1_name_len, cand_name_len) if max(s1_name_len, cand_name_len) > 0 else 0
    
    s1_addr_len = len(s1_addr)
    cand_addr_len = len(cand_addr)
    features["addr_len_ratio"] = min(s1_addr_len, cand_addr_len) / max(s1_addr_len, cand_addr_len) if max(s1_addr_len, cand_addr_len) > 0 else 0
    
    return features


def extract_features_batch(
    candidates_df: pl.DataFrame,
    s1_df: pl.DataFrame,
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame,
    is_train: bool = True,
    gt_df: pl.DataFrame = None,
) -> pl.DataFrame:
    """
    Extract features for all candidate pairs.
    
    Returns DataFrame with features + label (if train).
    """
    print(f"[Feature Extraction] Processing {len(candidates_df):,} candidate pairs...")
    start = time.time()
    
    # Build lookup dictionaries for fast access
    print("  Building lookup dictionaries...")
    s1_lookup = {row["entity_id"]: row for row in s1_df.iter_rows(named=True)}
    s2_lookup = {row["entity_id"]: row for row in s2_df.iter_rows(named=True)}
    s3_lookup = {row["entity_id"]: row for row in s3_df.iter_rows(named=True)}
    
    # Prepare ground truth lookup if training
    gt_lookup = {}
    if is_train and gt_df is not None:
        for row in gt_df.iter_rows(named=True):
            s1_id = row["source1_entity_id"]
            matched = parse_matched_ids(row["matched_entity_ids"])
            gt_lookup[s1_id] = set(matched)
    
    # Extract features row by row (vectorized would be better but this works)
    print("  Computing features...")
    feature_rows = []
    
    for i, row in enumerate(candidates_df.iter_rows(named=True)):
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        cand_source = row["candidate_source"]
        
        if not cand_id or cand_id == "":
            # Empty candidate (singleton padding) - skip feature extraction
            continue
        
        # Get source rows
        s1_row = s1_lookup.get(s1_id)
        if cand_source == "S2":
            cand_row = s2_lookup.get(cand_id)
        else:
            cand_row = s3_lookup.get(cand_id)
        
        if not s1_row or not cand_row:
            continue
        
        # Add candidate source to cand_row for meta features
        cand_row = dict(cand_row)
        cand_row["candidate_source"] = cand_source
        
        # Compute features
        feats = compute_pair_features(s1_row, cand_row)
        
        # Add label for training
        if is_train:
            label = int(cand_id in gt_lookup.get(s1_id, set()))
            feats["label"] = label
        
        # Add identifiers
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = cand_id
        feats["candidate_source"] = cand_source
        
        feature_rows.append(feats)
        
        if (i + 1) % 100000 == 0:
            print(f"    Processed {i+1:,} pairs...")
    
    features_df = pl.DataFrame(feature_rows)
    print(f"  Extracted features for {len(features_df):,} pairs in {time.time()-start:.1f}s")
    
    return features_df


def run_feature_extraction(
    candidates_path: Path,
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    output_path: Path,
    is_train: bool = True,
    gt_path: Path = None,
):
    """Run feature extraction pipeline."""
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION {'(TRAIN)' if is_train else '(TEST)'}")
    print(f"{'='*60}")
    
    # Load candidates
    print(f"[Loading] Candidates from {candidates_path}...")
    candidates_df = pl.read_parquet(candidates_path)
    
    # Load source data (need normalized columns)
    print("[Loading] Source data...")
    s1_lf = load_source_tsv(str(s1_path))
    s2_lf = load_source_tsv(str(s2_path))
    s3_lf = load_source_tsv(str(s3_path))
    
    from utils import normalize_source_lf
    s1_lf = normalize_source_lf(s1_lf)
    s2_lf = normalize_source_lf(s2_lf)
    s3_lf = normalize_source_lf(s3_lf)
    
    print("  Collecting source data...")
    s1_df = s1_lf.collect()
    s2_df = s2_lf.collect()
    s3_df = s3_lf.collect()
    print(f"  S1: {len(s1_df):,} | S2: {len(s2_df):,} | S3: {len(s3_df):,}")
    
    # Load ground truth if training
    gt_df = None
    if is_train and gt_path:
        gt_df = load_ground_truth(str(gt_path))
    
    # Extract features
    features_df = extract_features_batch(
        candidates_df, s1_df, s2_df, s3_df, is_train, gt_df
    )
    
    # Write output
    print(f"[Output] Writing to {output_path}...")
    features_df.write_parquet(output_path)
    print(f"  Rows: {len(features_df):,} | Cols: {len(features_df.columns)}")
    
    if is_train and "label" in features_df.columns:
        pos = features_df.filter(pl.col("label") == 1).height
        neg = features_df.filter(pl.col("label") == 0).height
        print(f"  Positive: {pos:,} | Negative: {neg:,} | Ratio: 1:{neg/pos:.1f}")


def main():
    """Run feature extraction for train and test."""
    print("="*60)
    print("FEATURE EXTRACTION PIPELINE")
    print("="*60)
    
    # Train
    run_feature_extraction(
        candidates_path=CANDIDATES_TRAIN,
        s1_path=TRAIN_DIR / "train_source1.tsv",
        s2_path=TRAIN_DIR / "train_source2.tsv",
        s3_path=TRAIN_DIR / "train_source3.tsv",
        output_path=FEATURES_TRAIN_OUT,
        is_train=True,
        gt_path=TRAIN_DIR / "train_ground_truth.tsv",
    )
    
    # Test
    run_feature_extraction(
        candidates_path=CANDIDATES_TEST,
        s1_path=TEST_DIR / "test_source1.tsv",
        s2_path=TEST_DIR / "test_source2.tsv",
        s3_path=TEST_DIR / "test_source3.tsv",
        output_path=FEATURES_TEST_OUT,
        is_train=False,
        gt_path=None,
    )
    
    print("\nFEATURE EXTRACTION COMPLETE")


if __name__ == "__main__":
    main()