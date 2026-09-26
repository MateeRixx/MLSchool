"""
Blocking / Candidate Generation for Entity Resolution.

Blocking Strategy (union of all keys):
1. Country exact match (mandatory first-level block)
2. Name token overlap: first 2 normalized tokens (within country)
3. Phonetic: Soundex of first name token (within country)
4. ZIP/PIN code extracted from address (within country)

Outputs:
- output/candidate_pairs_train.parquet
- output/candidate_pairs_test.parquet

Schema: source1_entity_id, candidate_entity_id, candidate_source
"""

import polars as pl
import jellyfish
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Iterator
from collections import defaultdict
import time
import gc

from utils import (
    load_source_tsv,
    load_ground_truth,
    normalize_name,
    extract_name_tokens,
    get_first_token,
    soundex_token,
    extract_zip_pin,
    compute_blocking_recall,
    normalize_source_lf,
    MAX_BUCKET_SIZE,
    SEED,
    OUTPUT_DIR,
)

# Paths
TRAIN_DIR = Path("dataset/train")
TEST_DIR = Path("dataset/test")

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

CANDIDATES_TRAIN_OUT = OUTPUT_DIR / "candidate_pairs_train.parquet"
CANDIDATES_TEST_OUT = OUTPUT_DIR / "candidate_pairs_test.parquet"

# Columns to read
SRC_COLS = ["entity_id", "business_name", "business_address", "country"]

# Countries in data
COUNTRIES = ["US", "India", "France"]

# Max candidates per S1 entity (safety valve)
MAX_CANDIDATES_PER_S1 = 5000


def normalize_source_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add normalized columns to source LazyFrame.
    """
    return lf.with_columns([
        pl.col("business_name").map_elements(normalize_name, return_dtype=pl.String).alias("name_norm"),
        pl.col("business_name").map_elements(lambda x: get_first_token(x), return_dtype=pl.String).alias("name_first_token"),
        pl.col("business_name").map_elements(lambda x: " ".join(extract_name_tokens(x, 2)), return_dtype=pl.String).alias("name_first_2_tokens"),
        pl.col("business_name").map_elements(lambda x: soundex_token(get_first_token(x)), return_dtype=pl.String).alias("name_soundex"),
        pl.struct(["business_address", "country"]).map_elements(
            lambda x: extract_zip_pin(x["business_address"], x["country"]), 
            return_dtype=pl.String
        ).alias("zip_pin"),
    ])


def build_blocking_index_df(
    df: pl.DataFrame,
    key_col: str,
    source_prefix: str,
    max_bucket: int = MAX_BUCKET_SIZE,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build blocking index from a collected DataFrame: key -> list of (entity_id, source_prefix).
    Filters out buckets larger than max_bucket.
    """
    if key_col not in df.columns:
        return {}
    
    index = defaultdict(list)
    # Use iter_rows for memory efficiency
    for row in df.iter_rows(named=True):
        key = row[key_col]
        if key and key != "null" and key != "":
            index[key].append((row["entity_id"], source_prefix))
    
    # Filter oversized buckets
    filtered = {k: v for k, v in index.items() if len(v) <= max_bucket}
    skipped = len(index) - len(filtered)
    if skipped > 0:
        print(f"    [Index {key_col}] Skipped {skipped} oversized buckets (> {max_bucket})")
    
    return filtered


def generate_candidates_from_index_batch(
    s1_df: pl.DataFrame,
    s1_key_col: str,
    index: Dict[str, List[Tuple[str, str]]],
    max_per_s1: int = MAX_CANDIDATES_PER_S1,
) -> List[Tuple[str, str, str]]:
    """
    Generate candidate pairs for a batch of S1 entities.
    Limits candidates per S1 to max_per_s1.
    """
    candidates = []
    s1_counts = defaultdict(int)
    
    for row in s1_df.iter_rows(named=True):
        s1_id = row["entity_id"]
        key = row[s1_key_col]
        if key and key in index:
            for cand_id, cand_source in index[key]:
                if s1_counts[s1_id] < max_per_s1:
                    candidates.append((s1_id, cand_id, cand_source))
                    s1_counts[s1_id] += 1
                else:
                    break
    
    return candidates


def process_country_batch(
    country: str,
    s1_lf: pl.LazyFrame,
    s2_lf: pl.LazyFrame,
    s3_lf: pl.LazyFrame,
) -> List[Tuple[str, str, str]]:
    """
    Process blocking for a single country.
    Returns list of (source1_entity_id, candidate_entity_id, candidate_source)
    """
    print(f"  [Country: {country}] Processing...")
    start = time.time()
    
    # Filter to this country
    s1_country = s1_lf.filter(pl.col("country") == country)
    s2_country = s2_lf.filter(pl.col("country") == country)
    s3_country = s3_lf.filter(pl.col("country") == country)
    
    # Collect country-specific data (much smaller than full dataset)
    print(f"    Collecting country data...")
    s1_df = s1_country.collect()
    s2_df = s2_country.collect()
    s3_df = s3_country.collect()
    
    s1_count = len(s1_df)
    s2_count = len(s2_df)
    s3_count = len(s3_df)
    print(f"    S1: {s1_count:,} | S2: {s2_count:,} | S3: {s3_count:,}")
    
    if s1_count == 0 or (s2_count == 0 and s3_count == 0):
        print(f"    No data for {country}, skipping")
        return []
    
    all_candidates = []
    
    # Key 1: Name first 2 tokens
    print(f"    Building name token index...")
    s2_idx_name = build_blocking_index_df(s2_df, "name_first_2_tokens", "S2")
    s3_idx_name = build_blocking_index_df(s3_df, "name_first_2_tokens", "S3")
    all_candidates.extend(generate_candidates_from_index_batch(s1_df, "name_first_2_tokens", s2_idx_name))
    all_candidates.extend(generate_candidates_from_index_batch(s1_df, "name_first_2_tokens", s3_idx_name))
    
    # Key 2: Phonetic (Soundex)
    print(f"    Building phonetic index...")
    s2_idx_phone = build_blocking_index_df(s2_df, "name_soundex", "S2")
    s3_idx_phone = build_blocking_index_df(s3_df, "name_soundex", "S3")
    all_candidates.extend(generate_candidates_from_index_batch(s1_df, "name_soundex", s2_idx_phone))
    all_candidates.extend(generate_candidates_from_index_batch(s1_df, "name_soundex", s3_idx_phone))
    
    # Key 3: ZIP/PIN
    print(f"    Building ZIP/PIN index...")
    s2_idx_zip = build_blocking_index_df(s2_df, "zip_pin", "S2")
    s3_idx_zip = build_blocking_index_df(s3_df, "zip_pin", "S3")
    all_candidates.extend(generate_candidates_from_index_batch(s1_df, "zip_pin", s2_idx_zip))
    all_candidates.extend(generate_candidates_from_index_batch(s1_df, "zip_pin", s3_idx_zip))
    
    print(f"    Generated {len(all_candidates):,} candidates in {time.time()-start:.1f}s")
    
    # Free memory
    del s1_df, s2_df, s3_df
    gc.collect()
    
    return all_candidates


def run_blocking(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    gt_path: Optional[Path] = None,
    output_path: Path = None,
    is_train: bool = True,
) -> pl.DataFrame:
    """
    Run full blocking pipeline for train or test set.
    
    Processes by country batches for memory efficiency.
    """
    print(f"\n{'='*60}")
    print(f"BLOCKING PIPELINE {'(TRAIN)' if is_train else '(TEST)'}")
    print(f"{'='*60}")
    
    total_start = time.time()
    
    # Load all sources as LazyFrames
    print("\n[Loading] Reading source files...")
    s1_lf = load_source_tsv(str(s1_path))
    s2_lf = load_source_tsv(str(s2_path))
    s3_lf = load_source_tsv(str(s3_path))
    
    # Add normalized columns
    print("[Loading] Computing normalized columns...")
    s1_lf = normalize_source_lf(s1_lf)
    s2_lf = normalize_source_lf(s2_lf)
    s3_lf = normalize_source_lf(s3_lf)
    
    # Get S1 row count
    s1_count = s1_lf.select(pl.len()).collect().item()
    print(f"[Loading] S1 entities: {s1_count:,}")
    
    # Process by country batches
    all_candidates = []
    for country in COUNTRIES:
        country_candidates = process_country_batch(country, s1_lf, s2_lf, s3_lf)
        all_candidates.extend(country_candidates)
    
    # Deduplicate candidates
    print("\n[Deduplication] Removing duplicate candidate pairs...")
    unique_pairs = set(all_candidates)
    print(f"  Before: {len(all_candidates):,} | After: {len(unique_pairs):,}")
    del all_candidates
    gc.collect()
    
    # Convert to DataFrame
    candidates_df = pl.DataFrame(
        list(unique_pairs),
        schema=["source1_entity_id", "candidate_entity_id", "candidate_source"],
        orient="row"
    )
    del unique_pairs
    gc.collect()
    
    # Ensure all S1 entities have at least one row (even if no candidates)
    print("\n[Padding] Ensuring all S1 entities present...")
    s1_ids = s1_lf.select("entity_id").collect()["entity_id"].to_list()
    existing_s1 = set(candidates_df["source1_entity_id"].to_list())
    missing_s1 = [sid for sid in s1_ids if sid not in existing_s1]
    
    if missing_s1:
        print(f"  Adding {len(missing_s1)} S1 entities with no candidates...")
        empty_rows = pl.DataFrame({
            "source1_entity_id": missing_s1,
            "candidate_entity_id": [""] * len(missing_s1),
            "candidate_source": [""] * len(missing_s1),
        })
        candidates_df = pl.concat([candidates_df, empty_rows])
    
    # Write output
    print(f"\n[Output] Writing to {output_path}...")
    candidates_df.write_parquet(output_path)
    print(f"  Rows written: {len(candidates_df):,}")
    print(f"  Unique S1 entities: {candidates_df['source1_entity_id'].n_unique():,}")
    
    # Stats
    non_empty = candidates_df.filter(pl.col("candidate_entity_id") != "")
    print(f"  Non-empty candidates: {len(non_empty):,}")
    if len(non_empty) > 0:
        avg_per_s1 = len(non_empty) / candidates_df["source1_entity_id"].n_unique()
        print(f"  Avg candidates per S1: {avg_per_s1:.1f}")
    
    # Compute blocking recall on training set
    if is_train and gt_path:
        print("\n[Evaluation] Computing blocking recall...")
        gt_df = load_ground_truth(str(gt_path))
        recall = compute_blocking_recall(non_empty, gt_df)
        print(f"  BLOCKING RECALL: {recall:.4f} ({recall*100:.2f}%)")
    
    print(f"\n[Complete] Total time: {time.time()-total_start:.1f}s")
    
    return candidates_df


def main():
    """Run blocking for both train and test sets."""
    print("="*60)
    print("ENTITY RESOLUTION - BLOCKING / CANDIDATE GENERATION")
    print("="*60)
    print(f"Seed: {SEED}")
    print(f"Max bucket size: {MAX_BUCKET_SIZE}")
    print(f"Max candidates per S1: {MAX_CANDIDATES_PER_S1}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Countries: {COUNTRIES}")
    
    # Train set
    print("\n" + "="*60)
    run_blocking(
        s1_path=TRAIN_SOURCE1,
        s2_path=TRAIN_SOURCE2,
        s3_path=TRAIN_SOURCE3,
        gt_path=TRAIN_GT,
        output_path=CANDIDATES_TRAIN_OUT,
        is_train=True,
    )
    
    # Test set
    print("\n" + "="*60)
    run_blocking(
        s1_path=TEST_SOURCE1,
        s2_path=TEST_SOURCE2,
        s3_path=TEST_SOURCE3,
        gt_path=None,
        output_path=CANDIDATES_TEST_OUT,
        is_train=False,
    )
    
    print("\n" + "="*60)
    print("BLOCKING COMPLETE")
    print("="*60)
    print(f"Train candidates: {CANDIDATES_TRAIN_OUT}")
    print(f"Test candidates:  {CANDIDATES_TEST_OUT}")


if __name__ == "__main__":
    main()