"""
Blocking / Candidate Generation - Streaming Chunked Version
Uses pre-built indexes + S1 chunking for memory efficiency.
"""

import polars as pl
import jellyfish
from pathlib import Path
from typing import List, Optional
import time
import gc
import shutil

from utils import (
    load_source_tsv,
    load_ground_truth,
    normalize_source_lf,
    compute_blocking_recall,
    BASE_PATH,
    OUTPUT_DIR,
    SEED,
)

TRAIN_DIR = Path(BASE_PATH + "/dataset/train")
TEST_DIR = Path(BASE_PATH + "/dataset/test")

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

CANDIDATES_TRAIN_OUT = OUTPUT_DIR / "candidate_pairs_train.parquet"
CANDIDATES_TEST_OUT = OUTPUT_DIR / "candidate_pairs_test.parquet"

COUNTRIES = ["US", "India", "France"]
BLOCKING_KEYS = ["name_first_2_tokens", "name_soundex", "zip_pin"]
CHUNK_SIZE = 50000
MAX_CANDIDATES_PER_S1 = 2000
MAX_BUCKET_SIZE = 500


def build_and_save_indexes(
    s2_path: Path, s3_path: Path, index_base: Path
):
    """Build and save S2/S3 indexes partitioned by country + key."""
    print("[Index] Building S2/S3 indexes...")
    
    for src_name, src_path in [("S2", s2_path), ("S3", s3_path)]:
        lf = load_source_tsv(str(src_path))
        lf = normalize_source_lf(lf)
        
        for country in COUNTRIES:
            lf_c = lf.filter(pl.col("country") == country)
            
            for key_col in BLOCKING_KEYS:
                idx_dir = index_base / f"index_{src_name}_{country}_{key_col}"
                if idx_dir.exists():
                    print(f"  Skipping existing: {idx_dir}")
                    continue
                
                print(f"  Building {src_name}/{country}/{key_col}...")
                
                # Extract key + entity_id
                idx = lf_c.select(["entity_id", key_col]).filter(
                    pl.col(key_col).is_not_null() & (pl.col(key_col) != "")
                )
                
                # Filter oversized buckets
                counts = idx.group_by(key_col).agg(pl.len().alias("cnt"))
                valid_keys = counts.filter(pl.col("cnt") <= MAX_BUCKET_SIZE).select(key_col)
                idx = idx.join(valid_keys, on=key_col, how="inner")
                
                # Save
                idx_dir.mkdir(parents=True, exist_ok=True)
                idx.sink_parquet(idx_dir / "data.parquet")
                
                n = idx.select(pl.len()).collect().item()
                print(f"    Saved {n:,} entries")
        
        del lf
        gc.collect()


def process_s1_chunks(
    s1_path: Path,
    output_path: Path,
    is_train: bool,
    gt_path: Optional[Path] = None,
):
    """Process S1 in chunks, joining with pre-built indexes."""
    print(f"\n[Chunked Processing] S1: {s1_path}")
    
    s1_lf = load_source_tsv(str(s1_path))
    s1_lf = normalize_source_lf(s1_lf)
    s1_count = s1_lf.select(pl.len()).collect().item()
    print(f"  Total S1: {s1_count:,} | Chunk size: {CHUNK_SIZE}")
    
    # Clear output
    if output_path.exists():
        output_path.unlink()
    
    for chunk_start in range(0, s1_count, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, s1_count)
        print(f"  Chunk {chunk_start:,}-{chunk_end:,}...", end=" ", flush=True)
        
        s1_chunk = s1_lf.slice(chunk_start, chunk_end - chunk_start).collect()
        
        chunk_candidates = []
        
        for country in COUNTRIES:
            s1_c = s1_chunk.filter(pl.col("country") == country)
            if len(s1_c) == 0:
                continue
            
            for src_name in ["S2", "S3"]:
                for key_col in BLOCKING_KEYS:
                    idx_dir = OUTPUT_DIR / f"index_{src_name}_{country}_{key_col}"
                    if not idx_dir.exists():
                        continue
                    
                    idx_df = pl.read_parquet(idx_dir / "data.parquet")
                    
                    joined = s1_c.select(["entity_id", key_col]).join(
                        idx_df.rename({"entity_id": "candidate_entity_id"}),
                        on=key_col, how="inner"
                    ).with_columns(pl.lit(src_name).alias("candidate_source"))
                    
                    if len(joined) > 0:
                        chunk_candidates.append(joined.rename({"entity_id": "source1_entity_id"}))
        
        if chunk_candidates:
            chunk_df = pl.concat(chunk_candidates).unique()
            
            # Limit per S1
            chunk_df = chunk_df.with_columns(
                pl.int_range(pl.len()).over("source1_entity_id").alias("rn")
            ).filter(pl.col("rn") < MAX_CANDIDATES_PER_S1).drop("rn")
            
            # Append to output
            chunk_df.write_parquet(
                output_path,
                pyarrow_options={"compression": "snappy"},
                append=True if chunk_start > 0 else False
            )
            print(f"{len(chunk_df):,} candidates")
        else:
            print("0 candidates")
        
        del s1_chunk, chunk_candidates
        gc.collect()
    
    # Read final candidates
    candidates_df = pl.read_parquet(output_path)
    print(f"  Total candidates: {len(candidates_df):,}")
    
    # Pad missing S1 entities
    all_s1 = s1_lf.select("entity_id").collect()["entity_id"].to_list()
    existing = set(candidates_df["source1_entity_id"].to_list())
    missing = [s for s in all_s1 if s not in existing]
    
    if missing:
        print(f"  Padding {len(missing)} S1 entities with no candidates...")
        empty = pl.DataFrame({
            "source1_entity_id": missing,
            "candidate_entity_id": [""] * len(missing),
            "candidate_source": [""] * len(missing),
        })
        candidates_df = pl.concat([candidates_df, empty])
        candidates_df.write_parquet(output_path)
    
    # Compute recall on train
    if is_train and gt_path:
        print("\n[Evaluation] Computing blocking recall...")
        gt_df = load_ground_truth(str(gt_path))
        non_empty = candidates_df.filter(pl.col("candidate_entity_id") != "")
        recall = compute_blocking_recall(non_empty, gt_df)
        print(f"  BLOCKING RECALL: {recall:.4f} ({recall*100:.2f}%)")
    
    return candidates_df


def run_blocking(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    gt_path: Optional[Path] = None,
    output_path: Path = None,
    is_train: bool = True,
) -> pl.DataFrame:
    """Main blocking pipeline."""
    print(f"\n{'='*60}")
    print(f"BLOCKING PIPELINE {'(TRAIN)' if is_train else '(TEST)'} [CHUNKED STREAMING]")
    print(f"{'='*60}")
    
    total_start = time.time()
    
    index_base = OUTPUT_DIR / "blocking_indexes"
    
    # Step 1: Build indexes (only once)
    build_and_save_indexes(s2_path, s3_path, index_base)
    
    # Step 2: Process S1 chunks
    candidates_df = process_s1_chunks(s1_path, output_path, is_train, gt_path)
    
    print(f"\n[Complete] Total time: {time.time()-total_start:.1f}s")
    print(f"  Output: {output_path}")
    print(f"  Unique S1: {candidates_df['source1_entity_id'].n_unique():,}")
    
    return candidates_df


def main():
    print("="*60)
    print("ENTITY RESOLUTION - BLOCKING / CANDIDATE GENERATION")
    print("="*60)
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Max candidates/S1: {MAX_CANDIDATES_PER_S1}")
    print(f"Max bucket size: {MAX_BUCKET_SIZE}")
    
    # Train
    run_blocking(
        s1_path=TRAIN_SOURCE1,
        s2_path=TRAIN_SOURCE2,
        s3_path=TRAIN_SOURCE3,
        gt_path=TRAIN_GT,
        output_path=CANDIDATES_TRAIN_OUT,
        is_train=True,
    )
    
    # Test
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


if __name__ == "__main__":
    main()