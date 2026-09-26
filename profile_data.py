#!/usr/bin/env python3
"""
Data profiling script for Entity Resolution challenge.
Uses Polars for efficient processing of large TSV files.
"""

import polars as pl
import os
import json
from pathlib import Path

DATA_DIR = Path("dataset")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = Path("project_context")

def profile_tsv(filepath, name, sample_size=10000):
    """Profile a TSV file using Polars streaming."""
    print(f"\n{'='*60}")
    print(f"Profiling: {name} ({filepath})")
    print(f"{'='*60}")
    
    # Read with streaming to handle large files
    try:
        df = pl.scan_csv(filepath, separator="\t", infer_schema_length=10000)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    # Get schema
    schema = df.collect_schema()
    print(f"\nSchema: {schema}")
    
    # Row count
    row_count = df.select(pl.len()).collect().item()
    print(f"Row count: {row_count:,}")
    
    # Column count
    col_count = len(schema)
    print(f"Column count: {col_count}")
    
    # Null counts and unique counts (sample for speed on large files)
    print("\nColumn analysis:")
    col_stats = {}
    for col in schema.names():
        col_df = df.select(pl.col(col))
        
        # Null count
        null_count = col_df.select(pl.col(col).null_count()).collect().item()
        null_pct = (null_count / row_count * 100) if row_count > 0 else 0
        
        # Unique count (sample for very large columns)
        try:
            unique_count = col_df.select(pl.col(col).n_unique()).collect().item()
        except:
            # Fallback: sample
            unique_count = col_df.sample(n=min(sample_size, row_count)).select(pl.col(col).n_unique()).collect().item()
        
        print(f"  {col}: nulls={null_count:,} ({null_pct:.2f}%), unique={unique_count:,}")
        
        col_stats[col] = {
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
            "dtype": str(schema[col])
        }
    
    # Sample rows
    print(f"\nFirst 5 rows:")
    sample = df.head(5).collect()
    print(sample)
    
    return {
        "name": name,
        "filepath": str(filepath),
        "row_count": row_count,
        "col_count": col_count,
        "schema": {k: str(v) for k, v in schema.items()},
        "column_stats": col_stats
    }

def profile_ground_truth(filepath):
    """Profile the ground truth file specifically."""
    print(f"\n{'='*60}")
    print(f"Profiling Ground Truth: {filepath}")
    print(f"{'='*60}")
    
    df = pl.scan_csv(filepath, separator="\t")
    schema = df.collect_schema()
    print(f"Schema: {schema}")
    
    row_count = df.select(pl.len()).collect().item()
    print(f"Row count: {row_count:,}")
    
    # Analyze matched_entity_ids column
    print("\nAnalyzing matched_entity_ids...")
    gt_df = df.collect()
    
    # Count empty matches (singletons)
    empty_matches = gt_df.filter(pl.col("matched_entity_ids") == "").height
    print(f"Singletons (empty matches): {empty_matches:,} ({empty_matches/row_count*100:.2f}%)")
    
    # Parse matched IDs
    all_matched = []
    match_counts = []
    for row in gt_df.iter_rows(named=True):
        ids_str = row["matched_entity_ids"]
        if ids_str:
            ids = ids_str.split(",")
            match_counts.append(len(ids))
            all_matched.extend(ids)
        else:
            match_counts.append(0)
    
    print(f"Total matched IDs referenced: {len(all_matched):,}")
    print(f"Unique matched IDs: {len(set(all_matched)):,}")
    print(f"Avg matches per S1 entity (non-singleton): {sum(match_counts)/len([c for c in match_counts if c>0]):.2f}" if any(c>0 for c in match_counts) else "No matches")
    print(f"Max matches for single S1 entity: {max(match_counts)}")
    print(f"Match count distribution: {sorted(set(match_counts))[:20]}...")
    
    # Check source distribution of matched IDs
    s2_count = sum(1 for id in all_matched if id.startswith("S2-"))
    s3_count = sum(1 for id in all_matched if id.startswith("S3-"))
    s1_count = sum(1 for id in all_matched if id.startswith("S1-"))
    print(f"Matched ID sources: S2={s2_count:,}, S3={s3_count:,}, S1={s1_count:,}")
    
    return {
        "row_count": row_count,
        "singleton_count": empty_matches,
        "total_matched_refs": len(all_matched),
        "unique_matched_ids": len(set(all_matched)),
        "match_count_stats": {
            "min": min(match_counts),
            "max": max(match_counts),
            "avg_non_singleton": sum(match_counts)/len([c for c in match_counts if c>0]) if any(c>0 for c in match_counts) else 0
        },
        "matched_source_dist": {"S2": s2_count, "S3": s3_count, "S1": s1_count}
    }

def analyze_overlap(train_profiles):
    """Analyze cross-source overlap in training data."""
    print(f"\n{'='*60}")
    print("Cross-Source Overlap Analysis")
    print(f"{'='*60}")
    
    # Collect entity IDs from each source
    source_ids = {}
    for name, profile in train_profiles.items():
        if "source" in name.lower():
            filepath = profile["filepath"]
            df = pl.scan_csv(filepath, separator="\t").select("entity_id").collect()
            ids = set(df["entity_id"].to_list())
            source_ids[name] = ids
            print(f"{name}: {len(ids):,} unique entity_ids")
    
    # Check overlaps
    if "train_source1" in source_ids and "train_source2" in source_ids:
        overlap_12 = source_ids["train_source1"] & source_ids["train_source2"]
        print(f"S1 ∩ S2: {len(overlap_12):,} (should be 0)")
    
    if "train_source1" in source_ids and "train_source3" in source_ids:
        overlap_13 = source_ids["train_source1"] & source_ids["train_source3"]
        print(f"S1 ∩ S3: {len(overlap_13):,} (should be 0)")
    
    if "train_source2" in source_ids and "train_source3" in source_ids:
        overlap_23 = source_ids["train_source2"] & source_ids["train_source3"]
        print(f"S2 ∩ S3: {len(overlap_23):,} (should be 0)")
    
    # Check ground truth coverage
    gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    gt_df = pl.scan_csv(gt_path, separator="\t").collect()
    s1_in_gt = set(gt_df["source1_entity_id"].to_list())
    s1_in_data = source_ids.get("train_source1", set())
    print(f"S1 entities in ground truth: {len(s1_in_gt):,}")
    print(f"S1 entities in source1 data: {len(s1_in_data):,}")
    print(f"S1 in GT but not in data: {len(s1_in_gt - s1_in_data):,}")
    print(f"S1 in data but not in GT: {len(s1_in_data - s1_in_gt):,}")
    
    # Check matched IDs exist in source data
    all_matched = set()
    for row in gt_df.iter_rows(named=True):
        if row["matched_entity_ids"]:
            all_matched.update(row["matched_entity_ids"].split(","))
    
    s2_in_data = source_ids.get("train_source2", set())
    s3_in_data = source_ids.get("train_source3", set())
    all_source_ids = s2_in_data | s3_in_data
    
    print(f"Matched IDs found in source data: {len(all_matched & all_source_ids):,} / {len(all_matched):,}")
    print(f"Matched IDs NOT in source data: {len(all_matched - all_source_ids):,}")

def analyze_country_distribution(train_profiles):
    """Analyze country distribution across sources."""
    print(f"\n{'='*60}")
    print("Country Distribution Analysis")
    print(f"{'='*60}")
    
    for name, profile in train_profiles.items():
        if "source" in name.lower():
            filepath = profile["filepath"]
            try:
                df = pl.scan_csv(filepath, separator="\t").select("country").collect()
                counts = df["country"].value_counts()
                print(f"\n{name}:")
                for row in counts.iter_rows(named=True):
                    print(f"  {row['country']}: {row['count']:,}")
            except Exception as e:
                print(f"Error analyzing {name}: {e}")

def main():
    print("Starting data profiling...")
    
    # Profile all training files
    train_files = {
        "train_source1": TRAIN_DIR / "train_source1.tsv",
        "train_source2": TRAIN_DIR / "train_source2.tsv",
        "train_source3": TRAIN_DIR / "train_source3.tsv",
    }
    
    train_profiles = {}
    for name, path in train_files.items():
        train_profiles[name] = profile_tsv(path, name)
    
    # Profile ground truth
    gt_profile = profile_ground_truth(TRAIN_DIR / "train_ground_truth.tsv")
    train_profiles["train_ground_truth"] = gt_profile
    
    # Profile test files
    test_files = {
        "test_source1": TEST_DIR / "test_source1.tsv",
        "test_source2": TEST_DIR / "test_source2.tsv",
        "test_source3": TEST_DIR / "test_source3.tsv",
    }
    
    test_profiles = {}
    for name, path in test_files.items():
        test_profiles[name] = profile_tsv(path, name)
    
    # Cross-source overlap analysis
    analyze_overlap(train_profiles)
    
    # Country distribution
    analyze_country_distribution(train_profiles)
    analyze_country_distribution(test_profiles)
    
    # Save all profiles
    all_profiles = {
        "train": train_profiles,
        "test": test_profiles
    }
    
    with open(OUTPUT_DIR / "data_profiles.json", "w") as f:
        json.dump(all_profiles, f, indent=2, default=str)
    
    print(f"\n\nProfiles saved to {OUTPUT_DIR}/data_profiles.json")
    print("Profiling complete!")

if __name__ == "__main__":
    main()