# Dataset Inventory

## Training Files

| File | Rows | Columns | Size | Description |
|------|------|---------|------|-------------|
| `train_source1.tsv` | 2,206,821 | 4 | 121 MB | Deduplicated reference source (S1) |
| `train_source2.tsv` | 5,034,616 | 4 | 467 MB | Noisy source A (S2) |
| `train_source3.tsv` | 5,285,603 | 4 | 481 MB | Noisy source B (S3) |
| `train_ground_truth.tsv` | 2,206,821 | 2 | 121 MB | S1 → S2/S3 match mappings |

**Total training records**: 12,527,040 across 3 sources
**Total training size**: ~1.19 GB

## Test Files

| File | Rows | Columns | Size | Description |
|------|------|---------|------|-------------|
| `test_source1.tsv` | 1,732,544 | 4 | 167 MB | S1 test entities (must predict for ALL) |
| `test_source2.tsv` | 4,887,273 | 4 | 486 MB | S2 test candidates |
| `test_source3.tsv` | 5,082,316 | 4 | 483 MB | S3 test candidates |

**Total test records**: 11,702,133 across 3 sources
**Total test size**: ~1.14 GB

## Schema (All Source Files)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `entity_id` | String | No | Unique ID with source prefix (S1-/S2-/S3-) |
| `business_name` | String | No | Business name (noisy, variations) |
| `business_address` | String | **Yes** (S2/S3) | Business address (noisy, partial, ~3% null in S2/S3) |
| `country` | String | No | Country label (open set: US, India, France...) |

## Ground Truth Schema

| Column | Type | Description |
|--------|------|-------------|
| `source1_entity_id` | String | S1 entity ID (foreign key to train_source1) |
| `matched_entity_ids` | String | Comma-separated S2/S3 IDs (empty = no matches) |

## Country Distribution

### Training (2 countries)
| Country | S1 Count | S2 Count | S3 Count |
|---------|----------|----------|----------|
| US | 1,323,633 (60.0%) | 3,016,817 (60.0%) | 3,170,056 (60.0%) |
| India | 883,188 (40.0%) | 2,017,799 (40.0%) | 2,115,547 (40.0%) |

### Test (3 countries — **France is new**)
| Country | S1 Count | S2 Count | S3 Count |
|---------|----------|----------|----------|
| US | 663,106 (38.3%) | 1,871,330 (38.3%) | 1,945,701 (38.3%) |
| India | 809,986 (46.8%) | 2,312,565 (47.3%) | 2,405,000 (47.3%) |
| **France** | **259,452 (15.0%)** | **703,378 (14.4%)** | **731,615 (14.4%)** |

## Data Quality Summary

| Field | S1 Train | S2 Train | S3 Train | S1 Test | S2 Test | S3 Test |
|-------|----------|----------|----------|---------|---------|---------|
| entity_id unique | 100% | 100% | 100% | 100% | 100% | 100% |
| business_name nulls | 0% | 0% | 0% | 0% | 0% | 0% |
| business_name unique | 69.7% | 87.4% | 88.0% | 71.5% | 88.2% | 89.0% |
| business_address nulls | 0% | **3.36%** | **3.33%** | 0% | **2.65%** | **2.68%** |
| business_address unique | 96.5% | 86.1% | 87.6% | 96.8% | 86.4% | 87.7% |
| country unique values | 2 | 2 | 2 | **3** | **3** | **3** |

## Cross-Source Integrity (Verified)
- ✅ Zero entity_id overlap between any source pair (S1∩S2, S1∩S3, S2∩S3)
- ✅ Ground truth covers all 2.2M S1 train entities (1:1 mapping)
- ✅ All 7.6M matched IDs in ground truth exist in source data
- ✅ No S1 self-references in matched_entity_ids
- ✅ Matched IDs only from S2/S3 (no S1- prefixes)

## Implications for Pipeline Design
1. **Country is a strong blocking key** — exact match reduces search space ~3× (train) to ~6.7× (test with France)
2. **Address nulls in S2/S3** — must handle missing addresses gracefully (name-only matching fallback)
3. **France in test only** — cannot train country-specific models for France; must use country-agnostic features or zero-shot generalization
4. **High name uniqueness** (~70-89%) suggests name-based blocking/feature will be discriminative
5. **Many-to-many** — pairwise classification + thresholding per S1 entity, not 1:1 assignment