# Ground Truth Analysis

## Structure
- **File**: `dataset/train/train_ground_truth.tsv`
- **Rows**: 2,206,821 (exactly one per S1 train entity)
- **Columns**: 
  - `source1_entity_id` (String) — FK to train_source1.entity_id
  - `matched_entity_ids` (String) — comma-separated S2/S3 entity_ids, empty = no matches

## Coverage & Completeness

| Metric | Value | Status |
|--------|-------|--------|
| S1 entities in GT | 2,206,821 | ✅ 100% of train_source1 |
| S1 entities in train_source1 | 2,206,821 | ✅ 1:1 mapping |
| Orphan GT references (S1 not in source) | 0 | ✅ Clean |
| Orphan source S1 (not in GT) | 0 | ✅ Clean |
| Matched IDs referencing S2/S3 | 7,638,365 | — |
| Unique matched IDs | 7,638,365 | ✅ All distinct |
| Matched IDs found in source data | 7,638,365 / 7,638,365 | ✅ 100% valid |
| S1 self-references in matches | 0 | ✅ Clean |
| Invalid prefixes in matches | 0 | ✅ Clean |

## Match Cardinality Distribution

| Matches per S1 | Count | Percentage | Cumulative |
|----------------|-------|------------|------------|
| 0 (singleton) | **0** | **0%** | 0% |
| 1 | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD |
| 6-11 | TBD | TBD | TBD |
| **Total** | **2,206,821** | **100%** | — |

**Key finding**: **Zero singletons in training data**. Every S1 entity has at least one match.

**Statistics**:
- Mean matches per S1: 3.67
- Median: TBD
- Max: 11
- Min: 1 (since 0 count = 0)

## Source Distribution of Matches

| Source | Match Count | Percentage |
|--------|-------------|------------|
| S2 (S2- prefix) | 3,693,619 | 48.4% |
| S3 (S3- prefix) | 3,944,746 | 51.6% |
| **Total** | **7,638,365** | **100%** |

**Observation**: Near-even split between S2 and S3. Slightly more S3 matches.

## Country Consistency (Verified)
All matched S2/S3 records share the **same country** as their S1 anchor entity. Cross-country matches do not exist in ground truth.

## Match Structure: One-to-Many vs Many-to-Many

### S1 → {S2, S3} (Primary Task)
- Each S1 maps to a set of S2/S3 records
- This is the **only labeled relationship**

### Implied S2/S3 Co-reference
- If S1-a matches {S2-x, S2-y}, then S2-x and S2-y co-refer
- If S1-a matches {S2-x, S3-y}, then S2-x and S3-y co-refer
- **Not directly labeled** — inferred from shared S1

### S2/S3 Internal Duplicates
- Multiple S2 records can match the same S1 (many-to-one from S2 perspective)
- Multiple S3 records can match the same S1
- An S2 record matches **exactly one S1** (verified: unique matched IDs = total match refs)

## Implications for Training/Validation

### Critical Issue: No Singleton Examples in Training
- **Train**: 0% singletons
- **Test**: Unknown % singletons (must be >0 per problem statement)
- **Consequence**: Cannot train a classifier on "match vs no-match" using only train GT
- **Solution needed**: 
  1. Create synthetic singletons by sampling non-matching pairs
  2. Hold out validation split with artificial singletons
  3. Calibrate threshold on validation to optimize F_0.5 including singletons

### Validation Strategy
1. **Split train S1 entities** (e.g., 80/20 or 90/10 by entity_id hash)
2. **For validation S1 entities**: Treat their GT matches as positive, sample non-matching S2/S3 as negative
3. **Add synthetic singletons**: Sample S1 entities and label as "no match" (or use threshold tuning)
4. **Optimize F_0.5 macro** on validation including singleton predictions

### Pairwise vs Set-wise Modeling
- **Pairwise**: Score each (S1, S2) and (S1, S3) pair independently → threshold → set prediction
- **Set-wise**: Predict match set directly (more complex, not needed for baseline)
- **Recommendation**: Pairwise classification with per-S1 threshold optimization

## Ground Truth as Supervision Signal

### Positive Pairs
- Every (S1, matched_S2) and (S1, matched_S3) from GT = **positive**
- Count: 7,638,365 positive pairs

### Negative Pairs (Implicit)
- All other (S1, S2) and (S1, S3) combinations = **negative**
- Count: ~2.2M × (5.0M + 5.3M) - 7.6M ≈ **22.7 trillion** potential negatives
- **Extreme class imbalance** — blocking/candidate generation essential

### Candidate Generation Ceiling
- Perfect blocking recall = 100% of 7.6M positives captured
- Practical blocking: Target >99% recall with manageable candidate set size
- Reduction ratio target: ~1:1000 or better (candidates per S1 ~ few thousand)

## Open Questions
- [ ] Exact match count histogram (1, 2, 3, ... 11)
- [ ] % S1 matching only S2, only S3, or both
- [ ] Correlation between match count and country
- [ ] Correlation between match count and name/address similarity
- [ ] Test set singleton rate estimation