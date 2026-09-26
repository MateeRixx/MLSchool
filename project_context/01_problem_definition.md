# Problem Definition

## Real-World Entity
We are resolving **business entities** — real-world commercial organizations (companies, shops, services, corporations, etc.) that appear across multiple independent data sources. Each source captures partial, noisy fragments of the same business's identity information.

## Source Roles and Relationships

### Source 1 (S1) — Deduplicated Reference Source
- **Role**: The "golden" anchor set. Each record represents a unique, deduplicated business entity.
- **Count**: 2.2M (train), 1.7M (test)
- **Key property**: Every S1 entity in the test set MUST appear exactly once in the submission output.

### Source 2 (S2) — Noisy Source A
- **Role**: First auxiliary source containing business records that may match S1 entities.
- **Count**: 5.0M (train), 4.9M (test)
- **Relationship to S1**: Many-to-many. An S1 entity can match 0, 1, or many S2 records.

### Source 3 (S3) — Noisy Source B
- **Role**: Second auxiliary source containing business records that may match S1 entities.
- **Count**: 5.3M (train), 5.1M (test)
- **Relationship to S1**: Many-to-many. An S1 entity can match 0, 1, or many S3 records.

## Cross-Source Independence
- **No shared identifiers**: Sources share no common keys. Entity IDs are prefixed (S1-, S2-, S3-) and unique within each source.
- **No ID overlap**: Verified — zero entity_id overlap between any source pair in training data.
- **Ground truth bridges sources**: Only `train_ground_truth.tsv` links S1 entities to their matching S2/S3 records.

## Ground Truth Structure
- **File**: `train_ground_truth.tsv` (2.2M rows = one per S1 entity in train)
- **Columns**: 
  - `source1_entity_id`: S1 entity ID
  - `matched_entity_ids`: Comma-separated list of matching S2/S3 entity IDs (empty = no matches)
- **Critical finding**: **Zero singletons in training ground truth** — every S1 entity has ≥1 match.
- **Average matches per S1**: 3.67 (range: 1–11)
- **Source split**: ~48% S2 matches, ~52% S3 matches
- **Completeness**: All matched IDs verified to exist in source data (0 orphan references)

## Evaluation Metric: Macro F_0.5
```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```
- **Macro-averaged**: Compute per S1 entity, then average across ALL test S1 entities (1.7M)
- **Singletons matter**: Test set WILL contain S1 entities with no true matches (unlike train). Correctly predicting empty list = 1.0 score for that entity.
- **Precision-heavy**: False merges (matching distinct businesses) penalized 2× more than missed matches.

## Key Challenges
1. **Open-set country**: Test includes **France** (absent from training). Pipeline must generalize to unseen country labels.
2. **Noisy text fields**: Business names/addresses have abbreviations, typos, transliterations, format variations, missing components.
3. **Scale**: ~12M training records, ~11.7M test records — blocking/candidate generation is critical for computational feasibility.
4. **Many-to-many matching**: Not 1:1 linkage; must predict variable-length match lists.
5. **Singleton detection**: Must learn to predict "no match" despite zero singleton examples in training.

## Logical Next Steps for Baseline
1. **Blocking strategy**: Country-first blocking (exact match) + token-based name/address blocking within country
2. **Features**: String similarities (Jaccard, Levenshtein, TF-IDF cosine) on normalized name/address
3. **Model**: Lightweight classifier (XGBoost/Logistic Regression) on pairwise features
4. **Threshold tuning**: Optimize F_0.5 on validation split (hold out from train)
5. **Singleton handling**: Calibrate threshold or add "no-match" class; validate on held-out singletons