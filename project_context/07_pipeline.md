# Pipeline Architecture

## High-Level Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw Data   │────▶│  Preprocessing│────▶│  Blocking /     │────▶│  Pairwise        │────▶│  Thresholding  │
│  (TSV files)│     │  & Normalization│   │  Candidate Gen  │     │  Feature Extract │     │  & Prediction  │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────────┘     └─────────────────┘
                                                                                                  │
                                                                                                  ▼
                                                                                          ┌─────────────────┐
                                                                                          │  Output         │
                                                                                          │  matching_      │
                                                                                          │  results.tsv    │
                                                                                          └─────────────────┘
```

## Stage 1: Preprocessing & Normalization

### Input
- `train_source1/2/3.tsv`, `test_source1/2/3.tsv`, `train_ground_truth.tsv`

### Operations
1. **Read TSV with explicit tab separator** (critical — commas in addresses)
2. **Country validation**: Ensure open-set handling (no hardcoded country list)
3. **Text normalization** (applied to name + address):
   - Lowercase
   - Unicode normalization (NFC)
   - Remove extra whitespace
   - Standardize punctuation (`&`→`and`, `.`→``, `,`→``)
   - Expand common abbreviations (configurable dictionary)
   - Legal suffix normalization: `corp`→`corporation`, `inc`→`incorporated`, `ltd`→`limited`, `pvt`→`private`, `llc`→`llc`
3. **Address token extraction** (for blocking/features):
   - Numbers (street numbers, PIN/ZIP)
   - Street type tokens (rd, st, ave, blvd, road, street, avenue)
   - City/state tokens
   - PIN/ZIP codes (5-6 digits for US/India)
4. **Script detection** (India): Flag Devanagari vs Latin script records

### Output
- Normalized Parquet/CSV files per source per split
- Normalized columns: `entity_id`, `name_norm`, `addr_norm`, `country`, `name_tokens`, `addr_tokens`, `script`

## Stage 2: Blocking / Candidate Generation

### Goal
Reduce ~22 trillion possible pairs to manageable candidate set (~few billion max) while maintaining >99% recall on true matches.

### Blocking Keys (Progressive Strategy)

#### Level 1: Country Exact Match (Mandatory)
- **Key**: `country`
- **Reduction**: 3× (train) to ~6.7× (test with France)
- **Recall**: 100% (verified — no cross-country matches in GT)

#### Level 2: Name Token Overlap (Primary)
- **Key**: TF-IDF weighted tokens / MinHash LSH / q-grams (3-4 grams)
- **Threshold**: Jaccard ≥ 0.3 or top-K per S1
- **Target**: High recall, moderate candidate set

#### Level 3: Address Token Overlap (Secondary)
- **Key**: PIN/ZIP exact match + city token overlap + street number match
- **Use case**: Boost recall for name-ambiguous cases
- **Null handling**: Skip for records with null address

#### Level 4: Composite Keys (Ensemble)
- Union of multiple blocking strategies
- **Key combinations**: 
  - `country + name_first_token + name_last_token`
  - `country + PIN_prefix (first 3 digits)`
  - `country + phonetic_name (Soundex/Metaphone)`

### Candidate Output Format
```
source1_entity_id\tcandidate_entity_ids (comma-separated S2/S3 IDs)
```
- One row per test S1 entity
- Empty if no candidates found

## Stage 3: Pairwise Feature Extraction

### For each (S1, candidate) pair:

#### Name Features
| Feature | Description |
|---------|-------------|
| `name_jaccard` | Jaccard on word tokens (normalized) |
| `name_levenshtein` | Normalized Levenshtein distance |
| `name_levenshtein_ratio` | 1 - levenshtein/max_len |
| `name_token_overlap` | |tokens_S1 ∩ tokens_cand| / |tokens_S1 ∪ tokens_cand| |
| `name_prefix_match` | First token exact match (binary) |
| `name_suffix_match` | Last token exact match (binary) |
| `name_tfidf_cosine` | TF-IDF cosine similarity |
| `name_phonetic_match` | Soundex/Metaphone exact match (binary) |

#### Address Features
| Feature | Description |
|---------|-------------|
| `addr_jaccard` | Jaccard on address tokens |
| `addr_token_overlap` | Token overlap ratio |
| `street_num_match` | Exact street number match (binary) |
| `pin_zip_match` | Exact PIN/ZIP match (binary) |
| `pin_prefix_match` | First 3 digits of PIN/ZIP match (binary) |
| `city_match` | City token exact match (binary) |
| `state_match` | State token exact match (binary) |
| `addr_null` | Candidate address is null (binary) |

#### Cross/Categorical Features
| Feature | Description |
|---------|-------------|
| `country_match` | Always 1 (blocked on country) |
| `source_pair` | Categorical: S1-S2 vs S1-S3 |
| `name_len_ratio` | len(name_S1) / len(name_cand) |
| `addr_len_ratio` | len(addr_S1) / len(addr_cand) |
| `script_match` | Same script (Devanagari/Latin) for India |

#### Aggregated Features (Per S1 Entity)
- `num_candidates` — total candidates for this S1
- `candidate_source_ratio` — S2 vs S3 candidate proportion

## Stage 4: Matching Model

### Model Type (Baseline)
- **Algorithm**: XGBoost (CPU) or Logistic Regression
- **Task**: Binary classification (match / no-match) per pair
- **Training data**: 
  - Positives: All GT pairs (7.6M)
  - Negatives: Sampled from non-matching pairs within candidates (1:10 to 1:100 ratio)

### Threshold Selection
- **Per-S1 threshold**: Not global — optimize per entity or use calibrated probability
- **F_0.5 optimization**: Sweep threshold on validation to maximize macro F_0.5
- **Singleton handling**: 
  - Option A: Global threshold + "no match" if max_score < threshold
  - Option B: Learn "no match" class with synthetic singletons
  - Option C: Calibrate probabilities, predict empty if all p < 0.5

### Prediction Aggregation
For each S1 entity:
1. Score all candidates
2. Apply threshold → binary decisions
3. Collect matched IDs → comma-separated list
4. If empty and singleton logic triggers → output empty

## Stage 5: Output Generation

### matching_results.tsv
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```
- One row per test S1 entity (1.7M rows)
- Tab-separated, no quoting
- Empty matched_entity_ids for predicted singletons

### candidate_pairs.tsv
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```
- Superset of matching_results (every matched ID must appear in candidates)
- Used for blocking quality audit

## Computational Considerations

### Memory Management
- **Streaming/chunked processing**: Never load full 5M+ row DataFrames in memory
- **Polars streaming** or **DuckDB** for out-of-core operations
- **Candidate generation**: Write to disk incrementally (per S1 batch)

### Parallelization
- **Country-level parallelism**: Process US, India, France independently
- **S1-batch parallelism**: Split S1 entities across workers
- **Pairwise scoring**: Vectorized batch inference

### Estimated Scale
| Stage | Input Scale | Output Scale | Compute |
|-------|-------------|--------------|---------|
| Blocking | 1.7M S1 × 10M S2/S3 | ~1-5B candidates | High (similarity joins) |
| Features | ~1-5B pairs | ~1-5B × 20 features | Medium (vectorized) |
| Model | ~1-5B pairs | ~1-5B scores | Medium (batch inference) |
| Thresholding | 1.7M S1 | 1.7M predictions | Low |

## Validation & Monitoring
- **Blocking recall**: % of GT pairs captured in candidates (target >99%)
- **Candidate set size**: Mean/max candidates per S1
- **Validation F_0.5**: Macro F_0.5 on held-out train split
- **Singleton rate**: Predicted vs expected (calibrate)
- **Submission validation**: Run `validate_submission.py` before submit

## Phase 1 Baseline Scope
1. ✅ Country blocking only (exact match)
2. ✅ Name Jaccard + Levenshtein features
3. ✅ Logistic Regression on pairwise features
4. ✅ Global threshold + singleton heuristic
5. ✅ Validation split from train (20% S1 entities)
6. ✅ Output both TSV files + validation