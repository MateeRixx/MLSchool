# Next Steps: Phase 2 — Baseline Pipeline

## Phase 2 Objective
Build a working, validated, CPU-only baseline pipeline that:
1. Generates candidates via country-blocking + name token blocking
2. Extracts pairwise features (name + address)
3. Trains a binary classifier (Logistic Regression → XGBoost)
4. Optimizes threshold for macro F_0.5 on validation split
5. Produces valid `matching_results.tsv` and `candidate_pairs.tsv` for test set
6. Passes `validate_submission.py` pre-check

## Immediate Actions (Week 1)

### 2.1 Environment & Utilities
- [ ] Install baseline dependencies: `xgboost`, `scikit-learn`, `rapidfuzz` (for Levenshtein), `jellyfish` (phonetic)
- [ ] Create `requirements.txt` with pinned versions
- [ ] Set up project structure:
  ```
  student_resource/
  ├── src/
  │   ├── preprocessing.py
  │   ├── blocking.py
  │   ├── features.py
  │   ├── model.py
  │   ├── predict.py
  │   └── utils.py
  ├── experiments/
  │   └── run_baseline.py
  ├── output/              # matching_results.tsv, candidate_pairs.tsv
  ├── project_context/     # (existing)
  └── dataset/             # (immutable)
  ```

### 2.2 Validation Split Creation
- [ ] Deterministic split of train S1 entities: `hash(entity_id) % 10 < 8` → train, `>= 8` → val
- [ ] Save split indices to `project_context/train_val_split.json`
- [ ] Verify country distribution balance in both splits
- [ ] Create synthetic singleton labels for validation S1 entities (sample 10-20% as "no match")

### 2.3 Blocking Implementation (EXP-101)
- [ ] **Country exact match** (mandatory): Join S1 with S2/S3 on `country`
- [ ] **Name token blocking**: 
  - Normalize names (lowercase, remove punctuation, expand abbreviations)
  - Extract word tokens (split on whitespace)
  - Block on token overlap: Jaccard ≥ 0.3 OR top-50 candidates per S1 by token overlap
  - Use Polars streaming for memory efficiency
- [ ] **Blocking recall validation**: Compute % of GT pairs captured on validation split
- [ ] **Target**: >99% blocking recall, <5000 candidates/S1 (mean)

### 2.4 Feature Extraction (EXP-101/102)
- [ ] **Name features**: Jaccard, Levenshtein (rapidfuzz), token overlap, prefix/suffix match, TF-IDF cosine
- [ ] **Address features**: Token overlap, street number match, PIN/ZIP match (exact + prefix), city match, null flag
- [ ] **Meta features**: source_pair (S1-S2 vs S1-S3), name/addr length ratios, script match
- [ ] Output: Parquet file with (s1_id, cand_id, features..., label) for train/val

### 2.5 Model Training (EXP-101/103/104)
- [ ] **Negative sampling**: For each positive pair, sample 20-50 negatives from same S1's candidates
- [ ] **Logistic Regression baseline**: `sklearn.linear_model.LogisticRegression` (class_weight='balanced')
- [ ] **XGBoost upgrade**: `xgboost.XGBClassifier` (max_depth=6, n_estimators=200, scale_pos_weight)
- [ ] **Calibration**: `CalibratedClassifierCV` for probability outputs
- [ ] Save model artifacts to `models/baseline/`

### 2.6 Threshold Optimization (EXP-105/106)
- [ ] Sweep threshold 0.01→0.99 on validation set
- [ ] Compute **macro F_0.5** at each threshold
- [ ] Select threshold maximizing macro F_0.5
- [ ] **Per-country thresholds**: Separate sweep for US, India, France (France uses conservative default)
- [ ] **Singleton calibration**: Adjust to match expected singleton rate (estimate from candidate-less S1)

### 2.7 Test Inference & Output
- [ ] Run blocking on full test set (1.7M S1 × ~10M S2/S3)
- [ ] Extract features for all test candidates
- [ ] Batch inference (chunked to manage memory)
- [ ] Apply optimized thresholds → binary decisions
- [ ] Aggregate per S1 → comma-separated matched IDs
- [ ] Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- [ ] **Validate**: `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

## Success Criteria for Phase 2

| Metric | Minimum Target | Stretch Target |
|--------|----------------|----------------|
| Blocking Recall (val) | 99.0% | 99.9% |
| Candidates/S1 (mean) | <10,000 | <2,000 |
| Val Macro F_0.5 | >0.60 | >0.75 |
| Test submission validation | PASS | PASS |
| Inference time (full test) | <4 hours | <1 hour |

## Phase 3 Preview (Iterative Improvement)

Once baseline passes validation:
1. **Error analysis** on validation errors → `09_error_taxonomy.md`
2. **Advanced blocking**: MinHash LSH, phonetic, composite keys
3. **Better features**: TF-IDF, embeddings (Sentence-BERT), cross-encoders
4. **Model upgrades**: XGBoost tuning, neural pairwise, ensemble
5. **France-specific**: Adversarial validation, conservative threshold
6. **Singleton mastery**: Per-entity threshold, learned "no-match" score

## Phase 4 Preview (Final Submission)
1. Package `code/business_entity_resolution/` with `src/`, `README.md`, `requirements.txt`
2. Fill `Documentation_template.md`
3. Create submission zip
4. Final leaderboard submission

## Open Questions to Resolve in Phase 2

1. **Exact synthetic singleton strategy**: How many? How to sample negatives?
2. **France threshold**: Conservative (high precision) or calibrated from US/India?
3. **Blocking strategy for France**: Same as US/India or adjusted?
4. **Address parsing**: Regex vs library (usaddress, libpostal) — license check needed
5. **TF-IDF vocabulary**: Fit on train only? Include test? (Leakage risk)

## Dependencies to Install
```bash
pip install xgboost scikit-learn rapidfuzz jellyfish polars pyarrow
```

## Commands to Run (Once Implemented)
```bash
# Create validation split
python -m src.create_split

# Run baseline experiment
python -m experiments.run_baseline --exp EXP-101

# Validate submission
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Human Review Gate
**Phase 1 Complete. Awaiting human review of `project_context/` files before proceeding to Phase 2.**