# Experiment Tracking

## Experiment Template

| Field | Description |
|-------|-------------|
| **Exp ID** | Unique identifier (e.g., EXP-001) |
| **Date** | YYYY-MM-DD |
| **Phase** | 1 (Data Understanding) / 2 (Baseline) / 3 (Iteration) / 4 (Final) |
| **Objective** | What hypothesis is being tested? |
| **Config** | Key hyperparameters, blocking keys, features, model |
| **Train/Val Split** | How data was split (hash, random, temporal) |
| **Blocking Recall** | % GT pairs captured |
| **Candidates/S1** | Mean/max candidate set size |
| **Val Macro F_0.5** | Primary metric |
| **Val Precision** | Secondary |
| **Val Recall** | Secondary |
| **Singleton Rate** | Predicted vs actual (if known) |
| **Runtime** | Training + inference time |
| **Artifacts** | Model path, predictions path, logs |
| **Notes** | Observations, issues, next steps |

---

## Phase 1: Data Understanding Experiments

| Exp ID | Objective | Status | Key Findings |
|--------|-----------|--------|--------------|
| EXP-001 | Profile all train/test TSV files | ✅ Complete | See `data_profiles.json` and `02_dataset_inventory.md` |
| EXP-002 | Analyze ground truth structure | ✅ Complete | 0% singletons in train, 3.67 avg matches/S1, S2/S3 ~50/50 split |
| EXP-003 | Verify cross-source integrity | ✅ Complete | No ID overlap, all GT refs valid, country consistent |
| EXP-004 | Country distribution analysis | ✅ Complete | Train: US/India only. Test: +France (15%). Open-set confirmed. |

---

## Phase 2: Baseline Experiments (Planned)

| Exp ID | Objective | Config | Target |
|--------|-----------|--------|--------|
| EXP-101 | Country-only blocking + name Jaccard features + Logistic Regression | Blocking: country exact. Features: name_jaccard, name_levenshtein. Model: LR. Threshold: sweep. | F_0.5 > 0.60, Blocking recall > 99% |
| EXP-102 | Add address features (token overlap, PIN/ZIP) | + addr_jaccard, pin_match, city_match | F_0.5 > 0.65 |
| EXP-103 | Add TF-IDF cosine + phonetic features | + name_tfidf_cosine, name_phonetic_match | F_0.5 > 0.70 |
| EXP-104 | XGBoost instead of Logistic Regression | Model: XGBoost (max_depth=6, n_estimators=100) | F_0.5 > 0.72 |
| EXP-105 | Per-country threshold tuning | Separate threshold for US, India, France | F_0.5 > 0.73 |
| EXP-106 | Synthetic singleton calibration | Inject synthetic singletons in validation | Singleton F_0.5 > 0.80 |
| EXP-107 | MinHash LSH blocking (vs exact country) | Blocking: country + MinHash on name tokens | Recall > 99.5%, candidates/S1 < 5000 |

---

## Phase 3: Advanced Experiments (Future)

| Exp ID | Idea | Rationale |
|--------|------|-----------|
| EXP-201 | Siamese neural network (Sentence-BERT embeddings) | Better semantic name/address matching |
| EXP-202 | Graph-based ER (connected components on high-confidence pairs) | Leverage transitivity |
| EXP-203 | Active learning for hard negative mining | Improve precision on confusing pairs |
| EXP-204 | Country-adversarial training for France generalization | Domain adaptation for unseen country |
| EXP-205 | Ensemble of multiple blocking strategies | Maximize recall ceiling |
| EXP-206 | Learned threshold per S1 entity (meta-features) | Handle variable match counts |

---

## Experiment Log Format (for Phase 2+)

```markdown
## EXP-XXX: [Title]

**Date**: 2026-XX-XX
**Phase**: 2
**Objective**: [What are we testing?]

### Configuration
- **Blocking**: [keys, thresholds]
- **Features**: [list]
- **Model**: [type, hyperparameters]
- **Train/Val Split**: [method, seed]
- **Negative Sampling**: [ratio, strategy]
- **Threshold Method**: [global sweep / per-country / per-entity]

### Results
| Metric | Value |
|--------|-------|
| Blocking Recall | X.XX% |
| Candidates/S1 (mean/max) | X / X |
| Val Macro F_0.5 | X.XXX |
| Val Precision | X.XXX |
| Val Recall | X.XXX |
| Predicted Singleton Rate | X.X% |

### Error Analysis (Sample of 50 errors)
| Error Type | Count | Example |
|------------|-------|---------|
| FP: Name Collision | X | ... |
| FN: Address Variation | X | ... |

### Artifacts
- Model: `models/exp-xxx/model.pkl`
- Predictions: `predictions/exp-xxx/val_preds.parquet`
- Logs: `logs/exp-xxx/train.log`

### Notes
[Observations, issues, next experiment ideas]
```

---

## Reproducibility Requirements

For every experiment:
1. **Code versioned** (git commit hash)
2. **Data split deterministic** (document hash function/seed)
3. **Dependencies pinned** (requirements.txt or equivalent)
4. **Random seeds set** (numpy, python, xgboost, etc.)
5. **Artifacts saved** (model, predictions, metrics JSON)
6. **Config serialized** (YAML/JSON with all hyperparameters)