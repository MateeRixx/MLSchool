# Evaluation Specification

## Metric: Macro-Averaged F_0.5

### Formula
```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Where for each S1 entity:
- **Precision** = TP / (TP + FP) = |Predicted ∩ True| / |Predicted|
- **Recall** = TP / (TP + FN) = |Predicted ∩ True| / |True|

### Macro-Averaging
1. Compute F_0.5 **per S1 entity** (1.7M test entities)
2. Average across **all** S1 entities
3. **Singletons included**: 
   - True matches = ∅, Predicted = ∅ → Precision=1, Recall=1, F_0.5=1.0 ✅
   - True matches = ∅, Predicted ≠ ∅ → Precision=0, Recall=1, F_0.5=0.0 ❌
   - True matches ≠ ∅, Predicted = ∅ → Precision=1, Recall=0, F_0.5=0.0 ❌

### Why Precision-Heavy (β=0.5)?
- False merge (matching two different businesses) → Precision penalty
- Missed match (failing to link same business) → Recall penalty
- **Business impact**: Merging distinct entities causes data corruption, legal issues, customer complaints
- **Weighting**: Precision weighted 2× over recall (β=0.5 → precision weight = 1/(1+β²) = 0.8)

## Scoring Examples

| True Matches | Predicted | Precision | Recall | F_0.5 |
|--------------|-----------|-----------|--------|-------|
| {S2-a, S3-b} | {S2-a, S3-b} | 1.0 | 1.0 | **1.000** |
| {S2-a, S3-b} | {S2-a} | 1.0 | 0.5 | **0.667** |
| {S2-a, S3-b} | {S2-a, S2-c} | 0.5 | 0.5 | **0.500** |
| {S2-a, S3-b} | {S2-a, S3-b, S2-c} | 0.667 | 1.0 | **0.714** |
| ∅ (singleton) | ∅ | 1.0 | 1.0 | **1.000** |
| ∅ (singleton) | {S2-a} | 0.0 | 1.0 | **0.000** |
| {S2-a} | ∅ | 1.0 | 0.0 | **0.000** |

## Validation Protocol

### Internal Validation (Phase 1-2)
1. **Split train S1 entities** by hash (e.g., 80/20 or 90/10)
   - Use `hash(entity_id) % 10` for deterministic split
   - Ensures no data leakage across sources
2. **Train on 80%**, validate on 20%
3. **Validation positives**: GT pairs for validation S1 entities
4. **Validation negatives**: Sampled non-matching pairs from same country
5. **Synthetic singletons**: Add validation S1 entities with "no match" label (or threshold tuning)

### Metrics to Track
| Metric | Target | Purpose |
|--------|--------|---------|
| Blocking Recall | >99% | Upper bound on final recall |
| Candidate Reduction Ratio | <1:1000 | Computational feasibility |
| Validation Macro F_0.5 | Maximize | Primary optimization target |
| Validation Precision | Monitor | False merge control |
| Validation Recall | Monitor | Missed match control |
| Predicted Singleton Rate | Match expected | Calibration check |

## Submission Validation

### Pre-Submission Checklist (via `validate_submission.py`)
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Rules Enforced
1. ✅ File exists and is readable UTF-8
2. ✅ Tab-separated (not CSV) — header check
3. ✅ Correct headers: `source1_entity_id\tmatched_entity_ids` / `source1_entity_id\tcandidate_entity_ids`
4. ✅ Every test S1 entity has exactly one row
5. ✅ No duplicate S1 entity rows
6. ✅ No duplicate IDs within any ID list
6. ✅ Only S2-/S3- prefixes in ID lists (no S1-)
7. ⚠️ (Warning) Matched IDs should exist in test S2/S3 files (`--check-ids`)
8. ⚠️ (Warning) Matched IDs ⊆ Candidate IDs (pipeline consistency)

### Exit Codes
- **0 (PASS)**: Safe to submit
- **1 (FAIL)**: Fix listed issues

## Leaderboard Evaluation

### Public Leaderboard
- Subset of test set (hidden split)
- Real-time feedback during challenge
- Based on `matching_results.tsv` only

### Private Leaderboard
- Remaining test set portion
- Revealed after challenge ends
- **Final rankings based on this**

### Final Submission Package
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv    # Same as leaderboard upload
│   └── candidate_pairs.tsv     # Blocking audit
├── code/
│   └── business_entity_resolution/
│       ├── src/                # All source code
│       ├── README.md           # Reproduction instructions
│       └── requirements.txt    # Pinned dependencies
└── Documentation_template.md   # Filled methodology doc
```

## F_0.5 Optimization Strategy

### Threshold Tuning
1. **Generate validation predictions** with scores/probabilities
2. **Sweep threshold** from 0.0 to 1.0 (or probability space)
3. **Compute macro F_0.5** at each threshold
4. **Select threshold** maximizing validation macro F_0.5
5. **Verify on held-out test** (if available) or trust validation

### Singleton Calibration
Since train has 0% singletons but test has >0%:
1. **Estimate test singleton rate** from data patterns (e.g., S1 entities with no plausible candidates)
2. **Adjust threshold** to achieve target singleton prediction rate
3. **Or**: Use validation set with injected synthetic singletons

### Per-Country Thresholds?
- Consider different thresholds for US, India, France
- France has no training data — may need conservative threshold
- Validate per-country F_0.5 on validation split

## Error Analysis Framework

### False Positives (Wrong Merges) — High Cost
- Different businesses predicted as same
- Analyze: name similarity high but address different? Common words? Legal suffix confusion?

### False Negatives (Missed Matches) — Lower Cost  
- Same business not linked
- Analyze: name very different? Address only match? Transliteration? Null address?

### Singleton Errors
- False positive singleton: Predicted match for true singleton
- False negative singleton: Predicted empty for true match

## Baseline Target
- **Blocking recall**: >99.5%
- **Validation macro F_0.5**: >0.60 (baseline), >0.75 (strong), >0.85 (excellent)
- **Inference time**: <4 hours on CPU for full test set