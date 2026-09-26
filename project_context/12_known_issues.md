# Known Issues & Risks

## Data Issues

| ID | Issue | Severity | Impact | Mitigation | Status |
|----|-------|----------|--------|------------|--------|
| DI-001 | **Zero singletons in training ground truth** | Critical | Model never sees "no match" examples; threshold calibration impossible without synthetic data | Create synthetic singletons in validation; calibrate threshold on held-out set | 🔴 Open |
| DI-002 | **France only in test set (unseen country)** | Critical | Name/address patterns may differ; no training signal for France-specific features | Open-set country design; country-agnostic features; validate France separately if possible | 🔴 Open |
| DI-003 | **Address nulls in S2/S3 (~3%)** | Medium | Name-only matching required for these records; lower confidence | Name-only blocking fallback; address_null feature for model | 🟡 Planned |
| DI-004 | **High name duplicate rate within source (12-30%)** | Medium | Name alone insufficient for disambiguation; need address/cross-field features | Combine name + address + country in features; use address tokens for blocking | 🟡 Planned |
| DI-005 | **Extreme class imbalance (~1:3000 positive:negative)** | High | Naive sampling wastes compute; hard negatives dominate | Smart negative sampling (hard negatives from blocking); blocking recall critical | 🟡 Planned |
| DI-006 | **Devanagari/Latin script mix in India records** | Medium | Cross-script matching needed; standard tokenizers fail | Script detection; separate tokenization; transliteration features | 🟡 Planned |
| DI-007 | **Test set smaller than train (1.7M vs 2.2M S1)** | Low | Less test data; validation split must be representative | Stratified split by country; ensure France represented in val if possible | 🟢 Monitor |

## Modeling Risks

| ID | Risk | Severity | Impact | Mitigation | Status |
|----|------|----------|--------|------------|--------|
| MR-001 | **Threshold calibration without true singletons** | Critical | Poor singleton predictions → F_0.5 collapse | Synthetic singletons in validation; per-entity threshold; probability calibration | 🔴 Open |
| MR-002 | **France generalization failure** | Critical | 15% of test entities in France; zero training data | Country-agnostic features; adversarial validation; conservative France threshold | 🔴 Open |
| MR-003 | **Blocking recall < 100%** | High | Missed true pairs unrecoverable; hard ceiling on recall | Multiple blocking strategies union; MinHash LSH; monitor recall on validation | 🟡 Planned |
| MR-004 | **Candidate set too large for pairwise scoring** | High | Inference time/memory explosion | Aggressive blocking; top-K per S1; efficient feature extraction (Polars/DuckDB) | 🟡 Planned |
| MR-005 | **Overfitting to US/India patterns** | Medium | Poor France performance | Regularization; feature selection; cross-validation by country | 🟡 Planned |
| MR-006 | **Legal suffix over-normalization causes false merges** | Medium | "Corp" ≈ "Inc" merges distinct entities | Preserve suffix as feature; don't strip completely; suffix match feature | 🟡 Planned |
| MR-007 | **Macro F_0.5 optimization instability** | Medium | Per-entity averaging sensitive to singleton predictions | Large validation set; bootstrap confidence intervals; monitor per-country | 🟡 Planned |

## Infrastructure & Operational Risks

| ID | Risk | Severity | Impact | Mitigation | Status |
|----|------|----------|--------|------------|--------|
| IR-001 | **Memory explosion with 5M+ row DataFrames** | High | OOM crashes; slow processing | Polars streaming / DuckDB out-of-core; chunked processing; Parquet intermediates | 🟡 Planned |
| IR-002 | **Inference time > 4 hours on CPU** | Medium | Missed iteration cycles; submission deadline risk | Profile bottlenecks; vectorize; parallelize by country; candidate pruning | 🟡 Planned |
| IR-003 | **Submission format rejection** | High | Wasted submission slot | Mandatory `validate_submission.py` pre-check; CI-style validation in pipeline | 🟢 Enforced |
| IR-004 | **Dependency version drift** | Medium | Non-reproducible results | Pinned `requirements.txt`; lock files; document versions in `project_context` | 🟢 Planned |
| IR-005 | **LLM context loss / session reset** | High | Loss of all analysis and decisions | `project_context/` as persistent memory; all findings written to markdown | 🟢 Enforced |

## Compliance Risks

| ID | Risk | Severity | Impact | Mitigation | Status |
|----|------|----------|--------|------------|--------|
| CR-001 | **Accidental external data usage** | Critical | Disqualification | Code audit; no network calls in pipeline; dependencies vetted | 🟢 Enforced |
| CR-002 | **Model license violation (>8B params or non-MIT/Apache)** | Critical | Disqualification | Use only open-source models <8B params; document license in submission | 🟢 Enforced |
| CR-003 | **Missing candidate_pairs.tsv in final zip** | High | Submission incomplete | Pipeline always generates both files; checklist in README | 🟢 Planned |

## Tracking & Resolution

### Issue Resolution Template
```markdown
## RESOLVED: DI-XXX / MR-XXX / IR-XXX / CR-XXX

**Date Resolved**: 2026-XX-XX
**Resolution**: [What was done]
**Verification**: [How it was tested]
**Impact on Metrics**: [Before/after if applicable]
**Related Experiments**: [EXP-XXX]
```

### Priority Legend
- 🔴 **Critical** — Blocks progress or causes failure; must resolve before Phase 2
- 🟡 **High** — Significant impact; resolve early in Phase 2
- 🟢 **Medium/Low** — Monitor; resolve before final submission

---

## Issue Status Summary (as of Phase 1 Complete)

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Data | 2 | 1 | 3 | 1 | 7 |
| Modeling | 2 | 1 | 5 | 0 | 8 |
| Infrastructure | 0 | 1 | 1 | 2 | 4 |
| Compliance | 0 | 0 | 0 | 2 | 2 |
| **Total** | **4** | **3** | **9** | **5** | **21** |

**Top 3 Priorities for Phase 2 Start**:
1. DI-001 / MR-001: Synthetic singleton validation strategy
2. DI-002 / MR-002: France generalization approach
3. MR-003: Blocking recall validation framework