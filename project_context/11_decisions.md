# Key Decisions Log

## Decision Format
| Decision ID | Date | Category | Decision | Rationale | Alternatives Considered | Status |
|-------------|------|----------|----------|-----------|------------------------|--------|

---

## D-001: Data Immutability
- **Date**: 2026-09-26
- **Category**: Data Policy
- **Decision**: Never modify `dataset/train/*.tsv` or `dataset/test/*.tsv`. All processing writes to `project_context/`, `output/`, or temporary directories.
- **Rationale**: Preserve original data for reproducibility; prevent accidental corruption; enable clean re-runs.
- **Alternatives**: In-place cleaning, symlinks
- **Status**: ✅ Enforced

## D-002: Tab-Separated Format Enforcement
- **Date**: 2026-09-26
- **Category**: Data Policy
- **Decision**: All TSV reads/writes must use explicit `sep="\t"` (pandas) or `separator="\t"` (Polars). Never rely on defaults.
- **Rationale**: Business addresses and ID lists contain commas; CSV parsing would corrupt data.
- **Alternatives**: None — this is a hard requirement from challenge specs.
- **Status**: ✅ Enforced

## D-003: Open-Set Country Handling
- **Date**: 2026-09-26
- **Category**: Modeling
- **Decision**: Treat `country` as open-set string label. No hardcoded country lists, no one-hot encoding, no country-specific model branches trained only on seen countries.
- **Rationale**: Test set contains France (unseen in train). Hardcoding {US, India} will fail on France entities.
- **Alternatives**: Train separate models per country, map unseen to "other"
- **Status**: ✅ Enforced

## D-004: Country as Primary Blocking Key
- **Date**: 2026-09-26
- **Category**: Pipeline Design
- **Decision**: Use exact country match as mandatory first-level blocking key.
- **Rationale**: 
  - Zero cross-country matches in ground truth (verified)
  - Reduces search space 3× (train) to ~6.7× (test)
  - Clean, noiseless field
  - France in test naturally handled (blocks within France)
- **Alternatives**: Soft country matching, no country blocking
- **Status**: ✅ Enforced

## D-005: Pairwise Classification Approach
- **Date**: 2026-09-26
- **Category**: Modeling
- **Decision**: Frame matching as pairwise binary classification (S1, candidate) → score, then threshold per S1 entity.
- **Rationale**: 
  - Natural fit for many-to-many matching
  - Scalable with candidate generation
  - Compatible with F_0.5 threshold optimization
  - Simpler than set-wise or clustering approaches
- **Alternatives**: Set-wise prediction, clustering, graph-based
- **Status**: ✅ Enforced for baseline

## D-006: Validation Split by S1 Entity Hash
- **Date**: 2026-09-26
- **Category**: Evaluation
- **Decision**: Split training S1 entities by `hash(entity_id) % 10` (e.g., 80/20). Never random shuffle rows.
- **Rationale**: 
  - Prevents leakage: same S1 entity in train and val
  - Deterministic, reproducible
  - Maintains country distribution
- **Alternatives**: Random split, temporal split (no timestamp)
- **Status**: ✅ Planned for Phase 2

## D-007: Synthetic Singletons for Validation
- **Date**: 2026-09-26
- **Category**: Evaluation
- **Decision**: Create synthetic singleton examples in validation by sampling S1 entities and labeling as "no match" (or sampling non-matching pairs), since train has 0% singletons.
- **Rationale**: Test will have singletons; model must learn to predict empty. Without singleton examples, threshold calibration is impossible.
- **Alternatives**: Use global threshold heuristic, ignore singletons in training
- **Status**: ✅ Planned for Phase 2

## D-008: Macro F_0.5 as Primary Optimization Target
- **Date**: 2026-09-26
- **Category**: Evaluation
- **Decision**: Optimize all thresholds and model selection for macro-averaged F_0.5 on validation split.
- **Rationale**: Matches challenge evaluation exactly. Per-entity averaging means each S1 entity contributes equally regardless of match count.
- **Alternatives**: Micro F_0.5, F1, accuracy, AUC
- **Status**: ✅ Enforced

## D-009: CPU-Only Baseline Constraint
- **Date**: 2026-09-26
- **Category**: Infrastructure
- **Decision**: Phase 2 baseline must run on CPU only (no GPU requirement). Use Polars/DuckDB for out-of-core, XGBoost/sklearn for CPU training.
- **Rationale**: Accessibility, reproducibility, cost. Final model can use GPU but baseline should not require it.
- **Alternatives**: GPU-accelerated baseline
- **Status**: ✅ Enforced

## D-010: Output Format Compliance
- **Date**: 2026-09-26
- **Category**: Submission
- **Decision**: Every pipeline run must produce `matching_results.tsv` and `candidate_pairs.tsv` validated by `utils/validate_submission.py` before any submission attempt.
- **Rationale**: Avoid wasted submissions on format errors. Validator catches all structural issues.
- **Alternatives**: Manual checking
- **Status**: ✅ Enforced

## D-011: Persistent Project Context
- **Date**: 2026-09-26
- **Category**: Project Management
- **Decision**: All findings, decisions, and state stored in `project_context/` markdown files. Chat context is ephemeral.
- **Rationale**: Survives LLM reset. Enables handoff. Single source of truth.
- **Alternatives**: Rely on chat history, external docs
- **Status**: ✅ Enforced

## D-012: No External Data Lookup
- **Date**: 2026-09-26
- **Category**: Compliance
- **Decision**: Strictly prohibit any external API, database, geocoding service, or business registry lookup. Use only provided training data.
- **Rationale**: Challenge rules — immediate disqualification for violation.
- **Alternatives**: None — hard constraint
- **Status**: ✅ Enforced

---

## Decision Categories
- **Data Policy**: How data is read, written, transformed
- **Modeling**: Algorithm, architecture, feature choices
- **Pipeline Design**: Stage decomposition, data flow
- **Evaluation**: Metrics, validation, optimization targets
- **Infrastructure**: Compute, dependencies, reproducibility
- **Submission**: Format, validation, packaging
- **Project Management**: Documentation, state, handoff
- **Compliance**: Challenge rules, legal, ethical