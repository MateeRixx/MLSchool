# Current State

## CURRENT PHASE
**Phase 1: Data Understanding & Project Scaffolding** ✅ COMPLETE

---

## COMPLETED WORK

### Project Infrastructure
- ✅ Created `project_context/` directory with 15 markdown files
- ✅ Established data immutability policy (never modify raw TSVs)
- ✅ Established tab-separated format enforcement
- ✅ Created profiling script (`profile_data.py`) using Polars streaming

### Data Profiling (All Train + Test Files)
- ✅ **Row counts, schemas, null %, unique counts, duplicates** for all 7 TSV files
- ✅ **Ground truth analysis**: Structure, cardinality, source distribution, completeness
- ✅ **Cross-source overlap**: Verified zero ID overlap between any source pair
- ✅ **Country distribution**: Train (US/India), Test (US/India/France) — France confirmed as unseen
- ✅ **Data quality**: Null rates, duplicate rates, noise pattern identification
- ✅ **Validation rules**: Parsed `utils/validate_submission.py` and challenge specs
- ✅ **Evaluation metric**: Macro F_0.5 with singleton handling documented

### Documentation Created
- ✅ `00_project_overview.md` — Executive summary
- ✅ `01_problem_definition.md` — Entity, sources, GT, evaluation, challenges
- ✅ `02_dataset_inventory.md` — Complete file stats, schema, country distributions
- ✅ `03_schema_analysis.md` — Column definitions, noise patterns, train→test evolution
- ✅ `04_entity_model.md` — Conceptual model, cardinality, matching formalization
- ✅ `05_data_quality.md` — Nulls, duplicates, noise taxonomy, country patterns
- ✅ `06_ground_truth.md` — GT structure, match stats, validation implications
- ✅ `07_pipeline.md` — 5-stage architecture with blocking, features, model, output
- ✅ `08_evaluation.md` — F_0.5 formula, macro averaging, validation protocol, leaderboard
- ✅ `09_error_taxonomy.md` — FP/FN/singleton/structural error classification
- ✅ `10_experiments.md` — Experiment template, planned baseline experiments
- ✅ `11_decisions.md` — 12 key decisions with rationale and alternatives
- ✅ `12_known_issues.md` — 21 tracked issues across data/modeling/infra/compliance
- ✅ `13_current_state.md` — This file
- ✅ `14_next_steps.md` — Phase 2 plan (to be finalized)

### Key Findings Summary
1. **Entity**: Real-world businesses across 3 noisy sources
2. **Task**: Many-to-many S1→{S2,S3} matching, macro F_0.5
3. **Train GT**: 2.2M S1 entities, **0% singletons**, 3.67 avg matches/S1
4. **Test**: 1.7M S1 entities, **France (15%) unseen in train**
5. **Blocking**: Country exact match (100% GT recall, 3-7× reduction)
6. **Critical Gap**: No singleton training examples → requires synthetic validation
7. **Critical Gap**: France unseen → open-set country design mandatory

---

## IMPORTANT FINDINGS

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Zero singletons in train GT | 0/2.2M empty `matched_entity_ids` | Must create synthetic singletons for threshold calibration |
| France only in test | 259K S1, 703K S2, 732K S3 test entities with country=France | Country-agnostic features; no France-specific training |
| Country perfect blocking key | 0 cross-country matches in 7.6M GT pairs | Mandatory first-level block; 3-7× reduction |
| Address nulls in S2/S3 only | ~3% train, ~2.7% test; S1 has 0% | Name-only fallback needed |
| Name duplicates within source | 12-30% duplicate names | Name alone insufficient; need address combo |
| All GT references valid | 7.6M/7.6M matched IDs exist in sources | Clean supervision signal |
| Macro F_0.5 includes singletons | Spec: "Singletons included in average" | Singleton prediction quality directly impacts score |

---

## KNOWN ISSUES (Top Priority)

| ID | Issue | Severity | Status |
|----|-------|----------|--------|
| DI-001 | Zero singletons in training GT | Critical | 🔴 Open |
| DI-002 | France unseen in training | Critical | 🔴 Open |
| MR-001 | Threshold calibration without singletons | Critical | 🔴 Open |
| MR-002 | France generalization failure | Critical | 🔴 Open |
| MR-003 | Blocking recall < 100% risk | High | 🟡 Planned |
| DI-003 | Address nulls in S2/S3 | Medium | 🟡 Planned |
| IR-001 | Memory management for 5M+ rows | High | 🟡 Planned |

*Full list in `12_known_issues.md`*

---

## DECISIONS (Key Enforced Policies)

| ID | Decision | Enforced |
|----|----------|----------|
| D-001 | Data immutability | ✅ |
| D-002 | Explicit tab separator | ✅ |
| D-003 | Open-set country handling | ✅ |
| D-004 | Country as primary blocking key | ✅ |
| D-005 | Pairwise classification baseline | ✅ |
| D-006 | S1-hash validation split | ✅ |
| D-007 | Synthetic singletons for validation | ✅ |
| D-008 | Macro F_0.5 optimization target | ✅ |
| D-009 | CPU-only baseline | ✅ |
| D-010 | Pre-submission validation | ✅ |
| D-011 | Persistent project context | ✅ |
| D-012 | No external data lookup | ✅ |

*Full list in `11_decisions.md`*

---

## ARTIFACTS PRODUCED

| Artifact | Location | Description |
|----------|----------|-------------|
| `data_profiles.json` | `project_context/` | Complete profiling stats (JSON) |
| `profile_data.py` | `student_resource/` | Re-runnable profiling script |
| 15 markdown files | `project_context/` | Complete Phase 1 documentation |

---

## ENVIRONMENT
- **Python**: 3.13
- **Key packages**: polars (installed), pandas, numpy, scikit-learn, xgboost (to install)
- **OS**: Windows (PowerShell)
- **Working dir**: `C:\D-Drive\MLSchool\amazon-ml-2026\student_resource`