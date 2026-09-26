# Project Overview: Business Entity Resolution (ML Challenge 2026)

## Objective
Build an ML solution for cross-source business entity resolution: given business records from 3 independent data sources with noisy and inconsistent fields, determine which records across sources refer to the same real-world business entity.

## Source Roles
- **Source 1 (S1)**: Deduplicated reference source — the "anchor" entities we must find matches for
- **Source 2 (S2)**: Noisy source with potential matches to S1 entities
- **Source 3 (S3)**: Another noisy source with potential matches to S1 entities

## Task Type
**Many-to-many entity resolution**: Each S1 entity may match zero, one, or many records from S2 and/or S3.

## Data Scale
| Dataset | Source 1 | Source 2 | Source 3 |
|---------|----------|----------|----------|
| Train   | 2.2M     | 5.0M     | 5.3M     |
| Test    | 1.7M     | 4.9M     | 5.1M     |

## Output Format
Two TSV files in `output/`:
1. `matching_results.tsv` — Final matches (scored on leaderboard)
2. `candidate_pairs.tsv` — Blocking candidates (for audit, not scored)

## Evaluation Metric
**Macro-averaged F_0.5** (precision-weighted, β=0.5):
- Calculated per S1 entity, then averaged across ALL S1 entities
- Singletons included: correct empty prediction = 1.0, false match = 0.0
- Precision weighted 2× over recall (false merges penalized more than missed matches)

## Constraints
- No external data lookup (strictly prohibited)
- Model ≤ 8B parameters, MIT/Apache 2.0 license
- Test includes France (unseen in training) — open-set country handling required
- Submissions must be tab-separated (not CSV)

## Current Phase
**Phase 1: Data Understanding & Project Scaffolding** (this document)