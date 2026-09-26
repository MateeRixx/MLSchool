# Entity Model

## Conceptual Model

```
Real-World Business Entity
    │
    ├── Source 1 Record (S1) — deduplicated, canonical representation
    │     entity_id: S1-xxxxx
    │     business_name: "canonical-ish"
    │     business_address: "canonical-ish"
    │     country: "US" | "India" | "France"
    │
    ├── Source 2 Record(s) (S2) — noisy fragments
    │     entity_id: S2-xxxxx (multiple per real entity)
    │     business_name: noisy variants
    │     business_address: noisy/partial variants (nullable)
    │     country: same as S1 (verified via GT)
    │
    └── Source 3 Record(s) (S3) — noisy fragments
          entity_id: S3-xxxxx (multiple per real entity)
          business_name: noisy variants
          business_address: noisy/partial variants (nullable)
          country: same as S1 (verified via GT)
```

## Key Properties

### 1. One-to-Many (S1 → S2/S3)
- Each S1 entity maps to **0 to N** S2 records and **0 to M** S3 records
- Training data: **Every S1 has ≥1 match** (N+M ≥ 1)
- Test data: **Some S1 will have 0 matches** (singletons) — must predict empty

### 2. No S2 ↔ S3 Direct Links
- Ground truth only links S1 → {S2, S3}
- S2 and S3 records are linked **only through shared S1 entity**
- Two S2 records matching same S1 are "co-referent" but not directly labeled

### 3. Country Consistency
- Verified: Matched S2/S3 records share the same country as their S1 anchor
- Country is a **reliable blocking key** (no cross-country matches in GT)

### 4. No Within-Source Duplicates
- S1 is deduplicated by definition
- S2/S3 entity_ids are unique within source
- But: **Business names/addresses can duplicate within source** (different entities, same name)

## Match Cardinality Statistics (Train)

| Metric | Value |
|--------|-------|
| S1 entities | 2,206,821 |
| Total S2 matches | 3,693,619 |
| Total S3 matches | 3,944,746 |
| Total match edges | 7,638,365 |
| Avg matches per S1 | 3.67 |
| Max matches per S1 | 11 |
| S1 with 1 match | ~X% |
| S1 with 2-3 matches | ~Y% |
| S1 with 4+ matches | ~Z% |

*(Exact distribution to be computed in detailed analysis)*

## Entity Resolution Task Formalization

**Input**: 
- Set of S1 entities: `E1 = {e1_1, e1_2, ..., e1_n}`
- Set of S2 entities: `E2 = {e2_1, e2_2, ..., e2_m}`
- Set of S3 entities: `E3 = {e3_1, e3_2, ..., e3_k}`

**Output** (for each e1 ∈ E1):
- `M(e1) ⊆ (E2 ∪ E3)` — predicted matching entities
- For test: must output for ALL 1.7M test S1 entities

**Ground Truth** (train only):
- `GT(e1) ⊆ (E2 ∪ E3)` — true matches

**Evaluation**: Macro F_0.5 over all e1 ∈ E1_test

## Implications for Modeling

### Blocking Strategy
1. **Country-first**: Exact match on country (reduces search space 3-7×)
2. **Name token blocking**: q-grams, TF-IDF, or phonetic encoding on business_name
3. **Address token blocking**: PIN/ZIP codes, city names, street numbers (where present)
4. **Hybrid**: Union of multiple blocking keys for recall ceiling

### Matching Model
- **Pairwise classification**: Score (e1, e2) and (e1, e3) pairs independently
- **Threshold per S1**: Convert scores to binary decisions per S1 entity
- **Singleton handling**: Learn "no match" threshold or explicit null class

### Clustering (Not Required)
- Output format is **star-shaped** from S1 perspective
- No need to cluster S2/S3 records among themselves
- But: Consistency check — if e1 matches e2_a and e2_b, they co-refer

## Open Questions for Phase 2

1. **Match count distribution**: What % of S1 have 1 vs 2-3 vs 4+ matches?
2. **S2 vs S3 balance per S1**: Do most S1 match both sources or just one?
3. **Cross-source consistency**: If S1 matches S2-a and S3-b, how similar are S2-a and S3-b?
4. **Singleton prevalence in test**: Estimate from data patterns?
5. **France-specific patterns**: Do French addresses/names follow different structure?