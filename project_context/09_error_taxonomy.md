# Error Taxonomy

## Classification of Entity Resolution Errors

### 1. False Positive (Wrong Merge) — **High Cost**
Predicting a match between two **different** real-world businesses.

| Subtype | Description | Example | Likely Cause |
|---------|-------------|---------|--------------|
| **Name Collision** | Distinct businesses share similar/common names | "Prime Money" (US) vs "Prime Money" (India) | Common words, generic names, insufficient address disambiguation |
| **Legal Suffix Confusion** | Same base name, different entity types | "ABC Corp" vs "ABC Inc" (different companies) | Over-normalization of legal suffixes |
| **Address Hallucination** | Null/missing address matched on name only | S1 has address, S2 address=null, names similar | Over-reliance on name when address missing |
| **Transliteration Error** | Different entities map to same Latin form | Two different Devanagari names → same transliteration | Lossy transliteration |
| **Franchise/Chain Confusion** | Same brand, different locations | "Starbucks #123" vs "Starbucks #456" | Name similarity without address discrimination |
| **Cross-Country Leakage** | Matches across countries | US entity matched to France entity | Blocking failure (should be impossible with country blocking) |

### 2. False Negative (Missed Match) — **Lower Cost**
Failing to predict a match between two records of the **same** real-world business.

| Subtype | Description | Example | Likely Cause |
|---------|-------------|---------|--------------|
| **Name Variation** | True match has highly dissimilar names | "Prabhav Business Center" vs "Prabhav Business Centre Pvt Ltd" | Abbreviation, legal suffix, DBA, typo beyond threshold |
| **Address Variation** | True match has highly dissimilar addresses | "1795 Westchester Drive" vs "Near SBI ATM, Westchester" | Landmark vs formal address, component reordering |
| **Null Address** | One record has null address | S1 has address, S2 address=null | Blocking/feature requires address |
| **Transliteration Gap** | Same entity, different scripts | Devanagari name vs English name | No cross-script matching |
| **Blocking Miss** | True pair never generated as candidate | Recall < 100% in blocking | Overly aggressive blocking keys |
| **Threshold Too High** | Score below threshold | Similar but not identical records | Global threshold not calibrated per entity |

### 3. Singleton Errors
| Subtype | Description | Cost |
|---------|-------------|------|
| **False Positive Singleton** | Predicted matches for true singleton (no true matches) | **High** — F_0.5 = 0.0 for that entity |
| **False Negative Singleton** | Predicted empty for entity with true matches | **Medium** — F_0.5 = 0.0 for that entity |

### 4. Structural Errors
| Error | Description | Detection |
|-------|-------------|-----------|
| **Duplicate Predictions** | Same S2/S3 ID listed multiple times for one S1 | Validator catches |
| **Self-Match** | S1-ID in matched_entity_ids | Validator catches |
| **Invalid ID** | Matched ID not in test set | Validator `--check-ids` |
| **Missing S1 Entity** | Test S1 entity not in output | Validator catches |
| **Extra S1 Entity** | Output contains S1 not in test set | Validator catches |
| **Matched ⊄ Candidates** | Final match not in candidate set | Validator warns |

## Error Attribution Framework

### For Each Error, Tag:
1. **Stage**: Blocking / Feature Extraction / Model / Thresholding / Post-processing
2. **Modality**: Name / Address / Country / Cross-field
3. **Geography**: US / India / France
4. **Frequency**: Count in validation error sample
5. **Fix Priority**: High/Medium/Low based on F_0.5 impact

### Analysis Workflow
1. **Sample errors** from validation set (stratified by error type)
2. **Root cause analysis** per error (manual inspection of 50-100 cases)
3. **Quantify** % of each error type in validation
4. **Prioritize fixes** by (frequency × F_0.5 impact)
5. **Implement** targeted fixes (blocking key, feature, threshold)
6. **Re-validate** and measure delta

## Common Error Patterns to Watch For

### India-Specific
- Devanagari ↔ Latin script mismatch
- Landmark addresses vs formal addresses
- PIN code format variations (6 digits, sometimes with spaces)
- "Private Limited" / "Pvt Ltd" / "Pvt. Ltd." normalization

### US-Specific
- State abbreviation vs full name (CA vs California)
- ZIP+4 vs 5-digit ZIP
- Suite/unit number variations (Ste, Suite, #, Unit, Apt)
- Directional prefixes (N, S, E, W, NE, NW)

### France-Specific (Test Only)
- "R." / "Rue" / "Route" variations
- "Av." / "Avenue" / "Boulevard" 
- Department codes in addresses (75, 69, 13, etc.)
- SARL / SAS / SA / SCI / EURL legal forms
- Accented characters (é, è, à, ç, ô)

### Cross-Cutting
- Legal suffix normalization (Corp/Corporation/Inc/Incorporated)
- Ampersand (&) vs "and"
- Case sensitivity
- Extra whitespace / punctuation

## Tracking Template

| Error ID | Type | Subtype | Stage | Modality | Geo | Count | F_0.5 Impact | Fix Idea | Status |
|----------|------|---------|-------|----------|-----|-------|--------------|----------|--------|
| E001 | FP | Name Collision | Model | Name | US | 47 | -0.023 | Add address weight | Open |
| E002 | FN | Null Address | Blocking | Address | India | 112 | -0.018 | Name-only blocking fallback | Open |

*(Populate during Phase 2 error analysis)*