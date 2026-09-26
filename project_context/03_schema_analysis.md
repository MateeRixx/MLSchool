# Schema Analysis

## Column Definitions

### `entity_id`
- **Format**: `{SOURCE_PREFIX}-{NUMERIC_ID}` (e.g., `S1-925783039`, `S2-166376419`)
- **Prefix semantics**: 
  - `S1-` = Source 1 (deduplicated reference)
  - `S2-` = Source 2
  - `S3-` = Source 3
- **Uniqueness**: Globally unique within each source file (verified)
- **Role**: Primary key for each source; foreign key in ground truth
- **No semantic meaning**: Numeric portion appears to be sequential/hash, not informative

### `business_name`
- **Type**: Free-text string
- **Nullability**: NOT NULL (0% nulls in all files)
- **Uniqueness**: 69.7% (S1) to 89.0% (S3 test) — many duplicate names exist
- **Noise patterns observed**:
  - Legal suffix variations: `Corp` vs `Corporation` vs `Inc` vs `Incorporated`
  - Abbreviations: `Pvt` vs `Private`, `Ltd` vs `Limited`, `LLC` vs `L.L.C.`
  - DBA/trade names: `Orelee's Barbershop` vs legal name
  - Punctuation: `&` vs `and`, `.` vs `,` vs nothing
  - Word order: `Prime Money` vs `Money Prime`
  - Typos: `Tetlecommunication` vs `Telecommunication`
  - Transliteration: Devanagari script (India) mixed with Latin
  - Special chars: `<< Team Ecole`, `-- Holloway Peak Inc Seafood`
- **Language mix**: English + Devanagari (India), French (France test), English (US)

### `business_address`
- **Type**: Free-text string
- **Nullability**: NULLABLE — 0% (S1), ~3% (S2/S3 train), ~2.7% (S2/S3 test)
- **Uniqueness**: 86-97% — higher than names, more discriminative
- **Noise patterns observed**:
  - Abbreviations: `Rd`/`Road`, `St`/`Street`, `Ave`/`Avenue`, `Blvd`/`Boulevard`
  - Component reordering: `City, State, ZIP` vs `ZIP City State` vs `Street, City`
  - Missing components: No PIN/ZIP, no state, no country
  - Landmark references: `Near SBI ATM`, `Opp. Railway Station`
  - Municipal formats: `Door No 183`, `KH NO. -570/13`, `G-3/571`
  - Transliteration variants
  - Partial addresses: `Mack Rd, Haltom City, Texas` (no number)
  - French format: `63 R. DE DIEPPE, LILLE, Hauts-de-France`
- **Structure**: Highly unstructured, no consistent parsing possible

### `country`
- **Type**: String label (open set)
- **Values observed**: `US`, `India`, `France` (test only)
- **Distribution**: Balanced-ish in train (60/40), more India in test (47%), France ~15%
- **Critical**: **Open set** — do NOT hardcode {US, India}. France appears only in test.
- **Blocking utility**: Perfect exact-match blocking key (no noise in country field)

## Ground Truth Columns

### `source1_entity_id`
- Foreign key to `train_source1.entity_id`
- 1:1 mapping — every S1 train entity appears exactly once

### `matched_entity_ids`
- Comma-separated list of `entity_id` values from S2 and/or S3
- **Empty string** = no matches (singleton) — **0 occurrences in train**
- **Format**: `S2-12345,S3-67890,S2-11111` (no spaces, no quotes)
- **Source mix**: ~48% S2, ~52% S3
- **Count per S1**: Range 1–11, mean 3.67
- **All IDs verified**: Exist in respective source files

## Schema Evolution: Train → Test

| Aspect | Train | Test | Action Required |
|--------|-------|------|-----------------|
| Countries | US, India | US, India, **France** | Open-set handling; no country-specific params |
| S1 singletons | 0 | >0 (unknown count) | Must predict empty lists; calibrate threshold |
| S1 count | 2.2M | 1.7M | Smaller test set |
| S2/S3 counts | ~5M each | ~5M each | Similar scale |
| Address nulls | ~3.3% | ~2.7% | Slightly cleaner |

## Feature Engineering Implications

1. **Country**: Use as primary blocking key (exact match). Do not one-hot encode (open set).
2. **Name**: Primary matching signal. Need robust normalization + similarity metrics.
3. **Address**: Secondary signal. Handle nulls. Parse tokens (numbers, street types, city names) for overlap features.
4. **Cross-field**: Name+address concatenation, name+country, address+country combinations.
5. **Language-aware**: Consider script detection (Devanagari vs Latin) for India records.

## Validation Rules (from validate_submission.py)
- Output must be TAB-separated (not CSV)
- Headers: `source1_entity_id\tmatched_entity_ids` / `source1_entity_id\tcandidate_entity_ids`
- Every test S1 entity must have exactly one row
- IDs in lists: only S2-/S3- prefixes, no duplicates within list
- Matched IDs must be subset of candidate IDs (pipeline consistency check)