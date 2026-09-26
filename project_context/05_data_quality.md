# Data Quality Analysis

## Null / Missing Values

| Field | S1 Train | S2 Train | S3 Train | S1 Test | S2 Test | S3 Test |
|-------|----------|----------|----------|---------|---------|---------|
| entity_id | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| business_name | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| business_address | 0 (0%) | **168,967 (3.36%)** | **175,916 (3.33%)** | 0 (0%) | **129,408 (2.65%)** | **136,098 (2.68%)** |
| country | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |

**Key finding**: Address nulls exist **only in S2/S3** (~3%), never in S1. S1 addresses are complete.

## Duplicate Analysis

### Entity ID Uniqueness
- ✅ All entity_ids unique within each source file (verified)
- ✅ Zero cross-source ID overlap (verified)

### Business Name Duplicates (within source)
| Source | Total Rows | Unique Names | Duplicate Rate | Implication |
|--------|------------|--------------|----------------|-------------|
| S1 Train | 2.2M | 1.54M | **30.3%** | Many distinct entities share names |
| S2 Train | 5.0M | 4.40M | **12.6%** | Moderate name collision |
| S3 Train | 5.3M | 4.65M | **12.0%** | Moderate name collision |
| S1 Test | 1.7M | 1.24M | **28.5%** | Similar to train |
| S2 Test | 4.9M | 4.31M | **11.8%** | Similar to train |
| S3 Test | 5.1M | 4.52M | **11.0%** | Similar to train |

**Implication**: Name alone is **not a unique key**. Must combine with address/country for disambiguation.

### Business Address Duplicates (within source)
| Source | Total Rows | Unique Addresses | Duplicate Rate |
|--------|------------|------------------|----------------|
| S1 Train | 2.2M | 2.13M | 3.5% |
| S2 Train | 5.0M | 4.34M | 13.8% |
| S3 Train | 5.3M | 4.63M | 12.4% |

Addresses more unique than names but still have collisions (e.g., same building, different businesses).

## Noise Patterns (Qualitative)

### Business Name Noise
| Pattern | Examples | Frequency |
|---------|----------|-----------|
| Legal suffix variation | Corp/Corporation/Inc/Incorporated/Ltd/Limited/Pvt/Private/LLC/L.L.C. | Very High |
| Abbreviation | "Intl" vs "International", "Mgmt" vs "Management" | High |
| DBA/Trade name | "Orelee's Barbershop" vs legal "Orelee's Hair Salon Inc" | Medium |
| Punctuation | "A&B" vs "A and B" vs "A & B" | High |
| Word reordering | "Prime Money" vs "Money Prime Services" | Medium |
| Typos | "Tetlecommunication" vs "Telecommunication" | Low-Medium |
| Transliteration (India) | Devanagari script mixed with English | High (India) |
| Special prefixes | "<< Team Ecole", "-- Holloway Peak" | Low |
| Case variation | "SCI Ptit Àmicale" vs "sci ptit amicale" | High |

### Business Address Noise
| Pattern | Examples | Frequency |
|---------|----------|-----------|
| Street type abbreviation | Rd/Road, St/Street, Ave/Avenue, Blvd/Boulevard | Very High |
| Component reordering | "City, State ZIP" vs "ZIP City State" vs "Street, City" | High |
| Missing components | No ZIP, no state, no unit number | High |
| Landmark references | "Near SBI ATM", "Opp. Railway Station" | Medium (India) |
| Municipal numbering | "Door No 183", "KH NO. -570/13", "G-3/571" | Medium (India) |
| Transliteration | Devanagari address components | High (India) |
| French format | "63 R. DE DIEPPE, LILLE, Hauts-de-France" | France only |
| Partial addresses | "Mack Rd, Haltom City, Texas" (no street number) | Medium |

## Country-Specific Patterns

### United States
- Standard US address formats
- State abbreviations (CA, NY, TX) or full names
- ZIP codes (5 or 9 digit)
- English business names

### India
- Devanagari script in names/addresses (~30-40% of India records)
- PIN codes (6 digit)
- Landmark-based references common
- Municipal formats (Door No, KH No, Plot No)
- Legal suffixes: "Private Limited", "Pvt Ltd", "LLP", "LLP"

### France (Test Only)
- French address format: "Number Street, City, Department/Region"
- "R." for "Rue", "Av." for "Avenue"
- Business suffixes: "SARL", "S.A.S.", "S.A.", "EURL", "SCI"
- French business names ("École", "Groupe", "Société")

## Data Quality Implications for Pipeline

### Blocking
1. **Country exact match**: Clean, reliable, 3-7× reduction
2. **Name blocking**: Must handle noise — use normalized tokens, q-grams, or phonetic encoding
3. **Address blocking**: Use token overlap (street numbers, city names, PIN/ZIP) where present
4. **Null address handling**: Fall back to name-only blocking for ~3% of S2/S3 records

### Feature Engineering
1. **Name normalization**: Lowercase, remove punctuation, expand abbreviations, handle legal suffixes
2. **Address parsing**: Extract tokens (numbers, street types, city, PIN/ZIP) for overlap features
3. **Script detection**: Separate Devanagari vs Latin for India records
4. **Cross-field features**: Name+address concatenation, name+country

### Model Robustness
1. **France generalization**: Train on US+India only; features must transfer to France
2. **Address nulls**: Model must work with name-only features for ~3% of candidates
3. **Singleton prediction**: No training examples — need calibration/validation strategy

## Quality Checks Passed
- ✅ No duplicate entity_ids within or across sources
- ✅ Ground truth references all valid (0 orphan IDs)
- ✅ Ground truth covers all S1 train entities (1:1)
- ✅ No S1 self-references in matches
- ✅ Country consistency in matches (verified via GT)
- ✅ Test S1 entities all have country in {US, India, France}

## Remaining Quality Investigations
- [ ] Exact match count distribution per S1 entity
- [ ] S2 vs S3 match balance per S1 entity  
- [ ] Name/address similarity distribution for true matches vs non-matches
- [ ] France address/name pattern analysis
- [ ] Singleton prevalence estimation for test set