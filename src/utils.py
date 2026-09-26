"""
Shared utilities for Entity Resolution pipeline.
"""

import polars as pl
import re
from typing import List, Set, Optional
from pathlib import Path

# Business suffixes to remove during normalization (multi-country)
BUSINESS_SUFFIXES = {
    # US/Generic
    "inc", "incorporated", "llc", "llc.", "ltd", "ltd.", "limited",
    "corp", "corporation", "co", "company", "companies",
    "pvt", "private", "pllc", "plc",
    "lp", "llp", "lllp", "partnership",
    "corp.", "inc.", "ltd.", "co.",
    # India
    "pvtltd", "pvtltd.", "pvt.ltd", "private limited",
    "ltdllp", "llp", "llp.",
    # France
    "sarl", "sarl.", "sas", "sas.", "sasu", "sasu.",
    "eurl", "eurl.", "sa", "sa.", "snc", "snc.",
    "sci", "sci.", "scop", "scop.", "scic", "scic.",
    # Common abbreviations
    "and", "sons", "son", "bros", "brothers",
    "group", "grp", "holding", "holdings",
    "intl", "international", "global",
    "services", "service", "solutions", "solution",
    "tech", "technology", "technologies",
    "ind", "industries", "industry",
    "mfg", "manufacturing",
    "trdg", "trading",
    "ent", "enterprises", "enterprise",
}

# Compile regex for punctuation removal
PUNCTUATION_RE = re.compile(r"[^\w\s]")
WHITESPACE_RE = re.compile(r"\s+")

# ZIP/PIN patterns
US_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
INDIA_PIN_RE = re.compile(r"\b\d{6}\b")
FRANCE_POSTAL_RE = re.compile(r"\b\d{5}\b")


def normalize_name(name: str) -> str:
    """
    Normalize business name for matching.
    
    - Lowercase
    - Replace & with 'and'
    - Remove punctuation
    - Remove business suffixes
    - Collapse whitespace
    """
    if not name:
        return ""
    
    s = name.lower()
    s = s.replace("&", " and ")
    s = PUNCTUATION_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    
    # Split into tokens
    tokens = s.split()
    
    # Remove business suffixes from the end
    while tokens and tokens[-1] in BUSINESS_SUFFIXES:
        tokens.pop()
    
    return " ".join(tokens)


def extract_name_tokens(name: str, n_tokens: int = 3) -> List[str]:
    """
    Extract first n significant tokens from normalized name.
    """
    norm = normalize_name(name)
    tokens = norm.split()
    return tokens[:n_tokens] if tokens else []


def get_first_token(name: str) -> str:
    """Get first significant token of normalized name."""
    tokens = extract_name_tokens(name, n_tokens=1)
    return tokens[0] if tokens else ""


def soundex_token(token: str) -> str:
    """Compute Soundex code for a token using jellyfish."""
    try:
        import jellyfish
        return jellyfish.soundex(token)
    except ImportError:
        # Fallback simple soundex
        return token[:4].ljust(4, "0")


def extract_zip_pin(address: str, country: str) -> Optional[str]:
    """
    Extract ZIP/PIN code from address based on country.
    
    Returns:
        - 5-digit ZIP for US
        - 6-digit PIN for India
        - 5-digit postal for France
        - First matching pattern for other/unknown countries
    """
    if not address:
        return None
    
    # US ZIP (5 digits, optionally +4)
    if country == "US":
        match = US_ZIP_RE.search(address)
        if match:
            return match.group()[:5]
    
    # India PIN (6 digits)
    elif country == "India":
        match = INDIA_PIN_RE.search(address)
        if match:
            return match.group()
    
    # France postal (5 digits)
    elif country == "France":
        match = FRANCE_POSTAL_RE.search(address)
        if match:
            return match.group()
    
    # Generic: try all patterns
    for pattern in [US_ZIP_RE, INDIA_PIN_RE, FRANCE_POSTAL_RE]:
        match = pattern.search(address)
        if match:
            return match.group()
    
    return None


def load_source_tsv(path: str, columns: List[str] = None) -> pl.LazyFrame:
    """
    Load a TSV file as Polars LazyFrame for streaming processing.
    """
    if columns is None:
        columns = ["entity_id", "business_name", "business_address", "country"]
    
    return pl.scan_csv(
        path,
        separator="\t",
        has_header=True,
        schema_overrides={col: pl.String for col in columns},
        low_memory=True,
    )


def load_source_parquet(path: str) -> pl.LazyFrame:
    """
    Load a pre-processed parquet file (already normalized).
    """
    return pl.scan_parquet(path)


def load_ground_truth(path: str) -> pl.DataFrame:
    """Load ground truth as DataFrame (small enough to fit in memory)."""
    return pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        schema_overrides={
            "source1_entity_id": pl.String,
            "matched_entity_ids": pl.String,
        },
    )


def parse_matched_ids(matched_str: str) -> List[str]:
    """Parse comma-separated matched entity IDs."""
    if not matched_str or matched_str.strip() == "":
        return []
    return [x.strip() for x in matched_str.split(",") if x.strip()]


def compute_blocking_recall(
    candidates_df: pl.DataFrame,
    gt_df: pl.DataFrame,
) -> float:
    """
    Compute blocking recall: % of ground truth pairs captured in candidates.
    
    Args:
        candidates_df: DataFrame with columns [source1_entity_id, candidate_entity_id]
        gt_df: Ground truth DataFrame with [source1_entity_id, matched_entity_ids]
    
    Returns:
        Recall as float between 0 and 1
    """
    # Build set of candidate pairs
    candidate_pairs = set(
        zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"])
    )
    
    # Count total GT pairs and captured pairs
    total_pairs = 0
    captured_pairs = 0
    
    for row in gt_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        matched_ids = parse_matched_ids(row["matched_entity_ids"])
        total_pairs += len(matched_ids)
        for m_id in matched_ids:
            if (s1_id, m_id) in candidate_pairs:
                captured_pairs += 1
    
    if total_pairs == 0:
        return 1.0
    
    return captured_pairs / total_pairs


SEED = 42
MAX_BUCKET_SIZE = 1000

# Base data paths
BASE_PATH = "/kaggle/input/datasets/summohith/amazon-ml-2026/student_resource"
PROCESSED_BASE = "/kaggle/working/MLSchool/processed"

# Output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def normalize_source_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add normalized columns to source LazyFrame.
    """
    return lf.with_columns([
        pl.col("business_name").map_elements(normalize_name, return_dtype=pl.String).alias("name_norm"),
        pl.col("business_name").map_elements(lambda x: get_first_token(x), return_dtype=pl.String).alias("name_first_token"),
        pl.col("business_name").map_elements(lambda x: " ".join(extract_name_tokens(x, 2)), return_dtype=pl.String).alias("name_first_2_tokens"),
        pl.col("business_name").map_elements(lambda x: soundex_token(get_first_token(x)), return_dtype=pl.String).alias("name_soundex"),
        pl.struct(["business_address", "country"]).map_elements(
            lambda x: extract_zip_pin(x["business_address"], x["country"]), 
            return_dtype=pl.String
        ).alias("zip_pin"),
    ])