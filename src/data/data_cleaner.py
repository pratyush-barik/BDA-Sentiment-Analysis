"""
Data Cleansing, Normalization & Preprocessing Pipeline
Module: src.data.data_cleaner

Provides comprehensive text normalization, schema enforcement, null imputation,
tokenization, domain-specific stopword removal (with sentiment negation preservation),
and batch dataset serialization for Apache Spark MLlib and MapReduce pipelines.
"""

from __future__ import annotations

import ast
import datetime
import html
import json
import logging
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. CONSTANTS & LEXICAL DICTIONARIES
# ==============================================================================

CONTRACTIONS: Dict[str, str] = {
    "won't": "will not",
    "can't": "cannot",
    "cannot": "cannot",
    "shan't": "shall not",
    "ain't": "is not",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "here's": "here is",
    "where's": "where is",
    "who's": "who is",
    "how's": "how is",
    "let's": "let us",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not",
    "needn't": "need not",
}

SUFFIX_CONTRACTIONS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"n't\b", re.IGNORECASE), " not"),
    (re.compile(r"'re\b", re.IGNORECASE), " are"),
    (re.compile(r"'s\b", re.IGNORECASE), " is"),
    (re.compile(r"'d\b", re.IGNORECASE), " would"),
    (re.compile(r"'ll\b", re.IGNORECASE), " will"),
    (re.compile(r"'t\b", re.IGNORECASE), " not"),
    (re.compile(r"'ve\b", re.IGNORECASE), " have"),
    (re.compile(r"'m\b", re.IGNORECASE), " am"),
]

# Standard English Stopwords (179 words)
STANDARD_STOPWORDS: Set[str] = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
    'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
}

# Critical Sentiment Negation Tokens (MUST NEVER BE REMOVED)
NEGATION_WORDS: Set[str] = {
    'not', 'no', 'nor', 'never', 'hardly', 'scarcely', 'barely', 'without',
    'neither', 'cannot', 'nothing', 'nowhere', 'none', 'nobody', 'noone'
}

# Domain-Specific E-Commerce Noise Stopwords
ECOMMERCE_STOPWORDS: Set[str] = {
    'product', 'item', 'amazon', 'purchase', 'order', 'package', 'shipping', 'delivery',
    'bought', 'buy', 'seller', 'store', 'box', 'received', 'arrived', 'one', 'get', 'got',
    'also', 'would', 'really', 'use', 'using', 'used', 'time', 'price', 'customer', 'day',
    'days', 'review', 'reviews'
}

BASE_SCHEMA_FIELDS: Tuple[str, ...] = (
    "review_id",
    "reviewer_id",
    "reviewer_name",
    "product_id",
    "product_name",
    "product_category",
    "rating",
    "verified_purchase",
    "review_summary",
    "review_text",
    "review_date",
    "timestamp",
    "helpful_votes",
    "total_votes",
    "sentiment_label",
    "sentiment_numeric",
)

DERIVED_SCHEMA_FIELDS: Tuple[str, ...] = (
    "cleaned_text",
    "tokens",
    "token_count",
)

FULL_SCHEMA_FIELDS: Tuple[str, ...] = BASE_SCHEMA_FIELDS + DERIVED_SCHEMA_FIELDS


# ==============================================================================
# 2. STANDALONE FUNCTIONAL API
# ==============================================================================

def expand_contractions(text: str) -> str:
    """Expands standard English contractions with quote normalization."""
    if not text:
        return ""
    # Normalize unicode apostrophes
    t = text.replace('\u2019', "'").replace('\u2018', "'").replace('`', "'")
    for cont, repl in CONTRACTIONS.items():
        pattern = r'\b' + re.escape(cont) + r'\b'
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    for pattern, repl in SUFFIX_CONTRACTIONS:
        t = pattern.sub(repl, t)
    return t


def clean_text(text: Optional[str]) -> str:
    """
    Multi-stage regex text normalization:
    1. Null / NaN guard
    2. HTML entity unescape
    3. Script & style tag excision
    4. HTML tag stripping
    5. URL removal
    6. Email removal
    7. Contraction expansion
    8. Unicode normalization (NFKD accent stripping)
    9. Non-alphabetic character replacement
    10. Lowercase normalization
    11. Whitespace collapsing and trimming
    """
    if text is None:
        return ""
    if isinstance(text, float) and (math.isnan(text) or text != text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Unescape HTML entities
    t = html.unescape(text)
    # 2. Strip script and style tag contents
    t = re.sub(r'<script.*?>.*?</script>', ' ', t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r'<style.*?>.*?</style>', ' ', t, flags=re.DOTALL | re.IGNORECASE)
    # 3. Strip remaining HTML tags
    t = re.sub(r'<[^>]+>', ' ', t)
    # 4. Strip URLs
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    # 5. Strip emails
    t = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', ' ', t)
    # 6. Expand contractions
    t = expand_contractions(t)
    # 7. Normalize Unicode / accent stripping
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # 8. Strip non-alphabetic characters
    t = re.sub(r'[^a-zA-Z\s]', ' ', t)
    # 9. Lowercase
    t = t.lower()
    # 10. Collapse whitespace and trim
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def tokenize(text: str, min_token_length: int = 2) -> List[str]:
    """Tokenize normalized text, filtering out tokens shorter than min_token_length."""
    if not text:
        return []
    tokens = text.split()
    return [tok for tok in tokens if len(tok) >= min_token_length]


def remove_stopwords(
    tokens: List[str],
    include_ecommerce_stopwords: bool = True,
    preserve_negations: bool = True,
    custom_stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """
    Removes standard and domain stopwords while strictly preserving negation indicators.
    """
    stop_set = set(STANDARD_STOPWORDS)
    if include_ecommerce_stopwords:
        stop_set.update(ECOMMERCE_STOPWORDS)
    if custom_stopwords:
        stop_set.update(custom_stopwords)
    if preserve_negations:
        stop_set.difference_update(NEGATION_WORDS)
    return [tok for tok in tokens if tok not in stop_set]


def preprocess_text(
    text: Optional[str],
    include_ecommerce_stopwords: bool = True,
    preserve_negations: bool = True,
    min_token_length: int = 2,
) -> Tuple[str, List[str], int]:
    """Convenience pipeline returning (cleaned_text, tokens, token_count)."""
    cleaned = clean_text(text)
    toks = tokenize(cleaned, min_token_length=min_token_length)
    filtered = remove_stopwords(
        toks,
        include_ecommerce_stopwords=include_ecommerce_stopwords,
        preserve_negations=preserve_negations,
    )
    return cleaned, filtered, len(filtered)


# ==============================================================================
# 3. RECORD & DATAFRAME CLEANING ENGINE
# ==============================================================================

def clean_record(
    record: Dict[str, Any],
    drop_invalid: bool = False,
    recompute_sentiment: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Validates, coerces types, coalesces nulls, and appends derived NLP fields for a single record.
    Returns None if drop_invalid=True and critical fields or text are completely missing.
    """
    if not isinstance(record, dict):
        if drop_invalid:
            return None
        record = {}

    cleaned: Dict[str, Any] = {}

    # Primary Identifiers
    rev_id = record.get("review_id")
    cleaned["review_id"] = str(rev_id).strip() if rev_id is not None and str(rev_id).strip() else "REV_UNKNOWN"

    usr_id = record.get("reviewer_id")
    cleaned["reviewer_id"] = str(usr_id).strip() if usr_id is not None and str(usr_id).strip() else "USR_UNKNOWN"

    usr_name = record.get("reviewer_name")
    cleaned["reviewer_name"] = str(usr_name).strip() if usr_name is not None and str(usr_name).strip() else "Anonymous"

    prod_id = record.get("product_id")
    cleaned["product_id"] = str(prod_id).strip() if prod_id is not None and str(prod_id).strip() else "PROD_UNKNOWN"

    prod_name = record.get("product_name")
    cleaned["product_name"] = str(prod_name).strip() if prod_name is not None and str(prod_name).strip() else "Generic Product"

    prod_cat = record.get("product_category")
    cleaned["product_category"] = str(prod_cat).strip() if prod_cat is not None and str(prod_cat).strip() else "Uncategorized"

    # Rating coercion & clamping
    try:
        r = float(record.get("rating", 3.0))
        if math.isnan(r):
            r = 3.0
    except (ValueError, TypeError):
        r = 3.0
    cleaned["rating"] = round(max(1.0, min(5.0, r)), 1)

    # Verified purchase
    vp = record.get("verified_purchase")
    if isinstance(vp, bool):
        cleaned["verified_purchase"] = vp
    elif isinstance(vp, (int, float)):
        cleaned["verified_purchase"] = bool(vp)
    elif isinstance(vp, str):
        cleaned["verified_purchase"] = vp.lower() in ("true", "1", "yes", "t")
    else:
        cleaned["verified_purchase"] = False

    # Review text fields
    summary = record.get("review_summary")
    cleaned["review_summary"] = str(summary).strip() if summary is not None else ""

    body = record.get("review_text")
    cleaned["review_text"] = str(body).strip() if body is not None else ""

    # If review_text is empty, fallback to summary
    if not cleaned["review_text"] and cleaned["review_summary"]:
        cleaned["review_text"] = cleaned["review_summary"]

    # Date and timestamp
    rdate = record.get("review_date")
    if rdate and isinstance(rdate, str) and len(rdate.strip()) == 10:
        cleaned["review_date"] = rdate.strip()
    else:
        cleaned["review_date"] = "2024-01-01"

    ts = record.get("timestamp")
    try:
        cleaned["timestamp"] = int(ts) if ts is not None else int(datetime.datetime(2024, 1, 1).timestamp())
    except (ValueError, TypeError):
        cleaned["timestamp"] = int(datetime.datetime(2024, 1, 1).timestamp())

    # Helpful & Total Votes
    try:
        hv = int(record.get("helpful_votes", 0))
    except (ValueError, TypeError):
        hv = 0
    hv = max(0, hv)
    cleaned["helpful_votes"] = hv

    try:
        tv = int(record.get("total_votes", hv))
    except (ValueError, TypeError):
        tv = hv
    cleaned["total_votes"] = max(hv, tv, 0)

    # Sentiment labels and numeric mapping
    s_label = record.get("sentiment_label")
    if not recompute_sentiment and s_label in ("positive", "neutral", "negative"):
        cleaned["sentiment_label"] = s_label
    else:
        if cleaned["rating"] >= 4.0:
            cleaned["sentiment_label"] = "positive"
        elif cleaned["rating"] <= 2.0:
            cleaned["sentiment_label"] = "negative"
        else:
            cleaned["sentiment_label"] = "neutral"

    s_num = record.get("sentiment_numeric")
    try:
        s_num = float(s_num)
        if recompute_sentiment or s_num not in (0.0, 1.0, 2.0):
            raise ValueError
        cleaned["sentiment_numeric"] = s_num
    except (ValueError, TypeError):
        if cleaned["sentiment_label"] == "positive":
            cleaned["sentiment_numeric"] = 2.0
        elif cleaned["sentiment_label"] == "negative":
            cleaned["sentiment_numeric"] = 0.0
        else:
            cleaned["sentiment_numeric"] = 1.0

    # Text Preprocessing & Derived Fields
    combined_raw = f"{cleaned['review_summary']} {cleaned['review_text']}".strip()
    cleaned_t, toks, tok_count = preprocess_text(combined_raw)

    cleaned["cleaned_text"] = cleaned_t
    cleaned["tokens"] = toks
    cleaned["token_count"] = tok_count

    if drop_invalid and not cleaned_t and not toks:
        return None

    return cleaned


def clean_dataframe(
    df: pd.DataFrame,
    drop_missing_keys: bool = True,
    drop_empty_text: bool = True,
    recompute_sentiment: bool = False,
) -> pd.DataFrame:
    """
    Cleans and standardizes a Pandas DataFrame of review records.
    """
    df = df.copy()

    # Drop missing primary keys if requested
    if drop_missing_keys:
        subset = [col for col in ["review_id", "product_id"] if col in df.columns]
        if subset:
            df = df.dropna(subset=subset)

    # Coalesce string metadata
    for col, default_val in [
        ("review_id", "REV_UNKNOWN"),
        ("reviewer_id", "USR_UNKNOWN"),
        ("reviewer_name", "Anonymous"),
        ("product_id", "PROD_UNKNOWN"),
        ("product_name", "Generic Product"),
        ("product_category", "Uncategorized"),
        ("review_summary", ""),
        ("review_text", ""),
        ("review_date", "2024-01-01"),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(default_val).astype(str).str.strip()
        else:
            df[col] = default_val

    # Numeric ratings
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0).clip(1.0, 5.0).round(1)
    else:
        df["rating"] = 3.0

    # Verified purchase boolean
    def _coerce_bool(val: Any) -> bool:
        if pd.isna(val):
            return False
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        return str(val).lower() in ("true", "1", "yes", "t")

    if "verified_purchase" in df.columns:
        df["verified_purchase"] = df["verified_purchase"].apply(_coerce_bool)
    else:
        df["verified_purchase"] = False

    # Timestamps & Votes
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(1704067200).astype(np.int64)
    else:
        df["timestamp"] = 1704067200

    if "helpful_votes" in df.columns:
        df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    else:
        df["helpful_votes"] = 0

    if "total_votes" in df.columns:
        df["total_votes"] = pd.to_numeric(df["total_votes"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    else:
        df["total_votes"] = df["helpful_votes"]
    df["total_votes"] = np.maximum(df["total_votes"], df["helpful_votes"])

    # Sentiment labels
    conditions = [df["rating"] >= 4.0, df["rating"] <= 2.0]
    derived_labels = ["positive", "negative"]
    derived_numerics = [2.0, 0.0]

    if recompute_sentiment or "sentiment_label" not in df.columns:
        df["sentiment_label"] = np.select(conditions, derived_labels, default="neutral")
        df["sentiment_numeric"] = np.select(conditions, derived_numerics, default=1.0)
    else:
        df["sentiment_label"] = df["sentiment_label"].fillna(
            pd.Series(np.select(conditions, derived_labels, default="neutral"), index=df.index)
        )
        if "sentiment_numeric" not in df.columns:
            df["sentiment_numeric"] = np.select(conditions, derived_numerics, default=1.0)
        else:
            df["sentiment_numeric"] = pd.to_numeric(df["sentiment_numeric"], errors="coerce").fillna(
                pd.Series(np.select(conditions, derived_numerics, default=1.0), index=df.index)
            )

    # Review text fallback if body is empty
    empty_body_mask = (df["review_text"] == "") & (df["review_summary"] != "")
    df.loc[empty_body_mask, "review_text"] = df.loc[empty_body_mask, "review_summary"]

    # Text Normalization & Tokens
    combined_raw = df["review_summary"] + " " + df["review_text"]
    df["cleaned_text"] = combined_raw.apply(clean_text)
    df["tokens"] = df["cleaned_text"].apply(lambda t: remove_stopwords(tokenize(t)))
    df["token_count"] = df["tokens"].apply(len).astype(int)

    if drop_empty_text:
        df = df[(df["cleaned_text"] != "") | (df["token_count"] > 0)]

    # Reorder columns to full schema order
    ordered_cols = [c for c in FULL_SCHEMA_FIELDS if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + remaining].reset_index(drop=True)


# ==============================================================================
# 4. DATA CLEANER & REVIEW PREPROCESSOR CLASS
# ==============================================================================

class DataCleaner:
    """
    Unified Data Cleaner & Preprocessor class supporting record-level,
    Pandas DataFrame, and PySpark DataFrame pipelines.
    """

    def __init__(
        self,
        include_ecommerce_stopwords: bool = True,
        preserve_negations: bool = True,
        min_token_length: int = 2,
        custom_stopwords: Optional[Set[str]] = None,
    ):
        self.include_ecommerce_stopwords = include_ecommerce_stopwords
        self.preserve_negations = preserve_negations
        self.min_token_length = min_token_length
        self.custom_stopwords = custom_stopwords or set()

    def clean_text(self, text: Optional[str]) -> str:
        return clean_text(text)

    def tokenize(self, text: str) -> List[str]:
        return tokenize(text, min_token_length=self.min_token_length)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return remove_stopwords(
            tokens,
            include_ecommerce_stopwords=self.include_ecommerce_stopwords,
            preserve_negations=self.preserve_negations,
            custom_stopwords=self.custom_stopwords,
        )

    def preprocess_text(self, text: Optional[str]) -> Tuple[str, List[str], int]:
        return preprocess_text(
            text,
            include_ecommerce_stopwords=self.include_ecommerce_stopwords,
            preserve_negations=self.preserve_negations,
            min_token_length=self.min_token_length,
        )

    def clean_record(
        self,
        record: Dict[str, Any],
        drop_invalid: bool = False,
        recompute_sentiment: bool = False,
    ) -> Optional[Dict[str, Any]]:
        return clean_record(
            record,
            drop_invalid=drop_invalid,
            recompute_sentiment=recompute_sentiment,
        )

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        drop_missing_keys: bool = True,
        drop_empty_text: bool = True,
        recompute_sentiment: bool = False,
    ) -> pd.DataFrame:
        return clean_dataframe(
            df,
            drop_missing_keys=drop_missing_keys,
            drop_empty_text=drop_empty_text,
            recompute_sentiment=recompute_sentiment,
        )

    def clean_spark_dataframe(self, spark_df: Any, spark: Optional[Any] = None) -> Any:
        """
        Cleans a PySpark DataFrame using native PySpark SQL functions without UDF serialization overhead.
        """
        try:
            from pyspark.sql import functions as F
            from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
        except ImportError:
            raise RuntimeError("PySpark is not installed or available in this environment.")

        df = spark_df

        # Coalesce text fields
        summary_col = F.coalesce(F.col("review_summary"), F.lit(""))
        text_col = F.coalesce(F.col("review_text"), F.lit(""))
        combined = F.concat_ws(" ", summary_col, text_col)

        # Regex normalization
        t = F.regexp_replace(combined, r"<script.*?>.*?</script>", " ")
        t = F.regexp_replace(t, r"<style.*?>.*?</style>", " ")
        t = F.regexp_replace(t, r"<[^>]+>", " ")
        t = F.regexp_replace(t, r"https?://\S+|www\.\S+", " ")
        t = F.regexp_replace(t, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ")
        t = F.lower(t)
        t = F.regexp_replace(t, r"[^a-z\s]", " ")
        t = F.regexp_replace(t, r"\s+", " ")
        cleaned_col = F.trim(t)

        df = df.withColumn("cleaned_text", cleaned_col)

        # Tokenization via Spark MLlib RegexTokenizer
        tokenizer = RegexTokenizer(
            inputCol="cleaned_text",
            outputCol="raw_tokens",
            pattern=r"\s+",
            minTokenLength=self.min_token_length,
        )
        df = tokenizer.transform(df)

        # Stopword filtering
        effective_stops = (
            STANDARD_STOPWORDS | (ECOMMERCE_STOPWORDS if self.include_ecommerce_stopwords else set())
        ) - NEGATION_WORDS
        remover = StopWordsRemover(inputCol="raw_tokens", outputCol="tokens", stopWords=list(effective_stops))
        df = remover.transform(df)
        df = df.drop("raw_tokens")
        df = df.withColumn("token_count", F.size(F.col("tokens")))

        return df


# Class alias for contract equivalence
ReviewPreprocessor = DataCleaner


# ==============================================================================
# 5. DATASET SERIALIZATION & PERSISTENCE
# ==============================================================================

def clean_and_save(
    input_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/processed",
    file_prefix: str = "cleaned_reviews",
    formats: Tuple[str, ...] = ("csv", "parquet"),
    drop_empty: bool = True,
) -> Dict[str, Path]:
    """
    Reads a raw dataset, cleans it via DataCleaner, and serializes the processed dataset
    to CSV and/or Parquet in output_dir. Returns a dict mapping format to saved Path.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Source file not found at {input_path}")

    # Read input file based on extension
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() in (".json", ".jsonl"):
        df = pd.read_json(input_path, lines=True)
    else:
        df = pd.read_csv(input_path)

    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_dataframe(df, drop_empty_text=drop_empty)

    saved_paths: Dict[str, Path] = {}

    if "parquet" in formats:
        parquet_path = output_dir / f"{file_prefix}.parquet"
        try:
            cleaned_df.to_parquet(parquet_path, engine="pyarrow", index=False)
            saved_paths["parquet"] = parquet_path
            logger.info("Successfully exported cleaned Parquet to %s", parquet_path)
        except (ImportError, Exception) as exc:
            logger.warning("Could not export Parquet (pyarrow may not be installed): %s", exc)

    if "csv" in formats:
        csv_path = output_dir / f"{file_prefix}.csv"
        csv_df = cleaned_df.copy()
        # Serialize list of tokens safely for CSV storage
        csv_df["tokens"] = csv_df["tokens"].apply(json.dumps)
        csv_df.to_csv(csv_path, index=False)
        saved_paths["csv"] = csv_path
        logger.info("Successfully exported cleaned CSV to %s", csv_path)

    return saved_paths


def load_cleaned_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a cleaned dataset from CSV or Parquet, safely restoring tokens to List[str].
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {file_path}")

    if file_path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path)

    df = pd.read_csv(file_path)
    if "tokens" in df.columns:
        def _parse_tokens(val: Any) -> List[str]:
            if isinstance(val, list):
                return val
            if pd.isna(val) or not val:
                return []
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return str(val).split()

        df["tokens"] = df["tokens"].apply(_parse_tokens)
    return df
