"""
tests/test_tier1_preprocessing.py
Tier 1: Feature Coverage (Category-Partition: Ingestion, Schema, Preprocessing).
Big Data Product Review Sentiment Analysis System.
"""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import pytest

try:
    from tests.conftest import safe_import
except ImportError:
    from conftest import safe_import


EXPECTED_SCHEMA = {
    "review_id": str,
    "reviewer_id": str,
    "reviewer_name": str,
    "product_id": str,
    "product_name": str,
    "product_category": str,
    "rating": float,
    "verified_purchase": bool,
    "review_summary": str,
    "review_text": str,
    "review_date": str,
    "timestamp": int,
    "helpful_votes": int,
    "total_votes": int,
    "sentiment_label": str,
    "sentiment_numeric": float,
}

VALID_CATEGORIES = {
    "Electronics",
    "Books",
    "Home & Kitchen",
    "Beauty & Personal Care",
    "Sports & Outdoors",
}


@pytest.mark.tier1
def test_schema_conformance_and_types(tmp_path: Path):
    """
    Test 1.1: Validate that raw generated review datasets conform strictly
    to the 16-field schema, valid category domains, and non-empty primary keys.
    """
    gen_mod = safe_import("src.data.review_generator")
    if gen_mod is None:
        pytest.skip("Milestone M1 review_generator not yet implemented")

    generate_reviews = getattr(gen_mod, "generate_reviews", None)
    if generate_reviews is None:
        pytest.skip("generate_reviews function not found in src.data.review_generator")

    # Generate 50 records into a designated temporary CSV file for strict sandbox isolation
    target_csv = tmp_path / "reviews.csv"
    out_path = generate_reviews(
        num_records=50, seed=42, output_format="csv", output_path=target_csv
    )
    csv_file = Path(out_path) if out_path else target_csv
    assert csv_file.exists(), f"Generated CSV file does not exist at {csv_file}"

    df = pd.read_csv(csv_file)

    # Invariant 1: Exactly 16 columns
    assert len(df.columns) == 16, f"Expected 16 columns, got {len(df.columns)}: {df.columns.tolist()}"

    # Invariant 2: All required schema fields present
    for col_name in EXPECTED_SCHEMA:
        assert col_name in df.columns, f"Missing required column: {col_name}"

    # Invariant 3: Product categories belong to syllabus domain
    categories_present = set(df["product_category"].dropna().unique())
    assert categories_present.issubset(VALID_CATEGORIES), f"Unknown categories: {categories_present - VALID_CATEGORIES}"

    # Invariant 4: Primary keys non-empty and non-null
    assert (df["review_id"].str.len() > 0).all()
    assert (df["product_id"].str.len() > 0).all()

    # Invariant 5: Ratings within valid range [1.0, 5.0]
    assert df["rating"].min() >= 1.0
    assert df["rating"].max() <= 5.0

    # Invariant 6: Total votes >= helpful votes >= 0
    assert (df["helpful_votes"] >= 0).all()
    assert (df["total_votes"] >= df["helpful_votes"]).all()

    # Adversarial edge case: num_records <= 0 must raise ValueError
    with pytest.raises(ValueError):
        generate_reviews(num_records=0, seed=42, output_path=tmp_path / "reviews_zero.csv")


@pytest.mark.tier1
def test_null_and_missing_value_coalescing():
    """
    Test 1.2: Validate that null review summaries and missing vote counts
    are safely coalesced, while records missing essential text or IDs are dropped.
    """
    cleaner_mod = safe_import("src.data.data_cleaner")
    if cleaner_mod is None:
        pytest.skip("Milestone M1 data_cleaner not yet implemented")

    cleaner_cls = getattr(cleaner_mod, "DataCleaner", None)
    if cleaner_cls is None:
        pytest.skip("DataCleaner class not found in src.data.data_cleaner")

    cleaner = cleaner_cls()
    dirty_records = [
        {
            "review_id": "REV_00000001",
            "reviewer_id": "USR_00001",
            "reviewer_name": "Test User",
            "product_id": "PROD_001",
            "product_name": "Test Item",
            "product_category": "Books",
            "rating": 4.0,
            "verified_purchase": True,
            "review_summary": None,  # Should coalesce to ""
            "review_text": "Thoroughly enjoyed the chapters on distributed consensus.",
            "review_date": "2024-01-01",
            "timestamp": 1704067200,
            "helpful_votes": None,  # Should coalesce to 0
            "total_votes": None,    # Should coalesce to 0
            "sentiment_label": "positive",
            "sentiment_numeric": 0.0,
        },
        {
            "review_id": "REV_00000002",
            "reviewer_id": "USR_00002",
            "product_id": "PROD_002",
            "rating": 4.0,
            "review_summary": None,
            "review_text": None,  # Missing text -> MUST be dropped
        },
        {
            "review_id": "REV_00000003",
            "reviewer_id": "USR_00003",
            "product_id": "PROD_003",
            "rating": 4.0,
            "review_summary": "",
            "review_text": "   ",  # Whitespace-only text -> MUST be dropped
        },
        {
            "review_id": None,  # Missing ID -> MUST be dropped
            "reviewer_id": "USR_00004",
            "product_id": "PROD_004",
            "rating": 3.0,
            "review_text": "Good enough",
        },
    ]

    clean_method = (
        getattr(cleaner, "clean_dataframe", None)
        or getattr(cleaner, "clean_records", None)
        or getattr(cleaner, "clean", None)
    )
    if clean_method is None:
        pytest.skip("Cleaning method not found on DataCleaner")

    dirty_df = pd.DataFrame(dirty_records)
    cleaned = clean_method(dirty_df)

    if isinstance(cleaned, pd.DataFrame):
        cleaned_list = cleaned.to_dict(orient="records")
    else:
        cleaned_list = list(cleaned)

    # Invariant: Exactly 1 record survives (Record 1)
    assert len(cleaned_list) == 1, f"Expected 1 valid record, got {len(cleaned_list)}"
    survivor = cleaned_list[0]
    assert survivor["review_id"] == "REV_00000001"
    # Coalescing checks
    assert survivor["review_summary"] == "" or survivor["review_summary"] is not None
    assert survivor["helpful_votes"] == 0
    assert survivor["total_votes"] == 0

    # Invariant: DataCleaner output must strictly preserve all 16 canonical schema fields
    if isinstance(cleaned, pd.DataFrame):
        assert len(cleaned.columns) >= 16, (
            f"Expected at least 16 columns from DataCleaner, got {len(cleaned.columns)}: {cleaned.columns.tolist()}"
        )
        for col_name in EXPECTED_SCHEMA:
            assert col_name in cleaned.columns, f"DataCleaner DataFrame output missing canonical column: {col_name}"

    assert len(survivor) >= 16, f"Expected at least 16 fields in survivor record, got {len(survivor)}"
    for col_name in EXPECTED_SCHEMA:
        assert col_name in survivor, f"DataCleaner output record missing canonical field: {col_name}"

    # Also verify single-record clean_record API preserves all 16 canonical fields
    clean_rec_fn = getattr(cleaner, "clean_record", None)
    if clean_rec_fn is not None:
        rec_cleaned = clean_rec_fn(dirty_records[0])
        assert rec_cleaned is not None, "clean_record returned None for valid record"
        assert len(rec_cleaned) >= 16, f"Expected at least 16 fields from clean_record, got {len(rec_cleaned)}"
        for col_name in EXPECTED_SCHEMA:
            assert col_name in rec_cleaned, f"clean_record output missing canonical field: {col_name}"


@pytest.mark.tier1
def test_cleaner_schema_preservation_and_enrichment(sample_reviews_data):
    """
    Test 1.3: Validate that DataCleaner preserves all 16 schema columns from raw reviews
    and adds derived NLP fields (cleaned_text, tokens, token_count).
    Ensures cleaner schema regressions (e.g. dropped columns) are caught.
    """
    cleaner_mod = safe_import("src.data.data_cleaner")
    if cleaner_mod is None:
        pytest.skip("Milestone M1 data_cleaner not yet implemented")

    cleaner_cls = getattr(cleaner_mod, "DataCleaner", None)
    cleaner = cleaner_cls() if cleaner_cls else None
    if cleaner is None:
        pytest.skip("DataCleaner not found")

    raw_df = pd.DataFrame(sample_reviews_data)
    cleaned_df = cleaner.clean_dataframe(raw_df)

    assert len(cleaned_df) == len(raw_df), f"Expected {len(raw_df)} rows, got {len(cleaned_df)}"
    for col_name in EXPECTED_SCHEMA:
        assert col_name in cleaned_df.columns, f"DataCleaner dropped schema column: {col_name}"

    for derived_col in ["cleaned_text", "tokens", "token_count"]:
        assert derived_col in cleaned_df.columns, f"DataCleaner missing derived column: {derived_col}"


@pytest.mark.tier1
def test_regex_cleaning_html_urls_special_chars():
    """
    Test 1.3: Ensure HTML tags, hyperlinks, and punctuation noise are stripped,
    and text is normalized to lowercase.
    """
    cleaner_mod = safe_import("src.data.data_cleaner")
    if cleaner_mod is None:
        pytest.skip("Milestone M1 data_cleaner not yet implemented")

    cleaner_cls = getattr(cleaner_mod, "DataCleaner", None)
    cleaner = cleaner_cls() if cleaner_cls else None
    clean_fn = getattr(cleaner, "clean_text", None) or getattr(cleaner_mod, "clean_text", None) or getattr(cleaner, "normalize_text", None)
    if clean_fn is None:
        pytest.skip("clean_text / normalize_text function not found in src.data.data_cleaner")

    dirty_text = "<p>Superb sound! Check <a href='https://amazon.com/item'>manual</a> & visit www.audio.com. Rating: 10/10! ⭐ #Awesome</p>"
    cleaned = clean_fn(dirty_text)

    # Assertions
    assert "<" not in cleaned and ">" not in cleaned, "HTML tags were not stripped"
    assert "http" not in cleaned and "www" not in cleaned, "URLs were not stripped"
    assert "⭐" not in cleaned and "#" not in cleaned, "Emojis or special symbols were not stripped"
    assert cleaned == cleaned.lower(), "Text was not lowercased"
    assert "superb" in cleaned
    assert "sound" in cleaned
    assert "awesome" in cleaned

    # Edge cases
    assert clean_fn("") == "", "Empty string should return empty string"
    assert clean_fn("!@#$%^&*()") == "", "Punctuation-only string should return empty string"


@pytest.mark.tier1
def test_contraction_expansion():
    """
    Test 1.4: Ensure e-commerce contractions are expanded into full words
    so negation sentiment tokens are preserved.
    """
    cleaner_mod = safe_import("src.data.data_cleaner")
    if cleaner_mod is None:
        pytest.skip("Milestone M1 data_cleaner not yet implemented")

    cleaner_cls = getattr(cleaner_mod, "DataCleaner", None)
    cleaner = cleaner_cls() if cleaner_cls else None
    expand_fn = (
        getattr(cleaner, "expand_contractions", None)
        or getattr(cleaner_mod, "expand_contractions", None)
        or getattr(cleaner, "clean_text", None)
    )
    if expand_fn is None:
        pytest.skip("expand_contractions not found in src.data.data_cleaner")

    dirty_text = "Don't buy this. Won't charge, can't turn on, and it's completely unusable."
    expanded = expand_fn(dirty_text)

    # Contract invariants: negation words must be intact
    assert "do not" in expanded.lower() or "not" in expanded.lower()
    assert "will not" in expanded.lower() or "not" in expanded.lower()
    assert "cannot" in expanded.lower() or "can not" in expanded.lower() or "not" in expanded.lower()
    assert "don't" not in expanded.lower()
    assert "won't" not in expanded.lower()


@pytest.mark.tier1
def test_ecommerce_stopword_removal_and_tokenization():
    """
    Test 1.5: Verify standard English stopwords and e-commerce noise words are filtered
    while retaining sentiment tokens with length >= 2.
    """
    cleaner_mod = safe_import("src.data.data_cleaner")
    if cleaner_mod is None:
        pytest.skip("Milestone M1 data_cleaner not yet implemented")

    cleaner_cls = getattr(cleaner_mod, "DataCleaner", None)
    cleaner = cleaner_cls() if cleaner_cls else None
    if cleaner is None:
        pytest.skip("DataCleaner not found")

    input_text = "I ordered this product on Amazon and the seller item had fast shipping but the battery is terrible"

    if hasattr(cleaner, "preprocess_text"):
        _, tokens, _ = cleaner.preprocess_text(input_text)
    elif hasattr(cleaner, "remove_stopwords") and hasattr(cleaner, "tokenize"):
        tokens = cleaner.remove_stopwords(cleaner.tokenize(input_text))
    else:
        pytest.skip("No tokenization/stopword removal method found on DataCleaner")

    domain_stopwords = {"product", "item", "amazon", "seller", "shipping"}
    for sw in domain_stopwords:
        assert sw not in tokens, f"Domain stopword '{sw}' was not filtered"

    # Crucial sentiment/topic tokens must be preserved
    assert "terrible" in tokens
    assert "fast" in tokens
    assert "battery" in tokens

    # Minimum token length invariant
    assert all(len(t) >= 2 for t in tokens), "Found token shorter than 2 characters"

    # Edge case: Stopword-only input
    if hasattr(cleaner, "preprocess_text"):
        assert cleaner.preprocess_text("this is the of and or but")[1] == []


@pytest.mark.tier1
def test_sentiment_label_assignment_thresholds():
    """
    Test 1.6: Validate the 3-tier sentiment classification boundary rules:
      - rating >= 4.0 -> positive
      - rating <= 2.0 -> negative
      - rating == 3.0 -> neutral
    """
    cleaner_mod = safe_import("src.data.data_cleaner")
    if cleaner_mod is None:
        pytest.skip("Milestone M1 data_cleaner not yet implemented")

    cleaner_cls = getattr(cleaner_mod, "DataCleaner", None)
    cleaner = cleaner_cls() if cleaner_cls else None
    if cleaner is None:
        pytest.skip("DataCleaner not found")

    derive_fn = getattr(cleaner, "derive_sentiment", None) or getattr(cleaner_mod, "derive_sentiment", None)

    # Full 3-tier continuous rating matrix including fractional ratings (1.5, 2.5, 3.5, 4.5)
    # Boundary rules: rating <= 2.0 -> negative (0.0); 2.0 < rating < 4.0 -> neutral (1.0); rating >= 4.0 -> positive (2.0)
    test_matrix = [
        (1.0, "negative", 0.0),
        (1.5, "negative", 0.0),
        (2.0, "negative", 0.0),
        (2.5, "neutral", 1.0),
        (3.0, "neutral", 1.0),
        (3.5, "neutral", 1.0),
        (4.0, "positive", 2.0),
        (4.5, "positive", 2.0),
        (5.0, "positive", 2.0),
    ]

    for rating, expected_label, expected_numeric in test_matrix:
        if derive_fn is not None:
            res = derive_fn(rating)
            actual_label = res[0] if isinstance(res, (tuple, list)) else res
            actual_numeric = res[1] if isinstance(res, (tuple, list)) and len(res) > 1 else None
        else:
            rec = cleaner.clean_record({"rating": rating, "review_text": "Good"}, recompute_sentiment=True)
            actual_label = rec["sentiment_label"]
            actual_numeric = rec["sentiment_numeric"]

        assert actual_label == expected_label, (
            f"Rating {rating}: expected sentiment label '{expected_label}', got '{actual_label}'"
        )
        if actual_numeric is not None:
            assert actual_numeric == expected_numeric, (
                f"Rating {rating}: expected sentiment numeric {expected_numeric}, got {actual_numeric}"
            )

        # Invariant checks: catch boundary bleed / classification leakages
        if rating < 4.0:
            assert actual_label != "positive", f"Rating {rating} must NEVER be labeled positive"
        if rating > 2.0:
            assert actual_label != "negative", f"Rating {rating} must NEVER be labeled negative"

    # Also verify batch DataFrame cleaning respects fractional thresholds
    clean_df_fn = getattr(cleaner, "clean_dataframe", None)
    if clean_df_fn is not None:
        batch_records = [
            {"review_id": f"REV_T_{i:04d}", "product_id": f"P_{i}", "rating": r, "review_text": "Evaluating threshold"}
            for i, (r, _, _) in enumerate(test_matrix)
        ]
        df_in = pd.DataFrame(batch_records)
        df_out = clean_df_fn(df_in, recompute_sentiment=True)
        for i, (r, exp_label, exp_num) in enumerate(test_matrix):
            row = df_out.iloc[i]
            assert row["sentiment_label"] == exp_label, (
                f"DataFrame cleaning: Rating {r} expected '{exp_label}', got '{row['sentiment_label']}'"
            )
            assert row["sentiment_numeric"] == exp_num, (
                f"DataFrame cleaning: Rating {r} expected numeric {exp_num}, got {row['sentiment_numeric']}"
            )


@pytest.mark.tier1
def test_rating_range_boundary_filtering():
    """
    Test 1.7: Validate out-of-bounds ratings are clamped or filtered to valid range [1.0, 5.0].
    """
    cleaner_mod = safe_import("src.data.data_cleaner")
    if cleaner_mod is None:
        pytest.skip("Milestone M1 data_cleaner not yet implemented")

    cleaner_cls = getattr(cleaner_mod, "DataCleaner", None)
    if cleaner_cls is None:
        pytest.skip("DataCleaner class not found in src.data.data_cleaner")

    cleaner = cleaner_cls()
    clean_method = (
        getattr(cleaner, "clean_dataframe", None)
        or getattr(cleaner, "clean_records", None)
        or getattr(cleaner, "clean", None)
    )
    if clean_method is None:
        pytest.skip("Cleaning method not found on DataCleaner")

    records = [
        {"review_id": f"R_{i}", "reviewer_id": "USR_1", "product_id": "PROD_1", "rating": r, "review_text": "Good"}
        for i, r in enumerate([-1.0, 0.0, 5.5, 10.0, 3.5])
    ]

    dirty_df = pd.DataFrame(records)
    res = clean_method(dirty_df)

    # Invariant: All ratings in cleaned DataFrame must strictly satisfy 1.0 <= rating <= 5.0
    assert (res["rating"] >= 1.0).all(), "Found rating < 1.0 in cleaned dataset"
    assert (res["rating"] <= 5.0).all(), "Found rating > 5.0 in cleaned dataset"
    assert 3.5 in res["rating"].values, "Valid rating 3.5 was corrupted or lost"
