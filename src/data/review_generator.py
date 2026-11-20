"""
Realistic Amazon-Style Product Review Dataset Generator.
Supports 16-Field Schema, 5 Categories, 3 Dataset Tiers, and Multi-Format Output.
Complies with Milestone 1 and Interface Contract 1.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config import (
    CATEGORIES,
    DEFAULT_SEED,
    FIELD_TYPES,
    RAW_DATA_DIR,
    RAW_MEDIUM_CSV,
    RAW_SAMPLE_CSV,
    RAW_TIER_A_CSV,
    RAW_TIER_B_CSV,
    RAW_TIER_C_CSV,
    SCHEMA_FIELDS,
    SENTIMENT_MAP,
    TIER_COUNTS,
    ensure_directories,
)

# Alias for backwards and cross-module compatibility
PRODUCT_CATEGORIES = CATEGORIES
REVIEW_SCHEMA = SCHEMA_FIELDS

# ---------------------------------------------------------------------------
# Catalog Definitions: 8 Distinct Products per Category (40 Products Total)
# ---------------------------------------------------------------------------
PRODUCTS: Dict[str, List[Tuple[str, str]]] = {
    "Electronics": [
        ("PROD_ELEC_001", "Noise-Cancelling Wireless Headphones"),
        ("PROD_ELEC_002", "Ultra HD 4K Smart OLED Television"),
        ("PROD_ELEC_003", "Ergonomic Mechanical Gaming Keyboard"),
        ("PROD_ELEC_004", "High-Speed Dual-Band Wi-Fi 6 Router"),
        ("PROD_ELEC_005", "Portable Bluetooth Waterproof Speaker"),
        ("PROD_ELEC_006", "USB-C Multi-Port Hub Adapter"),
        ("PROD_ELEC_007", "Fast Wireless Charging Stand"),
        ("PROD_ELEC_008", "1080p Streaming Webcam with Microphone"),
    ],
    "Books": [
        ("PROD_BOOK_001", "Designing Data-Intensive Applications"),
        ("PROD_BOOK_002", "Learning Spark: Lightning-Fast Big Data Analysis"),
        ("PROD_BOOK_003", "Clean Code: A Handbook of Agile Software Craftsmanship"),
        ("PROD_BOOK_004", "Introduction to Algorithms, Fourth Edition"),
        ("PROD_BOOK_005", "Artificial Intelligence: A Modern Approach"),
        ("PROD_BOOK_006", "Hands-On Machine Learning with Scikit-Learn and PyTorch"),
        ("PROD_BOOK_007", "Site Reliability Engineering: How Google Runs Production Systems"),
        ("PROD_BOOK_008", "Deep Learning with Python"),
    ],
    "Home & Kitchen": [
        ("PROD_HOME_001", "Programmable Multi-Cooker Instant Pot"),
        ("PROD_HOME_002", "Countertop Espresso and Coffee Machine"),
        ("PROD_HOME_003", "Cast Iron Dutch Oven with Enameled Finish"),
        ("PROD_HOME_004", "Cordless Stick Vacuum Cleaner"),
        ("PROD_HOME_005", "Air Fryer XL 8-in-1 Compact Appliance"),
        ("PROD_HOME_006", "Stainless Steel Chef Knife 8-Inch"),
        ("PROD_HOME_007", "Non-Stick Ceramic Cookware 10-Piece Set"),
        ("PROD_HOME_008", "Smart Temperature Control Heated Mug"),
    ],
    "Beauty & Personal Care": [
        ("PROD_BEAU_001", "Hydrating Hyaluronic Acid Facial Serum"),
        ("PROD_BEAU_002", "Sonic Electric Toothbrush with Smart Timer"),
        ("PROD_BEAU_003", "Organic Argan Oil Daily Hair Conditioner"),
        ("PROD_BEAU_004", "Broad Spectrum SPF 50 Mineral Sunscreen"),
        ("PROD_BEAU_005", "Exfoliating Gentle Salicylic Acid Cleanser"),
        ("PROD_BEAU_006", "Vitamin C Brightening Skin Glow Oil"),
        ("PROD_BEAU_007", "Deep Moisturizing Shea Butter Body Lotion"),
        ("PROD_BEAU_008", "Professional Ionic Salon Hair Dryer"),
    ],
    "Sports & Outdoors": [
        ("PROD_SPRT_001", "Insulated Stainless Steel Sports Water Bottle"),
        ("PROD_SPRT_002", "High-Density Non-Slip Yoga Mat with Strap"),
        ("PROD_SPRT_003", "Adjustable Quick-Select Dumbbells Pair"),
        ("PROD_SPRT_004", "Lightweight 2-Person Backpacking Tent"),
        ("PROD_SPRT_005", "Trekking Poles Carbon Fiber Hiking Set"),
        ("PROD_SPRT_006", "Breathable Compression Running Socks"),
        ("PROD_SPRT_007", "Polarized UV400 Athletic Sunglasses"),
        ("PROD_SPRT_008", "Heavy-Duty Resistance Exercise Bands Kit"),
    ],
}

# ---------------------------------------------------------------------------
# Reviewer Pool Generation (Supports Bipartite Network Modeling in Unit 6)
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Alex", "David", "Sarah", "Michael", "Emily", "James", "Jessica", "John",
    "Emma", "Robert", "Olivia", "Daniel", "Sophia", "Matthew", "Ava", "Joseph",
    "Isabella", "William", "Mia", "Anthony", "Charlotte", "Ethan", "Amelia",
    "Benjamin", "Harper", "Lucas", "Evelyn", "Henry", "Abigail", "Alexander",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
]

def build_reviewer_pool(rng: random.Random, pool_size: int = 250) -> List[Tuple[str, str]]:
    """Build a deterministic pool of reviewers with fixed IDs and names."""
    pool = []
    for i in range(1, pool_size + 1):
        fn = rng.choice(FIRST_NAMES)
        ln = rng.choice(LAST_NAMES)
        pool.append((f"USR_{i:05d}", f"{fn} {ln}"))
    return pool

# ---------------------------------------------------------------------------
# Category-Specific Context Phrases
# ---------------------------------------------------------------------------
CATEGORY_CLAUSES = {
    "Electronics": [
        "The battery easily lasts through full workdays and pairing took under 10 seconds.",
        "Audio clarity and responsiveness are top-tier with zero latency issues.",
        "The firmware update was smooth and connectivity remains rock solid.",
        "Display brightness and build materials feel completely premium.",
    ],
    "Books": [
        "The architectural patterns and technical case studies are explained with remarkable clarity.",
        "The author's clear explanations and practical code examples make complex topics intuitive.",
        "Every chapter is well-structured with insightful diagrams and deep technical takeaways.",
        "A truly comprehensive resource that will remain on my reference shelf for years.",
    ],
    "Home & Kitchen": [
        "Cleans up effortlessly and fits cleanly on standard kitchen countertops.",
        "Heat distribution is remarkably even, cutting down preparation time significantly.",
        "Stainless steel finish looks stylish and withstands daily heavy kitchen use.",
        "Setup was straightforward and it operates quietly throughout use.",
    ],
    "Beauty & Personal Care": [
        "Gentle on sensitive skin without causing any irritation or oily residue.",
        "Lightweight formula absorbs rapidly and leaves skin feeling fresh and hydrated.",
        "Noticeable positive improvement within two weeks of consistent daily application.",
        "Subtle natural fragrance and dermatologist-approved soothing ingredients.",
    ],
    "Sports & Outdoors": [
        "Held up firmly during intense outdoor trail sessions and wet weather conditions.",
        "Ergonomic grip provides superior comfort and prevents fatigue during workouts.",
        "Tough weatherproof construction that easily survives rugged mountain terrain.",
        "Lightweight enough for long distance backpacking yet impressively durable.",
    ],
}

# ---------------------------------------------------------------------------
# Rating-Specific Summary & Text Templates (Ratings 1.0 to 5.0)
# ---------------------------------------------------------------------------
REVIEW_TEMPLATES = {
    5.0: {
        "summaries": [
            "Absolutely fantastic purchase!",
            "Exceeded all my expectations",
            "Best in its class, highly recommended",
            "Superb quality and great design",
            "Five stars without hesitation!",
            "Flawless performance and durable build",
            "Exceptional product, worth every penny",
            "Completely satisfied with this purchase",
        ],
        "texts": [
            "I am absolutely thrilled with this {product_name}! The build quality is exceptional and it exceeded all my expectations. {category_clause} Definitely worth every penny and highly recommended!",
            "An absolute game changer. From the moment I unpacked this {product_name}, I could tell it was engineered with attention to detail. It performs flawlessly and feels exceptionally durable. Five stars all the way!",
            "This {product_name} is by far the best purchase I have made this year. Setup was painless, and operation is smooth and reliable. {category_clause} Do not hesitate to buy this.",
            "Outstanding value and top-tier craftsmanship. The {product_name} works like a dream. Everything feels high quality and well thought out. Would give six stars if possible!",
        ],
    },
    4.0: {
        "summaries": [
            "Very good overall with minor quirks",
            "Great value for the price point",
            "Solid choice, does what it promises",
            "Pleasantly surprised by the performance",
            "Almost perfect, minor room for improvement",
            "Dependable and well made",
        ],
        "texts": [
            "Really impressed with the {product_name}. The performance is solid and reliable. Only giving 4 stars because the manual was slightly sparse, but {category_clause} Overall a very strong product.",
            "Great value for money. The {product_name} handles daily usage easily and the build quality is admirable. Minor room for improvement in packaging, but the item itself works great.",
            "A dependable choice! The {product_name} looks sleek and does exactly what it is advertised to do. {category_clause} Would certainly buy again.",
            "High quality construction and dependable features. A few small ergonomics could be tweaked, but the {product_name} functions reliably day after day.",
        ],
    },
    3.0: {
        "summaries": [
            "Decent product, average performance",
            "Mixed feelings about this purchase",
            "Acceptable but nothing extraordinary",
            "Meets basic expectations, neither great nor terrible",
            "Okay for the price, has some tradeoffs",
            "Mediocre experience overall",
        ],
        "texts": [
            "The {product_name} is okay, but nothing to write home about. It works for basic needs, though the materials feel rather average. It gets the job done if you have modest expectations.",
            "Mixed feelings regarding this {product_name}. On one hand the styling is fine, but on the other hand, performance is merely acceptable. Not bad, but definitely room for refinement.",
            "Mediocre experience overall. The {product_name} is decent for occasional use, but serious buyers might want to invest in higher tier alternatives.",
            "It functions as described, but lacks the polish seen in higher-end products. Decent budget option, but do not expect premium quality.",
        ],
    },
    2.0: {
        "summaries": [
            "Disappointing quality, would not repurchase",
            "Underwhelming experience with multiple flaws",
            "Fell well short of expectations",
            "Frequent issues and poor reliability",
            "Not worth the price tag",
            "Subpar craftsmanship",
        ],
        "texts": [
            "I was hoping for much better performance from this {product_name}. The build feels flimsy and frequent hiccups caused repeated frustration. Cannot recommend at this price point.",
            "Rather disappointed with the {product_name}. After just a few uses, issues started appearing with consistency. Customer service was slow to respond as well.",
            "Below average. The {product_name} failed to deliver on several core promises. The construction feels fragile and low-grade.",
            "Fell well short of what was advertised. The {product_name} feels cheap and struggles under regular use. Would look elsewhere.",
        ],
    },
    1.0: {
        "summaries": [
            "Completely defective and terrible",
            "Waste of money, do not buy!",
            "Horrible experience, returning immediately",
            "Broken right out of the box",
            "Utter disappointment, avoid at all costs",
            "Complete failure, zero stars if possible",
        ],
        "texts": [
            "Total disaster. The {product_name} arrived defective and completely stopped functioning on day two. The cheap materials made it totally unusable. Returning immediately for a refund!",
            "Terrible quality! Do not waste your hard-earned money on this {product_name}. The materials are cheap and flimsy, and it broke almost immediately. Worst purchase ever.",
            "Completely unacceptable. I experienced major failures right out of the box with this {product_name}. Avoid this brand at all costs!",
            "Defective unit and atrocious build quality. The {product_name} stopped working within hours. Do not buy under any circumstances!",
        ],
    },
}

# ---------------------------------------------------------------------------
# Core Record Generator Function
# ---------------------------------------------------------------------------
def generate_review_records(num_records: int, seed: int = DEFAULT_SEED) -> List[Dict[str, Any]]:
    """
    Generate an in-memory list of review records adhering to the 16-field Amazon schema.
    
    Args:
        num_records: Number of review records to generate.
        seed: Random seed for deterministic reproducibility.
        
    Returns:
        List of dictionaries with exactly 16 fields.
    """
    if num_records <= 0:
        raise ValueError(f"num_records must be positive, got {num_records}")

    rng = random.Random(seed)

    # Scale reviewer pool with record count
    pool_size = max(30, min(1000, num_records // 5))
    reviewer_pool = build_reviewer_pool(rng, pool_size=pool_size)
    num_power = max(3, int(pool_size * 0.10))
    power_reviewers = reviewer_pool[:num_power]
    normal_reviewers = reviewer_pool[num_power:]

    # Realistic rating distribution: 65% positive, 15% neutral, 20% negative
    rating_options = [5.0, 4.0, 3.0, 2.0, 1.0]
    rating_weights = [0.40, 0.25, 0.15, 0.10, 0.10]

    # Date range: 2023-01-01 to 2026-10-31
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2026, 10, 31)
    delta_days = (end_date - start_date).days

    records: List[Dict[str, Any]] = []

    for idx in range(1, num_records + 1):
        # 1. Review ID
        review_id = f"REV_{idx:08d}"

        # 2. Reviewer (35% power reviewer, 65% normal reviewer for rich graph connectivity)
        if rng.random() < 0.35 and power_reviewers:
            usr_id, usr_name = rng.choice(power_reviewers)
        else:
            usr_id, usr_name = rng.choice(normal_reviewers)

        # 3. Category & Product
        category = rng.choice(CATEGORIES)
        prod_id, prod_name = rng.choice(PRODUCTS[category])

        # 4. Rating & Sentiment
        rating = rng.choices(rating_options, weights=rating_weights)[0]
        sentiment_label, sentiment_numeric = SENTIMENT_MAP[rating]

        # 5. Verified Purchase (~85% True)
        verified_purchase = rng.random() < 0.85

        # 6. Review Summary & Text
        template_group = REVIEW_TEMPLATES[rating]
        summary = rng.choice(template_group["summaries"])
        text_template = rng.choice(template_group["texts"])
        cat_clause = rng.choice(CATEGORY_CLAUSES[category])
        review_text = text_template.format(
            product_name=prod_name,
            category_clause=cat_clause,
        )

        # 7. Temporal fields
        offset_days = rng.randint(0, delta_days)
        d = start_date + datetime.timedelta(days=offset_days)
        seconds_offset = rng.randint(0, 86399)
        dt = datetime.datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=seconds_offset)
        timestamp = int(dt.timestamp())
        review_date = d.isoformat()

        # 8. Votes (Invariant: helpful_votes <= total_votes)
        r_vote = rng.random()
        if r_vote < 0.60:
            total_votes = 0
            helpful_votes = 0
        elif r_vote < 0.90:
            total_votes = rng.randint(1, 10)
            helpful_votes = rng.randint(0, total_votes)
        else:
            total_votes = rng.randint(11, 60)
            helpful_votes = rng.randint(int(total_votes * 0.5), total_votes)

        record = {
            "review_id": review_id,
            "reviewer_id": usr_id,
            "reviewer_name": usr_name,
            "product_id": prod_id,
            "product_name": prod_name,
            "product_category": category,
            "rating": float(rating),
            "verified_purchase": bool(verified_purchase),
            "review_summary": summary,
            "review_text": review_text,
            "review_date": review_date,
            "timestamp": timestamp,
            "helpful_votes": int(helpful_votes),
            "total_votes": int(total_votes),
            "sentiment_label": sentiment_label,
            "sentiment_numeric": float(sentiment_numeric),
        }
        records.append(record)

    return records

# ---------------------------------------------------------------------------
# Schema Validation Routine
# ---------------------------------------------------------------------------
def validate_record_schema(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a record against the 16-field Amazon schema and integrity invariants.
    
    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    # Check missing fields
    for field in SCHEMA_FIELDS:
        if field not in record:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return False, errors

    # Check types and invariants
    if not isinstance(record["review_id"], str) or not record["review_id"].startswith("REV_"):
        errors.append(f"Invalid review_id format: {record.get('review_id')}")

    if not isinstance(record["rating"], (float, int)) or float(record["rating"]) not in [1.0, 2.0, 3.0, 4.0, 5.0]:
        errors.append(f"Invalid rating value: {record.get('rating')}")

    if not isinstance(record["verified_purchase"], bool):
        errors.append(f"verified_purchase must be bool, got {type(record.get('verified_purchase'))}")

    if record["product_category"] not in CATEGORIES:
        errors.append(f"Invalid product_category: {record.get('product_category')}")

    # Vote invariant
    if record["helpful_votes"] > record["total_votes"] or record["helpful_votes"] < 0:
        errors.append(f"Vote invariant violation: helpful_votes ({record['helpful_votes']}) > total_votes ({record['total_votes']})")

    # Sentiment mapping invariant
    expected_sentiment, expected_numeric = SENTIMENT_MAP.get(float(record["rating"]), ("", -1.0))
    if record["sentiment_label"] != expected_sentiment:
        errors.append(f"Sentiment label mismatch: expected '{expected_sentiment}' for rating {record['rating']}, got '{record['sentiment_label']}'")
    if float(record["sentiment_numeric"]) != float(expected_numeric):
        errors.append(f"Sentiment numeric mismatch: expected {expected_numeric} for rating {record['rating']}, got {record['sentiment_numeric']}")

    return len(errors) == 0, errors

# ---------------------------------------------------------------------------
# ReviewDatasetGenerator Class
# ---------------------------------------------------------------------------
class ReviewDatasetGenerator:
    """
    Class interface for generating synthetic Amazon-style review datasets.
    Provides deterministic batch generation and multi-format serialization.
    """

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = seed

    def generate(self, num_records: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate in-memory records."""
        effective_seed = seed if seed is not None else self.seed
        return generate_review_records(num_records=num_records, seed=effective_seed)

    def generate_to_file(
        self,
        num_records: int,
        seed: Optional[int] = None,
        output_format: str = "csv",
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate reviews and write to file."""
        effective_seed = seed if seed is not None else self.seed
        return generate_reviews(
            num_records=num_records,
            seed=effective_seed,
            output_format=output_format,
            output_path=output_path,
        )

    def generate_tier(
        self,
        tier: str = "b",
        seed: Optional[int] = None,
        output_format: str = "csv",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate a standard tier dataset."""
        effective_seed = seed if seed is not None else self.seed
        return generate_tier_dataset(
            tier=tier,
            seed=effective_seed,
            output_format=output_format,
            output_dir=output_dir,
        )

# Alias
ReviewGenerator = ReviewDatasetGenerator

# ---------------------------------------------------------------------------
# Public Dataset Generator Functions
# ---------------------------------------------------------------------------
def generate_reviews(
    num_records: int,
    seed: int = DEFAULT_SEED,
    output_format: str = "csv",
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Generate synthetic reviews and serialize them to disk in the requested format.
    
    Args:
        num_records: Number of review records.
        seed: Random seed for reproducibility.
        output_format: 'csv', 'json', or 'parquet'.
        output_path: Optional custom path. If None, saves to data/raw/reviews.<ext>.
        
    Returns:
        Path to the generated file.
    """
    ensure_directories()
    records = generate_review_records(num_records=num_records, seed=seed)

    output_format = output_format.lower().strip()
    if output_format not in ["csv", "json", "parquet"]:
        raise ValueError(f"Unsupported output_format: '{output_format}'. Choose from 'csv', 'json', 'parquet'.")

    if output_path is None:
        target_path = RAW_DATA_DIR / f"reviews.{output_format}"
    else:
        target_path = Path(output_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "csv":
        with open(target_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(records)

    elif output_format == "json":
        with open(target_path, mode="w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    elif output_format == "parquet":
        try:
            import pandas as pd
            df = pd.DataFrame(records)
            df.to_parquet(target_path, index=False)
        except ImportError as e:
            raise ImportError(
                "Exporting to Parquet requires 'pyarrow' or 'fastparquet'. "
                "Install via 'pip install pyarrow' or use output_format='csv'."
            ) from e

    return target_path

def generate_tier_dataset(
    tier: str = "b",
    seed: int = DEFAULT_SEED,
    output_format: str = "csv",
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Generate a standard tiered dataset.
    Supports tier aliases: 'a'/'sample' (100), 'b'/'medium' (5,000), 'c'/'benchmark' (25,000).
    
    Args:
        tier: 'a', 'b', 'c', 'sample', 'medium', or 'benchmark'.
        seed: Random seed.
        output_format: 'csv', 'json', or 'parquet'.
        output_dir: Target directory (default: data/raw).
        
    Returns:
        Path to generated dataset file.
    """
    tier_clean = tier.lower().strip()
    if tier_clean not in TIER_COUNTS:
        raise ValueError(f"Unknown tier: '{tier}'. Choose from 'a' ('sample'), 'b' ('medium'), 'c' ('benchmark').")

    count = TIER_COUNTS[tier_clean]
    base_dir = Path(output_dir) if output_dir else RAW_DATA_DIR

    # Canonical tier letter
    norm_tier = "a" if tier_clean in ("a", "sample") else ("b" if tier_clean in ("b", "medium") else "c")
    target_path = base_dir / f"reviews_tier_{norm_tier}.{output_format}"

    res = generate_reviews(
        num_records=count,
        seed=seed,
        output_format=output_format,
        output_path=target_path,
    )

    # Also create sample/medium alias copies if saving to RAW_DATA_DIR
    if base_dir == RAW_DATA_DIR:
        if norm_tier == "a":
            alias_path = base_dir / f"reviews_sample.{output_format}"
            shutil.copy2(target_path, alias_path)
        elif norm_tier == "b":
            alias_path = base_dir / f"reviews_medium.{output_format}"
            shutil.copy2(target_path, alias_path)

    return res

# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI runner for generating review datasets."""
    parser = argparse.ArgumentParser(description="Realistic Amazon Review Dataset Generator")
    parser.add_argument(
        "--tier",
        choices=["a", "b", "c", "sample", "medium", "benchmark"],
        default="b",
        help="Dataset tier: a/sample (100), b/medium (5,000), c/benchmark (25,000)",
    )
    parser.add_argument("--num-records", type=int, default=None, help="Custom record count (overrides --tier)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--format", choices=["csv", "json", "parquet"], default="csv", help="Output file format (default: csv)")
    parser.add_argument("--output-dir", type=str, default=str(RAW_DATA_DIR), help="Output directory")
    parser.add_argument("--all-tiers", action="store_true", help="Generate all three tiers (A, B, and C) simultaneously")

    args = parser.parse_args()
    ensure_directories()

    if args.all_tiers:
        print(f"Generating all three tiers with seed={args.seed}, format={args.format} into {args.output_dir}...")
        for t in ["a", "b", "c"]:
            p = generate_tier_dataset(tier=t, seed=args.seed, output_format=args.format, output_dir=args.output_dir)
            print(f"  [+] Tier {t.upper()} generated ({TIER_COUNTS[t]:,} rows): {p}")
        print("Done!")
    elif args.num_records is not None:
        target_file = Path(args.output_dir) / f"reviews_custom_{args.num_records}.{args.format}"
        p = generate_reviews(num_records=args.num_records, seed=args.seed, output_format=args.format, output_path=target_file)
        print(f"[+] Custom dataset generated ({args.num_records:,} rows): {p}")
    else:
        p = generate_tier_dataset(tier=args.tier, seed=args.seed, output_format=args.format, output_dir=args.output_dir)
        print(f"[+] Tier {args.tier.upper()} dataset generated ({TIER_COUNTS[args.tier]:,} rows): {p}")

if __name__ == "__main__":
    main()
