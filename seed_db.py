"""
seed_db.py — Seeds the price database from scraped/NBS data files.

Takes the output of jiji_scraper.py or nbs_price_fetcher.py and
calls your DB interface's submit_price() for every row.

Your backend teammate imports their real DBInterface here.

Usage:
    # Seed from NBS baseline
    python seed_db.py --source nbs_baseline_prices.csv

    # Seed from Jiji scrape
    python seed_db.py --source jiji_prices.csv

    # Seed from both
    python seed_db.py --source nbs_baseline_prices.csv jiji_prices.csv

    # Dry run (validate without writing)
    python seed_db.py --source nbs_baseline_prices.csv --dry-run
"""

import argparse
import logging
from datetime import datetime

import pandas as pd

# ── Swap this import for the real DB interface once backend is ready ──────────
# from your_backend.db import RealDBInterface as DBInterface
from db_interface import MockDBInterface as DBInterface
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seed data is submitted under a system user ID
SYSTEM_USER_ID = "system_seed_v1"

# High reputation weight for seeded data (we trust NBS/Jiji more than new users)
SEED_REPUTATION_SCORE = 0.85


def validate_row(row: pd.Series) -> tuple[bool, str]:
    """Check a row has the minimum fields needed to seed."""
    if pd.isna(row.get("product_canonical")) or not str(row["product_canonical"]).strip():
        return False, "missing product_canonical"
    if pd.isna(row.get("price_ngn")) or float(row["price_ngn"]) <= 0:
        return False, "invalid price"
    if pd.isna(row.get("location_city")) or not str(row["location_city"]).strip():
        return False, "missing location"
    return True, ""


def seed(sources: list[str], dry_run: bool = False):
    db = DBInterface()
    total_attempted = 0
    total_success = 0
    total_skipped = 0
    total_failed = 0

    for source_path in sources:
        logger.info(f"\nLoading: {source_path}")
        try:
            df = pd.read_csv(source_path)
        except FileNotFoundError:
            logger.error(f"File not found: {source_path}")
            continue

        logger.info(f"  {len(df)} rows loaded")

        # Drop flagged outlier rows if column exists
        if "review_flag" in df.columns:
            flagged = df["review_flag"].sum()
            df = df[~df["review_flag"]]
            logger.info(f"  Dropped {flagged} flagged outlier rows")

        for _, row in df.iterrows():
            total_attempted += 1

            valid, reason = validate_row(row)
            if not valid:
                logger.debug(f"  Skipping row: {reason}")
                total_skipped += 1
                continue

            product = str(row["product_canonical"]).strip()
            location = str(row.get("market_area") or row.get("location_city", "")).strip()
            unit = str(row.get("unit_canonical", "per unit")).strip()
            price = float(row["price_ngn"])
            listing_date = str(row.get("listing_date", datetime.now().isoformat()))
            source = str(row.get("source", "seed"))

            if dry_run:
                logger.info(
                    f"  [DRY RUN] Would submit: {product} @ ₦{price:,.0f} {unit} "
                    f"at {location} ({source})"
                )
                total_success += 1
                continue

            try:
                success = db.submit_price(
                    product=product,
                    location=location,
                    unit=unit,
                    price=price,
                    user_id=SYSTEM_USER_ID,
                    timestamp=listing_date,
                )
                if success:
                    total_success += 1
                else:
                    total_failed += 1
                    logger.warning(f"  DB rejected: {product} @ ₦{price} at {location}")
            except Exception as e:
                total_failed += 1
                logger.warning(f"  Exception submitting {product}: {e}")

    # Summary
    logger.info(f"""
╔══════════════════════════════════╗
║         Seeding Complete         ║
╠══════════════════════════════════╣
║  Attempted : {total_attempted:<5}                 ║
║  Success   : {total_success:<5}                 ║
║  Skipped   : {total_skipped:<5} (bad data)       ║
║  Failed    : {total_failed:<5} (DB errors)       ║
╚══════════════════════════════════╝
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed price DB from scraped/NBS data")
    parser.add_argument(
        "--source", nargs="+", required=True,
        help="CSV file(s) to seed from (jiji_prices.csv, nbs_baseline_prices.csv)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and log without writing to DB"
    )
    args = parser.parse_args()
    seed(args.source, dry_run=args.dry_run)