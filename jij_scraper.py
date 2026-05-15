"""
jiji_scraper.py — Scrapes food/grocery prices from Jiji.ng

Run this on YOUR LOCAL MACHINE (needs internet access).

Usage:
    pip install requests beautifulsoup4 pandas
    python jiji_scraper.py
    # outputs: jiji_prices.csv

What it scrapes:
    - Product name
    - Price (NGN)
    - Location (state + city)
    - Listing date
    - Unit/quantity description

Then import into your DB:
    python seed_db.py --source jiji_prices.csv
"""

import time
import random
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import re

import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_URL = "https://jiji.ng"

# Jiji category slugs for food/grocery items
SEARCH_QUERIES = [
    "tomato",
    "pepper",
    "onion",
    "yam",
    "cassava",
    "garri",
    "rice",
    "beans",
    "palm oil",
    "groundnut oil",
    "plantain",
    "banana",
    "crayfish",
    "stockfish",
    "smoked fish",
    "goat meat",
    "chicken",
    "eggs",
    "beef",
    "maize",
    "millet",
    "okra",
    "sweet potato",
    "cocoyam",
    "egusi",
    "groundnut",
    "coconut",
]

# Target Nigerian markets/cities
TARGET_LOCATIONS = [
    "lagos",
    "abuja",
    "kano",
    "ibadan",
    "port-harcourt",
    "onitsha",
    "aba",
    "kaduna",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Polite delay between requests (seconds) — don't hammer the server
MIN_DELAY = 2.0
MAX_DELAY = 4.5

MAX_PAGES_PER_QUERY = 3  # 3 pages × ~30 listings = ~90 results per product


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class PriceListing:
    product_raw: str
    product_canonical: str
    price_ngn: float
    unit_raw: str
    unit_canonical: str
    location_city: str
    location_state: str
    market_area: Optional[str]
    listing_date: str
    source: str = "jiji.ng"
    source_url: str = ""
    confidence: str = "medium"


# ─── Canonical product mapping ────────────────────────────────────────────────

PRODUCT_CANONICAL = {
    "tomato": "tomato", "tomatoes": "tomato", "tomatoe": "tomato",
    "pepper": "pepper", "red pepper": "pepper", "tatashe": "pepper",
    "scotch bonnet": "scotch bonnet", "atarodo": "scotch bonnet", "habanero": "scotch bonnet",
    "onion": "onion", "onions": "onion",
    "yam": "yam", "yams": "yam",
    "cassava": "cassava",
    "garri": "garri", "gari": "garri",
    "rice": "rice", "local rice": "rice", "foreign rice": "rice", "basmati": "rice",
    "beans": "beans", "black eyed beans": "beans", "brown beans": "beans",
    "palm oil": "palm oil",
    "groundnut oil": "groundnut oil", "peanut oil": "groundnut oil",
    "plantain": "plantain", "plantains": "plantain",
    "banana": "banana", "bananas": "banana",
    "crayfish": "crayfish",
    "stockfish": "stockfish", "panla": "stockfish", "okporoko": "stockfish",
    "smoked fish": "smoked fish", "titus": "smoked fish", "kote": "smoked fish",
    "goat meat": "goat meat", "chevon": "goat meat",
    "chicken": "chicken", "broiler": "chicken",
    "eggs": "eggs", "egg": "eggs",
    "beef": "beef", "cow meat": "beef",
    "maize": "maize", "corn": "maize",
    "millet": "millet", "gero": "millet",
    "okra": "okra", "okro": "okra",
    "sweet potato": "sweet potato",
    "cocoyam": "cocoyam",
    "egusi": "egusi", "melon": "egusi",
    "groundnut": "groundnut", "peanut": "groundnut",
    "coconut": "coconut",
}

UNIT_CANONICAL = {
    "kg": "per kg", "kilogram": "per kg", "kilograms": "per kg",
    "g": "per 100g", "gram": "per 100g", "grams": "per 100g",
    "bag": "per bag", "bags": "per bag", "50kg bag": "per 50kg bag",
    "basket": "per basket", "paint": "per paint bucket",
    "dozen": "per dozen", "crate": "per crate",
    "piece": "per piece", "pieces": "per piece",
    "litre": "per litre", "liter": "per litre", "litres": "per litre",
    "bottle": "per bottle",
    "bunch": "per bunch",
    "tuber": "per tuber",
}


def canonicalize_product(raw: str) -> str:
    raw_lower = raw.lower().strip()
    for key, canonical in PRODUCT_CANONICAL.items():
        if key in raw_lower:
            return canonical
    return raw_lower


def extract_unit(text: str) -> tuple[str, str]:
    """Extract unit from listing title/description. Returns (raw, canonical)."""
    text_lower = text.lower()
    for unit_key, canonical in UNIT_CANONICAL.items():
        if unit_key in text_lower:
            return unit_key, canonical
    return "unit", "per unit"


def extract_price(price_text: str) -> Optional[float]:
    """Extract numeric price from Jiji price string like '₦1,200' or 'NGN 800'."""
    cleaned = re.sub(r"[₦NGn,\s]", "", price_text)
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


# ─── Scraper ──────────────────────────────────────────────────────────────────

class JijiScraper:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.results: list[PriceListing] = []

    def _polite_sleep(self):
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)

    def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def scrape_product(self, query: str, location: str):
        logger.info(f"Scraping: {query} in {location}")

        for page in range(1, MAX_PAGES_PER_QUERY + 1):
            url = f"{BASE_URL}/{location}/food-and-agriculture?search%5Bkeywords%5D={query}&page={page}"

            soup = self._fetch(url)
            if not soup:
                break

            # Jiji listing cards
            listings = soup.select("article.b-list-advert-base")
            if not listings:
                # Try alternate selector
                listings = soup.select("div.qa-advert-list-item")

            if not listings:
                logger.info(f"  No listings on page {page} — stopping")
                break

            page_count = 0
            for listing in listings:
                result = self._parse_listing(listing, query, location)
                if result:
                    self.results.append(result)
                    page_count += 1

            logger.info(f"  Page {page}: {page_count} listings extracted")
            self._polite_sleep()

    def _parse_listing(self, listing, query: str, location: str) -> Optional[PriceListing]:
        try:
            # Title
            title_el = listing.select_one("div.b-advert-title-inner, h3.qa-advert-title")
            title = title_el.get_text(strip=True) if title_el else ""

            # Price
            price_el = listing.select_one("span.qa-advert-price, div.b-advert-price")
            if not price_el:
                return None
            price = extract_price(price_el.get_text(strip=True))
            if not price or price <= 0:
                return None

            # Location
            location_el = listing.select_one("span.b-list-advert__region__text, div.qa-advert-location")
            location_text = location_el.get_text(strip=True) if location_el else location

            # Parse city/state from location text (usually "Mushin, Lagos State")
            parts = [p.strip() for p in location_text.split(",")]
            city = parts[0] if parts else location
            state = parts[1].replace("State", "").strip() if len(parts) > 1 else location.title()

            # URL
            link_el = listing.select_one("a")
            listing_url = BASE_URL + link_el["href"] if link_el and link_el.get("href") else ""

            # Date
            date_el = listing.select_one("span.b-list-advert-base__item-date, time")
            listing_date = date_el.get_text(strip=True) if date_el else datetime.now().strftime("%Y-%m-%d")

            unit_raw, unit_canonical = extract_unit(title)

            return PriceListing(
                product_raw=title[:80],
                product_canonical=canonicalize_product(query),
                price_ngn=price,
                unit_raw=unit_raw,
                unit_canonical=unit_canonical,
                location_city=city,
                location_state=state,
                market_area=self._infer_market(city),
                listing_date=listing_date,
                source_url=listing_url,
            )

        except Exception as e:
            logger.debug(f"Failed to parse listing: {e}")
            return None

    def _infer_market(self, city: str) -> Optional[str]:
        """Map city names to known Nigerian market areas."""
        market_map = {
            "mile 12": "Mile 12 Market",
            "mile12": "Mile 12 Market",
            "oyingbo": "Oyingbo Market",
            "mushin": "Mushin Market",
            "tejuosho": "Tejuosho Market",
            "balogun": "Balogun Market",
            "idumota": "Idumota Market",
            "bodija": "Bodija Market",
            "dugbe": "Dugbe Market",
            "wuse": "Wuse Market",
            "garki": "Garki Market",
            "ariaria": "Ariaria Market",
            "onitsha": "Onitsha Main Market",
            "kasuwan": "Kasuwan Barci",
            "sabon gari": "Sabon Gari Market",
        }
        city_lower = city.lower()
        for key, market in market_map.items():
            if key in city_lower:
                return market
        return None

    def run(self):
        for location in TARGET_LOCATIONS:
            for query in SEARCH_QUERIES:
                self.scrape_product(query, location)
                self._polite_sleep()

        logger.info(f"\nTotal listings scraped: {len(self.results)}")
        return self.results

    def to_csv(self, path: str = "jiji_prices.csv"):
        if not self.results:
            logger.warning("No results to save.")
            return

        df = pd.DataFrame([asdict(r) for r in self.results])

        # Basic cleaning
        df = df.drop_duplicates(subset=["product_canonical", "price_ngn", "location_city", "listing_date"])
        df = df[df["price_ngn"] > 0]

        # Flag suspiciously low or high prices for manual review
        df["review_flag"] = False
        for product in df["product_canonical"].unique():
            mask = df["product_canonical"] == product
            q1 = df.loc[mask, "price_ngn"].quantile(0.05)
            q3 = df.loc[mask, "price_ngn"].quantile(0.95)
            df.loc[mask & ((df["price_ngn"] < q1) | (df["price_ngn"] > q3)), "review_flag"] = True

        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} listings to {path}")
        logger.info(f"Flagged {df['review_flag'].sum()} rows for manual review")


if __name__ == "__main__":
    scraper = JijiScraper()
    scraper.run()
    scraper.to_csv("jiji_prices.csv")
    print("\nDone. Next step: python seed_db.py --source jiji_prices.csv")