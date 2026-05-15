from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


# ── Shared data classes ───────────────────────────────────────────────────────

@dataclass
class PriceEntry:
    """A single price submission retrieved from the database."""
    price: float
    unit: str
    timestamp: str
    user_id: str
    reputation_score: float
    location: str
    product: str


@dataclass
class PriceSubmission:
    """A validated price submission to be written to the database."""
    product: str
    location: str
    unit: str
    price: float
    user_id: str
    timestamp: str


# ── Abstract interface ────────────────────────────────────────────────────────

class BaseDBInterface(ABC):

    @abstractmethod
    def get_prices(
        self,
        product: str,
        location: str = "",
        days: int = 7,
    ) -> List[PriceEntry]:
        """
        Retrieve price submissions for a product within N days.

        If location is provided, return prices for product + location.
        If location is empty, return prices for product across all locations.
        """
        pass

    @abstractmethod
    def submit_price(self, submission: PriceSubmission) -> bool:
        """
        Write a validated price submission to the database.
        """
        pass

    @abstractmethod
    def get_user_reputation(self, user_id: str) -> float:
        """
        Get the reputation score for a user.
        """
        pass


# ── Mock implementation for development / testing ─────────────────────────────

class MockDBInterface(BaseDBInterface):
    """
    In-memory mock DB.

    Notes:
    - Verified seed data uses higher reputation scores.
    - Demo seed data uses lower reputation scores.
    - Replace this with the real backend DB implementation before production.
    """

    def __init__(self):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)

        self._prices = [
            # ─────────────────────────────────────────────────────
            # Verified seed prices: Lagos State Ministry of Agriculture
            # Source: Lagos Food Price Tracker, February 2026
            # Location is "Lagos Average" because this is not market-specific.
            # ─────────────────────────────────────────────────────

            PriceEntry(
                price=30500.0,
                unit="25kg bag",
                product="rice",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=27388.89,
                unit="25kg bag",
                product="rice",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=61000.0,
                unit="50kg bag",
                product="rice",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=54444.44,
                unit="50kg bag",
                product="rice",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),

            PriceEntry(
                price=1480.0,
                unit="paint bucket",
                product="garri",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=1422.22,
                unit="paint bucket",
                product="garri",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=1644.44,
                unit="paint bucket",
                product="garri",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),

            PriceEntry(
                price=5433.33,
                unit="crate",
                product="egg",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=2577.78,
                unit="tuber",
                product="yam",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),

            PriceEntry(
                price=833.33,
                unit="kg",
                product="tomato",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=1605.56,
                unit="kg",
                product="pepper",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=1677.78,
                unit="kg",
                product="tatashe",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=1655.56,
                unit="kg",
                product="shombo",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=1033.33,
                unit="kg",
                product="onion",
                location="Lagos Average",
                user_id="seed_lagos_agric_feb_2026",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),

            # ─────────────────────────────────────────────────────
            # Verified seed prices: NBS Selected Food Price Watch
            # Source: NBS March 2026 reporting
            # Location is "Nigeria Average" because this is national average data.
            # ─────────────────────────────────────────────────────

            PriceEntry(
                price=6127.62,
                unit="crate",
                product="egg",
                location="Nigeria Average",
                user_id="seed_nbs_mar_2026",
                reputation_score=1.0,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=1325.85,
                unit="kg",
                product="beans",
                location="Nigeria Average",
                user_id="seed_nbs_mar_2026",
                reputation_score=1.0,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=801.54,
                unit="kg",
                product="garri",
                location="Nigeria Average",
                user_id="seed_nbs_mar_2026",
                reputation_score=1.0,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=1153.14,
                unit="kg",
                product="onion",
                location="Nigeria Average",
                user_id="seed_nbs_mar_2026",
                reputation_score=1.0,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=5541.25,
                unit="kg",
                product="ginger",
                location="Nigeria Average",
                user_id="seed_nbs_mar_2026",
                reputation_score=1.0,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),

            # ─────────────────────────────────────────────────────
            # Local demo seed data.
            # These are approximate testing values only.
            # They have low reputation_score so verified data weighs more.
            # ─────────────────────────────────────────────────────

            PriceEntry(
                price=850.0,
                unit="kg",
                product="garri",
                location="Mushin Market",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=2300.0,
                unit="paint bucket",
                product="garri",
                location="Mushin Market",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=5)).isoformat(),
            ),

            PriceEntry(
                price=1200.0,
                unit="kg",
                product="okra",
                location="Mile 12",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=2500.0,
                unit="paint bucket",
                product="okra",
                location="Mile 12",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=3)).isoformat(),
            ),

            PriceEntry(
                price=500.0,
                unit="bunch",
                product="efo",
                location="Mushin Market",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=4)).isoformat(),
            ),
            PriceEntry(
                price=700.0,
                unit="bunch",
                product="ugu",
                location="Mile 12",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=3)).isoformat(),
            ),
            PriceEntry(
                price=400.0,
                unit="bunch",
                product="bitter leaf",
                location="Oyingbo Market",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=5)).isoformat(),
            ),

            PriceEntry(
                price=6500.0,
                unit="kg",
                product="egusi",
                location="Mile 12",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=1)).isoformat(),
            ),
            PriceEntry(
                price=1800.0,
                unit="cup",
                product="egusi",
                location="Mile 12",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=8000.0,
                unit="kg",
                product="ogbono",
                location="Mushin Market",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=3)).isoformat(),
            ),
            PriceEntry(
                price=9000.0,
                unit="kg",
                product="crayfish",
                location="Oyingbo Market",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=6)).isoformat(),
            ),
            PriceEntry(
                price=1800.0,
                unit="litre",
                product="palm oil",
                location="Oyingbo Market",
                user_id="demo_seed",
                reputation_score=0.3,
                timestamp=(now - timedelta(hours=6)).isoformat(),
            ),

            # ─────────────────────────────────────────────────────
            # Older local demo data retained for exact-location testing.
            # These are not verified source data.
            # ─────────────────────────────────────────────────────

            PriceEntry(
                price=25000.0,
                unit="basket",
                product="tomato",
                location="Mile 12",
                user_id="u001",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=18000.0,
                unit="basket",
                product="tomato",
                location="Mile 12",
                user_id="u002",
                reputation_score=0.7,
                timestamp=(now - timedelta(hours=5)).isoformat(),
            ),
            PriceEntry(
                price=40000.0,
                unit="basket",
                product="tomato",
                location="Mile 12",
                user_id="u003",
                reputation_score=0.5,
                timestamp=(now - timedelta(days=1)).isoformat(),
            ),
            PriceEntry(
                price=22000.0,
                unit="basket",
                product="tomato",
                location="Mile 12",
                user_id="u004",
                reputation_score=0.8,
                timestamp=(now - timedelta(hours=1)).isoformat(),
            ),
            PriceEntry(
                price=30000.0,
                unit="basket",
                product="tomato",
                location="Mile 12",
                user_id="u005",
                reputation_score=0.6,
                timestamp=(now - timedelta(hours=8)).isoformat(),
            ),

            PriceEntry(
                price=1500.0,
                unit="tuber",
                product="yam",
                location="Oyingbo Market",
                user_id="u001",
                reputation_score=0.9,
                timestamp=(now - timedelta(hours=4)).isoformat(),
            ),
            PriceEntry(
                price=1200.0,
                unit="tuber",
                product="yam",
                location="Oyingbo Market",
                user_id="u006",
                reputation_score=0.4,
                timestamp=(now - timedelta(days=2)).isoformat(),
            ),

            PriceEntry(
                price=8000.0,
                unit="basket",
                product="pepper",
                location="Mile 12",
                user_id="u003",
                reputation_score=0.6,
                timestamp=(now - timedelta(hours=6)).isoformat(),
            ),
            PriceEntry(
                price=8000.0,
                unit="basket",
                product="pepper",
                location="Mushin Market",
                user_id="u002",
                reputation_score=0.7,
                timestamp=(now - timedelta(hours=3)).isoformat(),
            ),
            PriceEntry(
                price=6500.0,
                unit="basket",
                product="pepper",
                location="Mushin Market",
                user_id="u003",
                reputation_score=0.6,
                timestamp=(now - timedelta(hours=7)).isoformat(),
            ),

            PriceEntry(
                price=85000.0,
                unit="50kg bag",
                product="rice",
                location="Mile 12",
                user_id="u001",
                reputation_score=0.8,
                timestamp=(now - timedelta(hours=2)).isoformat(),
            ),
            PriceEntry(
                price=78000.0,
                unit="50kg bag",
                product="rice",
                location="Mile 12",
                user_id="u004",
                reputation_score=0.7,
                timestamp=(now - timedelta(hours=5)).isoformat(),
            ),
        ]

        self._reputations = {
            "u001": 0.9,
            "u002": 0.7,
            "u003": 0.5,
            "u004": 0.8,
            "u005": 0.6,
            "u006": 0.4,
            "demo_seed": 0.3,
            "seed_lagos_agric_feb_2026": 0.9,
            "seed_nbs_mar_2026": 1.0,
        }

    def get_prices(
        self,
        product: str,
        location: str = "",
        days: int = 7,
    ) -> List[PriceEntry]:
        """
        If location is empty, return prices for the product across all locations.
        If location is provided, return only exact product + location matches.
        """
        from datetime import datetime, timedelta

        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            product = (product or "").strip().lower()
            location = (location or "").strip().lower()

            results = []

            for entry in self._prices:
                entry_time = datetime.fromisoformat(entry.timestamp).replace(tzinfo=None)

                product_matches = entry.product.strip().lower() == product

                if location:
                    location_matches = entry.location.strip().lower() == location
                else:
                    location_matches = True

                within_time = entry_time >= cutoff.replace(tzinfo=None)

                if product_matches and location_matches and within_time:
                    results.append(entry)

            return results

        except Exception as e:
            print(f"[MockDBInterface] get_prices error: {e}")
            return []

    def submit_price(self, submission: PriceSubmission) -> bool:
        try:
            self._prices.append(
                PriceEntry(
                    price=submission.price,
                    unit=submission.unit,
                    product=submission.product,
                    location=submission.location,
                    user_id=submission.user_id,
                    reputation_score=self.get_user_reputation(submission.user_id),
                    timestamp=submission.timestamp,
                )
            )
            return True
        except Exception as e:
            print(f"[MockDBInterface] submit_price error: {e}")
            return False

    def get_user_reputation(self, user_id: str) -> float:
        return self._reputations.get(user_id, 0.3)