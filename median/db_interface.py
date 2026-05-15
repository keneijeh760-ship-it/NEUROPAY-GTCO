from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


# ── Shared data classes ───────────────────────────────────────────────────────

@dataclass
class PriceEntry:
    """A single price submission retrieved from the database."""
    price: float  # e.g. 850.0
    unit: str  # e.g. "basket"
    timestamp: str  # ISO 8601 e.g. "2024-11-01T09:32:00Z"
    user_id: str  # anonymized submitter ID
    reputation_score: float  # 0.0 (new/untrusted) to 1.0 (highly trusted)
    location: str  # canonical location name
    product: str  # canonical product name


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
            location: str,
            days: int = 7,
    ) -> List[PriceEntry]:
        """
        Retrieve all price submissions for a product+location within N days.

        Args:
            product:  canonical product name  e.g. "tomato"
            location: canonical location name e.g. "Mile 12"
            days:     lookback window in days (default 7)

        Returns:
            List of PriceEntry objects. Empty list if no data found.
            Must never raise — return [] on error.
        """
        pass

    @abstractmethod
    def submit_price(self, submission: PriceSubmission) -> bool:
        """
        Write a validated price submission to the database.

        Args:
            submission: PriceSubmission dataclass

        Returns:
            True on successful write, False on failure.
            Must never raise — return False on error.
        """
        pass

    @abstractmethod
    def get_user_reputation(self, user_id: str) -> float:
        """
        Get the reputation score for a user.

        Args:
            user_id: anonymized user identifier

        Returns:
            Float between 0.0 and 1.0.
            Return 0.3 (default low trust) for unknown users.
            Must never raise — return 0.3 on error.
        """
        pass


# ── Mock implementation for development / testing ─────────────────────────────
# Use this while the backend team builds the real implementation.
# Replace with the real implementation before deploying.

class MockDBInterface(BaseDBInterface):
    """
    In-memory mock. Pre-seeded with sample data for testing.
    Swap out for the real DB implementation before going live.
    """

    def __init__(self):
        from datetime import datetime, timedelta

        from datetime import timezone as _tz
        now = datetime.now(_tz.utc)

        self._prices = [
            PriceEntry(price=25000.0, unit="basket", product="tomato",
                       location="Mile 12", user_id="u001", reputation_score=0.9,
                       timestamp=(now - timedelta(hours=2)).isoformat()),
            PriceEntry(price=18000.0, unit="basket", product="tomato",
                       location="Mile 12", user_id="u002", reputation_score=0.7,
                       timestamp=(now - timedelta(hours=5)).isoformat()),
            PriceEntry(price=40000.0, unit="basket", product="tomato",
                       location="Mile 12", user_id="u003", reputation_score=0.5,
                       timestamp=(now - timedelta(days=1)).isoformat()),
            PriceEntry(price=22000.0, unit="basket", product="tomato",
                       location="Mile 12", user_id="u004", reputation_score=0.8,
                       timestamp=(now - timedelta(hours=1)).isoformat()),
            PriceEntry(price=30000.0, unit="basket", product="tomato",
                       location="Mile 12", user_id="u005", reputation_score=0.6,
                       timestamp=(now - timedelta(hours=8)).isoformat()),
            PriceEntry(price=1500.0, unit="tuber", product="yam",
                       location="Oyingbo Market", user_id="u001", reputation_score=0.9,
                       timestamp=(now - timedelta(hours=4)).isoformat()),
            PriceEntry(price=1200.0, unit="tuber", product="yam",
                       location="Oyingbo Market", user_id="u006", reputation_score=0.4,
                       timestamp=(now - timedelta(days=2)).isoformat()),
            PriceEntry(price=500.0, unit="mudu (dry measure)", product="garri",
                       location="Mushin Market", user_id="u002", reputation_score=0.7,
                       timestamp=(now - timedelta(hours=3)).isoformat()),
            PriceEntry(price=8000.0, unit="basket", product="pepper",
                       location="Mile 12", user_id="u003", reputation_score=0.6,
                       timestamp=(now - timedelta(hours=6)).isoformat()),
            PriceEntry(price=3500.0, unit="50kg bag", product="rice",
                       location="Wuse Market", user_id="u004", reputation_score=0.8,
                       timestamp=(now - timedelta(hours=1)).isoformat()),
            # Pepper — Mushin
            PriceEntry(price=8000.0, unit="basket", product="pepper",
                       location="Mushin Market", user_id="u002", reputation_score=0.7,
                       timestamp=(now - timedelta(hours=3)).isoformat()),
            PriceEntry(price=6500.0, unit="basket", product="pepper",
                       location="Mushin Market", user_id="u003", reputation_score=0.6,
                       timestamp=(now - timedelta(hours=7)).isoformat()),

            # Rice — Mile 12
            PriceEntry(price=85000.0, unit="50kg bag", product="rice",
                       location="Mile 12", user_id="u001", reputation_score=0.8,
                       timestamp=(now - timedelta(hours=2)).isoformat()),
            PriceEntry(price=78000.0, unit="50kg bag", product="rice",
                       location="Mile 12", user_id="u004", reputation_score=0.7,
                       timestamp=(now - timedelta(hours=5)).isoformat()),
        ]

        self._reputations = {
            "u001": 0.9, "u002": 0.7, "u003": 0.5,
            "u004": 0.8, "u005": 0.6, "u006": 0.4,
        }

    def get_prices(self, product: str, location: str, days: int = 7) -> List[PriceEntry]:
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        return [
            e for e in self._prices
            if e.product.lower() == product.lower()
               and e.location.lower() == location.lower()
               and datetime.fromisoformat(e.timestamp).replace(tzinfo=None) >= cutoff.replace(tzinfo=None)
        ]

    def submit_price(self, submission: PriceSubmission) -> bool:
        self._prices.append(PriceEntry(
            price=submission.price,
            unit=submission.unit,
            product=submission.product,
            location=submission.location,
            user_id=submission.user_id,
            reputation_score=self.get_user_reputation(submission.user_id),
            timestamp=submission.timestamp,
        ))
        return True

    def get_user_reputation(self, user_id: str) -> float:
        return self._reputations.get(user_id, 0.3)