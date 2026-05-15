import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from nlu.base import ParsedIntent
from median.db_interface import BaseDBInterface, PriceEntry, PriceSubmission


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class PriceEstimate:
    product: str
    location: str
    unit: Optional[str]
    price_low: Optional[float]
    price_high: Optional[float]
    price_median: Optional[float]
    data_points: int
    freshness: Optional[str]  # human-readable e.g. "3 hours ago"
    confidence: str  # high | medium | low | no_data
    status: str  # found | no_data | submitted | error


# ── Engine ────────────────────────────────────────────────────────────────────

class PriceEngine:
    # Recency decay factors per product category
    # Higher = prices go stale faster
    DECAY_FACTORS = {
        "vegetable": 0.30,  # tomato, pepper etc — volatile
        "protein": 0.20,  # fish, meat
        "staple": 0.08,  # garri, yam
        "grain": 0.05,  # rice, beans — stable
        "oil": 0.05,
        "condiment": 0.05,
        "default": 0.15,
    }

    # IQR multiplier — entries outside this are treated as outliers
    IQR_MULTIPLIER = 1.5

    # Minimum data points for each confidence level
    CONFIDENCE_THRESHOLDS = {
        "high": (5, 6),  # (min_data_points, max_hours_old)
        "medium": (2, 72),
        "low": (1, 168),
    }

    def __init__(self, db: BaseDBInterface, lookback_days: int = 7):
        self.db = db
        self.lookback_days = lookback_days

    # ── Public interface ──────────────────────────────────────────────────────
    def process(self, intent: ParsedIntent, user_id: str,
                timestamp: str) -> PriceEstimate:
        """
        Main entry point.
        Routes to _handle_query or _handle_submit based on intent.
        Never raises.
        """
        try:
            if intent.intent == "SUBMIT_PRICE":
                return self._handle_submit(intent, user_id, timestamp)
            else:
                return self._handle_query(intent)
        except Exception as e:
            print(f"[PriceEngine] Error: {e}")
            return PriceEstimate(
                product=intent.product or "unknown",
                location=intent.location or "unknown",
                unit=intent.unit, price_low=None, price_high=None,
                price_median=None, data_points=0,
                freshness=None, confidence="no_data", status="error",
            )

    # ── QUERY path ────────────────────────────────────────────────────────────
    def _handle_query(self, intent: ParsedIntent) -> PriceEstimate:
        product = intent.product or ""
        location = intent.location or ""

        # If no product, we cannot estimate anything useful
        if not product:
            return PriceEstimate(
                product=product,
                location=location,
                unit=intent.unit,
                price_low=None,
                price_high=None,
                price_median=None,
                data_points=0,
                freshness=None,
                confidence="no_data",
                status="no_data",
            )

        # 1. Exact search: product + requested location
        exact_entries = self.db.get_prices(
            product=product,
            location=location,
            days=self.lookback_days,
        )

        if exact_entries:
            return self._build_estimate_from_entries(
                entries=exact_entries,
                product=product,
                location=location,
                stated_unit=intent.unit,
                status="found",
                fallback=False,
            )

        # 2. Fallback search: same product, any location
        fallback_entries = self.db.get_prices(
            product=product,
            location="",
            days=self.lookback_days,
        )

        if fallback_entries:
            return self._build_estimate_from_entries(
                entries=fallback_entries,
                product=product,
                location=location,
                stated_unit=intent.unit,
                status="fallback",
                fallback=True,
            )

        # 3. No product data anywhere
        return PriceEstimate(
            product=product,
            location=location,
            unit=intent.unit,
            price_low=None,
            price_high=None,
            price_median=None,
            data_points=0,
            freshness=None,
            confidence="no_data",
            status="no_data",
        )

    def _build_estimate_from_entries(
            self,
            entries: List[PriceEntry],
            product: str,
            location: str,
            stated_unit: Optional[str],
            status: str,
            fallback: bool = False,
    ) -> PriceEstimate:
        # Step 1 — IQR anomaly filter
        filtered = self._iqr_filter(entries)
        if not filtered:
            filtered = entries

        # Step 2 — Compute weights
        weights = self._compute_weights(filtered, product)

        # Step 3 — Weighted median
        prices = [e.price for e in filtered]
        median = self._weighted_median(prices, weights)

        # Step 4 — Freshness
        freshest_hours = self._hours_since(
            min(filtered, key=lambda e: self._hours_since(e.timestamp)).timestamp
        )

        # Step 5 — Confidence
        confidence = self._compute_confidence(len(filtered), freshest_hours)

        # Fallback estimates should not pretend to be high confidence
        if fallback and confidence == "high":
            confidence = "medium"
        elif fallback and confidence == "medium":
            confidence = "low"

        return PriceEstimate(
            product=product,
            location=location,
            unit=self._infer_unit(filtered, stated_unit),
            price_low=round(min(prices), 2),
            price_high=round(max(prices), 2),
            price_median=round(median, 2),
            data_points=len(filtered),
            freshness=self._human_freshness(freshest_hours),
            confidence=confidence,
            status=status,
        )
    # ── SUBMIT path ───────────────────────────────────────────────────────────
    def _handle_submit(self, intent: ParsedIntent, user_id: str,
                       timestamp: str) -> PriceEstimate:
        # Sanity check — is this price plausible?
        if not self._sanity_check(intent):
            # Accept it but flag with low reputation weight (backend handles this
            # via the user's reputation score staying low)
            print(f"[PriceEngine] Suspicious price: {intent.price} for "
                  f"{intent.product} @ {intent.location}")

        submission = PriceSubmission(
            product=intent.product or "",
            location=intent.location or "",
            unit=intent.unit or "unit",
            price=intent.price or 0.0,
            user_id=user_id,
            timestamp=timestamp,
        )
        success = self.db.submit_price(submission)

        return PriceEstimate(
            product=intent.product or "",
            location=intent.location or "",
            unit=intent.unit,
            price_low=intent.price,
            price_high=intent.price,
            price_median=intent.price,
            data_points=1,
            freshness="just now",
            confidence="high" if success else "no_data",
            status="submitted" if success else "error",
        )

    # ── IQR anomaly filter ────────────────────────────────────────────────────
    def _iqr_filter(self, entries: List[PriceEntry]) -> List[PriceEntry]:
        """Remove entries outside 1.5× IQR from Q1/Q3."""
        if len(entries) < 4:
            return entries  # not enough data for IQR to be meaningful

        prices = sorted(e.price for e in entries)
        n = len(prices)
        q1 = prices[n // 4]
        q3 = prices[(3 * n) // 4]
        iqr = q3 - q1

        if iqr == 0:
            return entries  # all prices identical — nothing to filter

        lower = q1 - self.IQR_MULTIPLIER * iqr
        upper = q3 + self.IQR_MULTIPLIER * iqr

        filtered = [e for e in entries if lower <= e.price <= upper]
        return filtered if filtered else entries

    # ── Weight computation ────────────────────────────────────────────────────
    def _compute_weights(self, entries: List[PriceEntry],
                         product: str) -> List[float]:
        """
        Final weight = recency_weight × reputation_score

        Recency weight uses exponential decay:
            w = e^(-days_old × decay_factor)
        """
        category = self._guess_category(product)
        decay_factor = self.DECAY_FACTORS.get(category, self.DECAY_FACTORS["default"])
        weights = []

        for entry in entries:
            days_old = self._hours_since(entry.timestamp) / 24
            recency_weight = math.exp(-days_old * decay_factor)
            final_weight = recency_weight * entry.reputation_score
            weights.append(max(final_weight, 1e-6))  # never exactly zero

        return weights

    # ── Weighted median ───────────────────────────────────────────────────────
    def _weighted_median(self, values: List[float],
                         weights: List[float]) -> float:
        """
        Weighted median: sort by value, find the point where cumulative
        weight crosses 50% of total weight.
        """
        if not values:
            return 0.0

        paired = sorted(zip(values, weights), key=lambda x: x[0])
        total_w = sum(w for _, w in paired)
        cumulative = 0.0

        for value, weight in paired:
            cumulative += weight
            if cumulative >= total_w / 2:
                return value

        return paired[-1][0]  # fallback

    # ── Confidence ────────────────────────────────────────────────────────────
    def _compute_confidence(self, n_points: int, freshest_hours: float) -> str:
        high_n, high_h = self.CONFIDENCE_THRESHOLDS["high"]
        medium_n, medium_h = self.CONFIDENCE_THRESHOLDS["medium"]
        low_n, low_h = self.CONFIDENCE_THRESHOLDS["low"]

        if n_points >= high_n and freshest_hours <= high_h:
            return "high"
        elif n_points >= medium_n and freshest_hours <= medium_h:
            return "medium"
        elif n_points >= low_n and freshest_hours <= low_h:
            return "low"
        else:
            return "no_data"

    # ── Sanity check ──────────────────────────────────────────────────────────
    def _sanity_check(self, intent: ParsedIntent) -> bool:
        """
        Check if a submitted price is plausible against historical data.
        Returns True if plausible, False if suspicious.
        """
        if not intent.price or intent.price <= 0:
            return False

        entries = self.db.get_prices(
            intent.product or "", intent.location or "", days=30
        )
        if not entries:
            return True  # no history — give benefit of the doubt

        historical = [e.price for e in entries]
        median = sorted(historical)[len(historical) // 2]

        # Flag if submitted price is more than 5× or less than 0.1× the median
        return 0.1 * median <= intent.price <= 5 * median

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _hours_since(self, timestamp: str) -> float:
        """Return hours elapsed since a timestamp string."""
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            # If naive datetime, treat as UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = now - dt
            return delta.total_seconds() / 3600
        except Exception:
            return 999.0  # treat unparseable timestamp as very old

    def _human_freshness(self, hours: float) -> str:
        if hours < 1:
            return "less than an hour ago"
        elif hours < 24:
            return f"{int(hours)} hour{'s' if int(hours) > 1 else ''} ago"
        else:
            days = int(hours / 24)
            return f"{days} day{'s' if days > 1 else ''} ago"

    def _infer_unit(self, entries: List[PriceEntry],
                    stated_unit: Optional[str]) -> Optional[str]:
        """Use stated unit if present, otherwise take most common from entries."""
        if stated_unit:
            return stated_unit
        units = [e.unit for e in entries if e.unit]
        if not units:
            return None
        return max(set(units), key=units.count)

    def _guess_category(self, product: str) -> str:
        """Simple keyword-based category guesser for decay factor selection."""
        p = product.lower()
        if any(k in p for k in ["tomato", "pepper", "leaf", "vegetable", "onion"]):
            return "vegetable"
        if any(k in p for k in ["fish", "chicken", "beef", "meat", "egg", "goat"]):
            return "protein"
        if any(k in p for k in ["yam", "garri", "cassava", "fufu", "plantain"]):
            return "staple"
        if any(k in p for k in ["rice", "beans", "maize", "corn", "millet"]):
            return "grain"
        if any(k in p for k in ["oil", "palm"]):
            return "oil"
        if any(k in p for k in ["maggi", "salt", "seasoning", "spice"]):
            return "condiment"
        return "default"