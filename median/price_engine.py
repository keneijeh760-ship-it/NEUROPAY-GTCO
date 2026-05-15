import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from nlu.base import ParsedIntent
from median.db_interface import BaseDBInterface, PriceEntry, PriceSubmission
from median.normalizers import normalize_product, normalize_unit


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
    freshness: Optional[str]
    confidence: str  # high | medium | low | no_data
    status: str      # found | fallback | no_data | submitted | error

    # Extra unit-level estimates, e.g.
    # [
    #   {"unit": "kg", "low": 1088.89, "high": 1700.0, "median": 1220.0, "data_points": 4, "derived": True},
    #   {"unit": "25kg bag", "low": 27388.89, "high": 30500.0, "median": 28944.45, "data_points": 2, "derived": False},
    # ]
    unit_options: Optional[list] = None


# ── Engine ────────────────────────────────────────────────────────────────────

class PriceEngine:
    DECAY_FACTORS = {
        "vegetable": 0.30,
        "protein": 0.20,
        "staple": 0.08,
        "grain": 0.05,
        "oil": 0.05,
        "condiment": 0.05,
        "default": 0.15,
    }

    IQR_MULTIPLIER = 1.5

    CONFIDENCE_THRESHOLDS = {
        "high": (5, 6),
        "medium": (2, 72),
        "low": (1, 168),
    }

    def __init__(self, db: BaseDBInterface, lookback_days: int = 7):
        self.db = db
        self.lookback_days = lookback_days

    # ── Public interface ──────────────────────────────────────────────────────
    def process(self, intent: ParsedIntent, user_id: str, timestamp: str) -> PriceEstimate:
        """
        Main entry point.
        Routes to _handle_query or _handle_submit based on intent.
        Never raises.
        """
        try:
            if intent.intent == "SUBMIT_PRICE":
                return self._handle_submit(intent, user_id, timestamp)

            return self._handle_query(intent)

        except Exception as e:
            print(f"[PriceEngine] Error: {e}")
            return PriceEstimate(
                product=intent.product or "unknown",
                location=intent.location or "unknown",
                unit=intent.unit,
                price_low=None,
                price_high=None,
                price_median=None,
                data_points=0,
                freshness=None,
                confidence="no_data",
                status="error",
                unit_options=None,
            )

    # ── QUERY path ────────────────────────────────────────────────────────────
    def _handle_query(self, intent: ParsedIntent) -> PriceEstimate:
        product = normalize_product(intent.product or "")
        location = intent.location or ""
        requested_unit = normalize_unit(intent.unit, product)

        if not product:
            return PriceEstimate(
                product=product,
                location=location,
                unit=requested_unit,
                price_low=None,
                price_high=None,
                price_median=None,
                data_points=0,
                freshness=None,
                confidence="no_data",
                status="no_data",
                unit_options=None,
            )

        # 1. Exact product + location search
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
                stated_unit=requested_unit,
                status="found",
                fallback=False,
            )

        # 2. Fallback: same product, any location
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
                stated_unit=requested_unit,
                status="fallback",
                fallback=True,
            )

        # 3. No product data anywhere
        return PriceEstimate(
            product=product,
            location=location,
            unit=requested_unit,
            price_low=None,
            price_high=None,
            price_median=None,
            data_points=0,
            freshness=None,
            confidence="no_data",
            status="no_data",
            unit_options=None,
        )

    # ── Build estimate from DB entries ─────────────────────────────────────────
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

        # Build multi-unit options BEFORE selecting one main estimate
        all_unit_options = self._build_unit_options(filtered)

        # Step 2 — NEVER mix different units for the main estimate
        requested_unit = (stated_unit or "").strip().lower()

        if requested_unit:
            same_unit_entries = [
                e for e in filtered
                if (e.unit or "").strip().lower() == requested_unit
            ]
        else:
            same_unit_entries = []

        if same_unit_entries:
            filtered = same_unit_entries
        else:
            available_units = [
                (e.unit or "").strip().lower()
                for e in filtered
                if e.unit
            ]

            if available_units:
                most_common_unit = max(set(available_units), key=available_units.count)
                filtered = [
                    e for e in filtered
                    if (e.unit or "").strip().lower() == most_common_unit
                ]

        if not filtered:
            return PriceEstimate(
                product=product,
                location=location,
                unit=stated_unit,
                price_low=None,
                price_high=None,
                price_median=None,
                data_points=0,
                freshness=None,
                confidence="no_data",
                status="no_data",
                unit_options=all_unit_options,
            )

        # Step 3 — Compute weights AFTER unit filtering
        weights = self._compute_weights(filtered, product)

        # Step 4 — Weighted median
        prices = [e.price for e in filtered if e.price is not None]
        median = self._weighted_median(prices, weights)

        # Step 5 — Freshness
        freshest_hours = self._hours_since(
            min(filtered, key=lambda e: self._hours_since(e.timestamp)).timestamp
        )

        # Step 6 — Confidence
        confidence = self._compute_confidence(len(filtered), freshest_hours)

        if fallback and confidence == "high":
            confidence = "medium"
        elif fallback and confidence == "medium":
            confidence = "low"

        # Use the actual unit from the filtered records, not forced/default unit
        actual_unit = self._infer_unit(filtered, None)

        return PriceEstimate(
            product=product,
            location=location,
            unit=actual_unit,
            price_low=round(min(prices), 2),
            price_high=round(max(prices), 2),
            price_median=round(median, 2),
            data_points=len(filtered),
            freshness=self._human_freshness(freshest_hours),
            confidence=confidence,
            status=status,
            unit_options=all_unit_options,
        )

    # ── Multi-unit options ─────────────────────────────────────────────────────
    def _build_unit_options(self, entries: List[PriceEntry]) -> list:
        """
        Build separate price estimates per unit.

        Also derives smaller weight-based options from kg-based bag data.
        Example:
          50kg bag → kg, 5kg, 10kg, 25kg, 50kg bag

        We only derive from units that explicitly contain kg.
        We do NOT derive from basket, paint bucket, mudu, bunch, crate, etc.
        """
        grouped = {}

        for entry in entries:
            unit = (entry.unit or "unit").strip()
            grouped.setdefault(unit, []).append(entry)

        unit_options = []

        # 1. Actual reported units from database
        for unit, unit_entries in grouped.items():
            prices = sorted([e.price for e in unit_entries if e.price is not None])

            if not prices:
                continue

            n = len(prices)
            median = (
                prices[n // 2]
                if n % 2 == 1
                else (prices[n // 2 - 1] + prices[n // 2]) / 2
            )

            unit_options.append({
                "unit": unit,
                "low": round(min(prices), 2),
                "high": round(max(prices), 2),
                "median": round(median, 2),
                "data_points": len(prices),
                "derived": False,
            })

        # 2. Derive smaller kg-based options from entries like "25kg bag" or "50kg bag"
        per_kg_prices = []

        for entry in entries:
            unit = (entry.unit or "").strip().lower()

            match = re.search(r"(\d+)\s*kg", unit)

            if not match:
                continue

            kg_size = float(match.group(1))

            if kg_size <= 0:
                continue

            per_kg_price = entry.price / kg_size
            per_kg_prices.append(per_kg_price)

        if per_kg_prices:
            per_kg_prices = sorted(per_kg_prices)

            derived_sizes = [
                ("kg", 1),
                ("5kg", 5),
                ("10kg", 10),
                ("25kg bag", 25),
                ("50kg bag", 50),
            ]

            existing_units = {
                option["unit"].strip().lower()
                for option in unit_options
            }

            for unit_name, kg_size in derived_sizes:
                if unit_name.lower() in existing_units:
                    continue

                low = min(per_kg_prices) * kg_size
                high = max(per_kg_prices) * kg_size
                median = per_kg_prices[len(per_kg_prices) // 2] * kg_size

                unit_options.append({
                    "unit": unit_name,
                    "low": round(low, 2),
                    "high": round(high, 2),
                    "median": round(median, 2),
                    "data_points": len(per_kg_prices),
                    "derived": True,
                })

        # 3. Sort from smallest/most affordable unit to larger units
        def unit_sort_key(option):
            unit = option["unit"].strip().lower()

            if unit in ["cup", "bunch", "piece", "tuber"]:
                return 0
            if unit == "kg":
                return 1
            if unit == "5kg":
                return 2
            if unit == "10kg":
                return 3
            if "25kg" in unit:
                return 4
            if "50kg" in unit:
                return 5
            if unit in ["paint bucket", "basket"]:
                return 6
            if unit in ["crate"]:
                return 7
            if unit in ["bag"]:
                return 8

            return 99

        unit_options.sort(key=unit_sort_key)

        return unit_options

    # ── SUBMIT path ───────────────────────────────────────────────────────────
    def _handle_submit(self, intent: ParsedIntent, user_id: str, timestamp: str) -> PriceEstimate:
        product = normalize_product(intent.product or "")
        location = intent.location or ""
        unit = normalize_unit(intent.unit, product)

        normalized_intent = ParsedIntent(
            intent=intent.intent,
            product=product,
            unit=unit,
            location=location,
            price=intent.price,
            quantity=intent.quantity,
            confidence=intent.confidence,
        )

        if not self._sanity_check(normalized_intent):
            print(
                f"[PriceEngine] Suspicious price: {intent.price} "
                f"for {product} @ {location}"
            )

        submission = PriceSubmission(
            product=product,
            location=location,
            unit=unit,
            price=intent.price or 0.0,
            user_id=user_id,
            timestamp=timestamp,
        )

        success = self.db.submit_price(submission)

        return PriceEstimate(
            product=product,
            location=location,
            unit=unit,
            price_low=intent.price,
            price_high=intent.price,
            price_median=intent.price,
            data_points=1,
            freshness="just now",
            confidence="high" if success else "no_data",
            status="submitted" if success else "error",
            unit_options=[{
                "unit": unit,
                "low": intent.price,
                "high": intent.price,
                "median": intent.price,
                "data_points": 1,
                "derived": False,
            }] if intent.price is not None else None,
        )

    # ── IQR anomaly filter ────────────────────────────────────────────────────
    def _iqr_filter(self, entries: List[PriceEntry]) -> List[PriceEntry]:
        """Remove entries outside 1.5× IQR from Q1/Q3."""
        if len(entries) < 4:
            return entries

        prices = sorted(e.price for e in entries)
        n = len(prices)

        q1 = prices[n // 4]
        q3 = prices[(3 * n) // 4]
        iqr = q3 - q1

        if iqr == 0:
            return entries

        lower = q1 - self.IQR_MULTIPLIER * iqr
        upper = q3 + self.IQR_MULTIPLIER * iqr

        filtered = [e for e in entries if lower <= e.price <= upper]
        return filtered if filtered else entries

    # ── Weight computation ────────────────────────────────────────────────────
    def _compute_weights(self, entries: List[PriceEntry], product: str) -> List[float]:
        """
        Final weight = recency_weight × reputation_score.
        Recency weight uses exponential decay.
        """
        category = self._guess_category(product)
        decay_factor = self.DECAY_FACTORS.get(category, self.DECAY_FACTORS["default"])

        weights = []

        for entry in entries:
            days_old = self._hours_since(entry.timestamp) / 24
            recency_weight = math.exp(-days_old * decay_factor)
            final_weight = recency_weight * entry.reputation_score
            weights.append(max(final_weight, 1e-6))

        return weights

    # ── Weighted median ───────────────────────────────────────────────────────
    def _weighted_median(self, values: List[float], weights: List[float]) -> float:
        if not values:
            return 0.0

        paired = sorted(zip(values, weights), key=lambda x: x[0])
        total_w = sum(w for _, w in paired)
        cumulative = 0.0

        for value, weight in paired:
            cumulative += weight
            if cumulative >= total_w / 2:
                return value

        return paired[-1][0]

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
        if not intent.price or intent.price <= 0:
            return False

        entries = self.db.get_prices(
            intent.product or "",
            intent.location or "",
            days=30,
        )

        if not entries:
            return True

        # Only compare submitted price against same unit where possible
        same_unit_entries = [
            e for e in entries
            if (e.unit or "").strip().lower() == (intent.unit or "").strip().lower()
        ]

        if same_unit_entries:
            entries = same_unit_entries

        historical = [e.price for e in entries]
        median = sorted(historical)[len(historical) // 2]

        return 0.1 * median <= intent.price <= 5 * median

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _hours_since(self, timestamp: str) -> float:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            delta = now - dt

            return delta.total_seconds() / 3600

        except Exception:
            return 999.0

    def _human_freshness(self, hours: float) -> str:
        if hours < 1:
            return "less than an hour ago"
        elif hours < 24:
            return f"{int(hours)} hour{'s' if int(hours) > 1 else ''} ago"
        else:
            days = int(hours / 24)
            return f"{days} day{'s' if days > 1 else ''} ago"

    def _infer_unit(self, entries: List[PriceEntry], stated_unit: Optional[str]) -> Optional[str]:
        if stated_unit:
            return stated_unit

        units = [e.unit for e in entries if e.unit]

        if not units:
            return None

        return max(set(units), key=units.count)

    def _guess_category(self, product: str) -> str:
        p = product.lower()

        if any(k in p for k in ["tomato", "pepper", "tatashe", "shombo", "leaf", "vegetable", "onion", "okra"]):
            return "vegetable"
        if any(k in p for k in ["fish", "chicken", "beef", "meat", "egg", "goat", "crayfish"]):
            return "protein"
        if any(k in p for k in ["yam", "garri", "cassava", "fufu", "plantain"]):
            return "staple"
        if any(k in p for k in ["rice", "beans", "maize", "corn", "millet"]):
            return "grain"
        if any(k in p for k in ["oil", "palm"]):
            return "oil"
        if any(k in p for k in ["maggi", "salt", "seasoning", "spice", "ginger"]):
            return "condiment"

        return "default"