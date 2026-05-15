import random
from typing import Optional
from median.price_engine import PriceEstimate
from nlu.base import ParsedIntent


class ResponseGenerator:
    # ── Templates ─────────────────────────────────────────────────────────────
    # Each key maps to a list of variants for slight randomization.
    # Slot tokens: {product}, {location}, {unit}, {low}, {high},
    #              {median}, {price}, {freshness}, {data_points}

    TEMPLATES = {

        "PRICE_FOUND_FALLBACK": [
            "We no get enough exact data for {product} in {location} yet, "
            "but based on recent prices from other markets, estimated price na "
            "₦{low}–₦{high} per {unit}. Actual price for {location} fit differ.",

            "No exact price for {product} in {location} yet. "
            "General market estimate: ₦{low}–₦{high} per {unit}.",
        ],

        "PRICE_FOUND_HIGH": [
            "{product} dey go for ₦{low}–₦{high} per {unit} for {location}. "
            "Last update: {freshness}. ({data_points} reports)",

            "For {location}, {product} na ₦{low}–₦{high} per {unit} "
            "as of {freshness}.",

            "Current price for {product} for {location}: "
            "₦{low}–₦{high} per {unit}. Updated {freshness}.",
        ],

        "PRICE_FOUND_LOW": [
            "We get small data for {product} for {location}. "
            "Best estimate: ₦{median} per {unit} ({freshness}). "
            "Price fit don change.",

            "Limited reports for {product} for {location} — "
            "last known price na ₦{median} per {unit} ({freshness}). "
            "Help us update am by submitting today's price.",
        ],

        "NO_DATA": [
            "We never get price data for {product} for {location} yet. "
            "You fit help us? Reply with the price wey you see today 🙏",

            "No price info for {product} for {location} in our system. "
            "If you dey there, abeg submit the current price so others fit benefit 🙏",
        ],

        "SUBMIT_CONFIRMED": [
            "Thanks! We don record ₦{price} per {unit} for {product} "
            "for {location} ✅ This go help other buyers.",

            "Price recorded! ₦{price} per {unit} — {product} @ {location} ✅",
        ],

        "CLARIFICATION_NEEDED": [
            "Sorry, I no too understand. You dey ask price or you wan submit price? "
            "Try again like: 'how much tomato for Mile 12'",

            "I no understand your message well. "
            "Example: 'how much yam for Oyingbo' or 'garri na ₦500 per mudu for Mushin'",
        ],

        "GREETING": [
            "Hello! Ask me price for any market product for your area 🛒 "
            "Example: 'how much tomato for Mile 12'",

            "Welcome! I fit help you check current market prices across Nigeria 🇳🇬 "
            "Just ask: 'how much [product] for [market]'",
        ],

        "ERROR": [
            "Something go wrong on our end. Abeg try again in a moment 🙏",
        ],
    }

    def __init__(self, use_random_variants: bool = True):
        """
        Args:
            use_random_variants: if True, randomly pick from template variants.
                                 Set False for deterministic output (testing).
        """
        self.use_random = use_random_variants

    # ── Public interface ──────────────────────────────────────────────────────
    def generate(self, estimate: PriceEstimate,
                 intent: Optional[ParsedIntent] = None) -> str:
        """
        Main entry point.
        Returns a plain string — ready to send to WhatsApp.
        """
        try:
            template_key = self._select_template(estimate, intent)
            template = self._pick_variant(template_key)
            return self._fill_slots(template, estimate)
        except Exception as e:
            print(f"[ResponseGenerator] Error: {e}")
            return self._pick_variant("ERROR")

    def generate_clarification(self) -> str:
        return self._pick_variant("CLARIFICATION_NEEDED")

    def generate_greeting(self) -> str:
        return self._pick_variant("GREETING")

    # ── Template selection ────────────────────────────────────────────────────
    def _select_template(self, estimate: PriceEstimate,
                         intent: Optional[ParsedIntent]) -> str:
        if estimate.status == "submitted":
            return "SUBMIT_CONFIRMED"
        if estimate.status == "fallback":
            return "PRICE_FOUND_FALLBACK"
        if estimate.status == "error":
            return "ERROR"
        if estimate.status == "no_data":
            return "NO_DATA"
        if estimate.confidence in ("high", "medium"):
            return "PRICE_FOUND_HIGH"
        if estimate.confidence == "low":
            return "PRICE_FOUND_LOW"
        return "NO_DATA"

    # ── Variant selection ─────────────────────────────────────────────────────
    def _pick_variant(self, key: str) -> str:
        variants = self.TEMPLATES.get(key, self.TEMPLATES["ERROR"])
        if self.use_random and len(variants) > 1:
            return random.choice(variants)
        return variants[0]

    # ── Slot filling ──────────────────────────────────────────────────────────
    def _fill_slots(self, template: str, estimate: PriceEstimate) -> str:
        def fmt_price(p):
            if p is None:
                return "N/A"
            return f"{int(p):,}" if p == int(p) else f"{p:,.2f}"

        slots = {
            "product": self._titlecase(estimate.product or "the product"),
            "location": estimate.location or "your location",
            "unit": estimate.unit or "unit",
            "low": fmt_price(estimate.price_low),
            "high": fmt_price(estimate.price_high),
            "median": fmt_price(estimate.price_median),
            "price": fmt_price(estimate.price_median or estimate.price_low),
            "freshness": estimate.freshness or "recently",
            "data_points": str(estimate.data_points),
        }

        result = template
        for key, value in slots.items():
            result = result.replace(f"{{{key}}}", value)
        return result

    @staticmethod
    def _titlecase(text: str) -> str:
        return text.title() if text else text
