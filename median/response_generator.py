import random
from typing import Optional
from median.price_engine import PriceEstimate
from nlu.base import ParsedIntent


class ResponseGenerator:
    # ── Templates ─────────────────────────────────────────────────────────────
    # Slot tokens:
    # {product}, {location}, {unit}, {low}, {high}, {median}, {price},
    # {freshness}, {data_points}, {unit_options}

    TEMPLATES = {
        "PRICE_FOUND_FALLBACK": [
            "No exact price for {product} in {location} yet. "
            "Based on available market reports, here are estimated prices:\n"
            "{unit_options}\n"
            "Actual price for {location} may differ.",

            "We do not have enough exact data for {product} in {location} yet. "
            "General market estimates:\n"
            "{unit_options}",
        ],

        "PRICE_FOUND_HIGH": [
            "{product} price estimates for {location}:\n"
            "{unit_options}\n"
            "Last update: {freshness}. ({data_points} reports)",

            "Current price estimates for {product} in {location}:\n"
            "{unit_options}\n"
            "Updated {freshness}.",
        ],

        "PRICE_FOUND_LOW": [
            "We have limited data for {product} in {location}. "
            "Best available estimates:\n"
            "{unit_options}\n"
            "Prices may have changed.",

            "Limited reports for {product} in {location}. "
            "Last known estimates:\n"
            "{unit_options}",
        ],

        "NO_DATA": [
            "We do not have enough price data for {product} in {location} yet. "
            "You can help by submitting the price you saw today.",

            "No price info for {product} in {location} in our system yet. "
            "If you are there, please submit the current price so others can benefit.",
        ],

        "SUBMIT_CONFIRMED": [
            "Thanks! We recorded ₦{price} per {unit} for {product} in {location}. "
            "This will help other buyers.",

            "Price recorded: ₦{price} per {unit} — {product} @ {location}.",
        ],

        "CLARIFICATION_NEEDED": [
            "I could not understand the message clearly. Try: "
            "'how much is tomato in Mile 12' or 'rice is ₦61,000 per 50kg bag in Yaba'.",

            "Please include the product and market/location. Example: "
            "'how much is garri in Yaba'.",
        ],

        "GREETING": [
            "Welcome! Ask for a market price like: 'how much is garri in Yaba'.",

            "Hello! I can help estimate market prices. Try: 'how much is rice in Yaba'.",
        ],

        "ERROR": [
            "Something went wrong on our end. Please try again in a moment.",
        ],
    }

    def __init__(self, use_random_variants: bool = True):
        self.use_random = use_random_variants

    # ── Public interface ──────────────────────────────────────────────────────
    def generate(
        self,
        estimate: PriceEstimate,
        intent: Optional[ParsedIntent] = None
    ) -> str:
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
    def _select_template(
        self,
        estimate: PriceEstimate,
        intent: Optional[ParsedIntent]
    ) -> str:
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

        unit_options_text = self._format_unit_options(estimate)

        # Fallback if unit_options is missing for any reason
        if not unit_options_text:
            low = fmt_price(estimate.price_low)
            high = fmt_price(estimate.price_high)
            unit = estimate.unit or "unit"

            if estimate.price_low == estimate.price_high:
                unit_options_text = f"\n- ₦{low} per {unit}"
            else:
                unit_options_text = f"\n- ₦{low}–₦{high} per {unit}"

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
            "unit_options": unit_options_text,
        }

        result = template

        for key, value in slots.items():
            result = result.replace(f"{{{key}}}", value)

        return result.strip()

    def _format_unit_options(self, estimate: PriceEstimate) -> str:
        options = getattr(estimate, "unit_options", None)

        if not options:
            return ""

        lines = []

        for option in options[:8]:
            unit = option.get("unit", "unit")
            low = option.get("low")
            high = option.get("high")
            derived = option.get("derived", False)

            label = "estimated" if derived else "reported"

            low_text = self._format_price(low)
            high_text = self._format_price(high)

            if low == high:
                lines.append(f"- ₦{low_text} per {unit} ({label})")
            else:
                lines.append(f"- ₦{low_text}–₦{high_text} per {unit} ({label})")

        return "\n" + "\n".join(lines)

    @staticmethod
    def _format_price(p):
        if p is None:
            return "N/A"
        return f"{int(p):,}" if p == int(p) else f"{p:,.2f}"

    @staticmethod
    def _titlecase(text: str) -> str:
        return text.title() if text else text