"""
main.py
-------
The single public entry point for the entire AI/ML layer.

Exposes one function:
    process_message(raw_text, user_id, timestamp,
                    image_bytes=None) -> str

The backend team calls this for every incoming WhatsApp message.
If the message contains an image, they pass the raw image bytes.
They get back a reply string. They never touch anything else in this layer.

Image flow:
    If image_bytes is provided:
      1. Run ImageClassifier → identify product
      2. Inject product into ParsedIntent
      3. Continue normal pipeline from Stage 3 onwards

Text + Image:
    If user sends BOTH text and image:
      - Text NLU runs first to extract location / intent
      - Image classifier fills in the product if text didn't provide one
      - Image product overrides text product if text product is None

Wires all four stages:
    Stage 1 — Preprocessor
    Stage 2 — AfroXLMRParser (NLU) + ImageClassifier (if image present)
    Stage 3 — PriceEngine
    Stage 4 — ResponseGenerator
"""

from nlu.preprocessor              import Preprocessor
from nlu.base                      import ParsedIntent
from median.price_engine           import PriceEngine
from median.response_generator     import ResponseGenerator
from median.db_interface           import BaseDBInterface, MockDBInterface
from nlu.image_classifier          import ImageClassifier, ImageClassificationResult
from typing             import Optional


class AILayer:

    def __init__(
        self,
        intent_model_dir: str = "./models/intent",
        ner_model_dir:    str = "./models/ner",
        db: BaseDBInterface   = None,
        use_mock_parser: bool = False,
    ):
        self.preprocessor       = Preprocessor()
        self.price_engine       = PriceEngine(db=db or MockDBInterface())
        self.response_gen       = ResponseGenerator()
        self.image_classifier   = ImageClassifier()   # lazy-loads CLIP on first use

        if use_mock_parser:
            self.parser = _MockParser()
            print("[AILayer] Using mock NLU parser — for development only")
        else:
            from normalization.afroxlmr_parser import AfroXLMRParser
            self.parser = AfroXLMRParser(
                intent_model_dir=intent_model_dir,
                ner_model_dir=ner_model_dir,
            )

    # ── Public contract ───────────────────────────────────────────────────────
    def process_message(
        self,
        raw_text:    str,
        user_id:     str,
        timestamp:   str,
        image_bytes: Optional[bytes] = None,
    ) -> str:
        """
        Takes a raw WhatsApp message string and optional image bytes.
        Returns a reply string. Never raises.

        Backend team passes image_bytes when the WhatsApp message
        contains an attached image.
        """
        try:
            return self._pipeline(raw_text, user_id, timestamp, image_bytes)
        except Exception as e:
            print(f"[AILayer] Unhandled error: {e}")
            return "Something go wrong. Abeg try again 🙏"

    # ── Internal pipeline ─────────────────────────────────────────────────────
    def _pipeline(self, raw_text: str, user_id: str, timestamp: str,
                  image_bytes: Optional[bytes]) -> str:

        # ── Stage 1: Preprocess text ──────────────────────────────
        preprocessed = self.preprocessor.process(raw_text or "")

        # ── Stage 2a: NLU on text ─────────────────────────────────
        parsed = self.parser.parse(preprocessed.cleaned)

        # Handle greeting immediately
        if parsed.intent == "GREETING" and image_bytes is None:
            return self.response_gen.generate_greeting()

        # ── Stage 2b: Image classification (if image present) ─────
        if image_bytes is not None:
            parsed = self._handle_image(parsed, image_bytes)
            if parsed is None:
                # Image was unrecognisable and no text context either
                return (
                    "I no fit identify the product for your image. "
                    "Abeg type the product name — e.g. 'how much tomato for Mile 12'"
                )

        # Confidence gate — skip price lookup if NLU is uncertain and
        # we have no product (image classifier would have filled it in)
        if not parsed.above_gate() and parsed.product is None:
            return self.response_gen.generate_clarification()

        if parsed.intent == "UNKNOWN" and parsed.product is None:
            return self.response_gen.generate_clarification()

        # ── Stage 3: Price intelligence ───────────────────────────
        estimate = self.price_engine.process(parsed, user_id, timestamp)

        # ── Stage 4: Generate response ────────────────────────────
        return self.response_gen.generate(estimate, intent=parsed)

    # ── Image handling ────────────────────────────────────────────────────────
    def _handle_image(self, parsed: ParsedIntent,
                      image_bytes: bytes) -> Optional[ParsedIntent]:
        """
        Run image classifier, inject product into parsed intent.
        Returns updated ParsedIntent, or None if image unrecognisable
        and no other context available.
        """
        result: ImageClassificationResult = \
            self.image_classifier.classify_bytes(image_bytes)

        if not result.accepted:
            # Image confidence too low
            if parsed.product is not None:
                # Text gave us a product — use that, ignore the image
                return parsed
            # No product from text either — unrecoverable
            return None

        # Inject image product if text didn't provide one,
        # or if image is more confident than text NLU
        if parsed.product is None:
            parsed = ParsedIntent(
                intent     = parsed.intent if parsed.intent != "UNKNOWN" else "QUERY",
                product    = result.product,
                unit       = parsed.unit,
                location   = parsed.location,
                price      = parsed.price,
                quantity   = parsed.quantity,
                # Blend text and image confidence
                confidence = max(parsed.confidence, result.confidence),
            )

        return parsed


# ── Mock parser ───────────────────────────────────────────────────────────────
class _MockParser:
    QUERY_KEYWORDS   = {"how much", "price", "cost", "how many", "wetin",
                        "check", "find", "buy"}
    SUBMIT_KEYWORDS  = {"sells for", "is ₦", "na ₦", "cost ₦", "dey go for ₦",
                        "dey sell for"}
    GREETING_WORDS   = {"hello", "hi", "good morning", "good afternoon",
                        "good evening", "hey", "please"}

    KNOWN_PRODUCTS  = ["tomato", "yam", "pepper", "garri", "rice", "beans",
                       "onion", "palm oil", "mackerel", "catfish", "chicken",
                       "beef", "plantain", "cassava", "corn pap", "garlic",
                       "red bell pepper", "scotch bonnet pepper", "eggs",
                       "groundnut", "okra", "cucumber", "carrot"]
    KNOWN_LOCATIONS = ["Mile 12", "Oyingbo Market", "Mushin Market",
                       "Oshodi Market", "Bodija Market", "Wuse Market",
                       "Onitsha Main Market", "Ariaria Market Aba",
                       "Kurmi Market", "Mile 3 Market PH"]

    def parse(self, message: str) -> ParsedIntent:
        import re as _re
        msg = message.lower()

        if any(g in msg for g in self.GREETING_WORDS) and len(msg.split()) <= 3:
            return ParsedIntent(intent="GREETING", product=None, unit=None,
                                location=None, price=None, quantity=None,
                                confidence=0.95)

        is_submit = any(k in msg for k in self.SUBMIT_KEYWORDS)
        is_query  = any(k in msg for k in self.QUERY_KEYWORDS)
        intent    = "SUBMIT_PRICE" if is_submit else (
                    "QUERY" if is_query else "UNKNOWN")

        product  = next((p for p in self.KNOWN_PRODUCTS
                         if p.lower() in msg), None)
        location = next((l for l in self.KNOWN_LOCATIONS
                         if l.lower() in msg), None)

        price_match = _re.search(r'[₦]?([\d,]+)', msg)
        price = float(price_match.group(1).replace(",", "")) \
                if price_match and intent == "SUBMIT_PRICE" else None

        return ParsedIntent(
            intent     = intent,
            product    = product,
            unit       = None,
            location   = location,
            price      = price,
            quantity   = None,
            confidence = 0.70 if intent != "UNKNOWN" else 0.40,
        )


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import datetime, timezone
    print("AILayer — development mode (mock parser + mock DB)\n")

    layer = AILayer(use_mock_parser=True)
    ts    = datetime.now(timezone.utc).isoformat()

    # Text-only tests
    text_tests = [
        ("hello",                                         "u001"),
        ("how much tomato for Mile 12",                   "u001"),
        ("hw mch tomatoe 4 mile12",                       "u002"),
        ("abeg how much yam dey go for Oyingbo Market",   "u003"),
        ("tomato is ₦900 per basket for Mile 12",         "u004"),
        ("weytin e dey cost",                             "u005"),
    ]

    print("── Text-only messages ──")
    for msg, uid in text_tests:
        reply = layer.process_message(msg, uid, ts)
        print(f"IN:  {msg}")
        print(f"OUT: {reply}\n")

    # Simulate image-only message (no text, unknown product in image)
    print("── Image message simulation ──")
    print("(Real CLIP inference requires actual image bytes + model download)")
    print("In production: layer.process_message('', user_id, ts, image_bytes=<bytes>)")