import json
import numpy as np
from typing import Optional
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)
import torch
from nlu.base import BaseNLUParser, ParsedIntent

MAX_LENGTH = 128


class AfroXLMRParser(BaseNLUParser):
    def __init__(self, intent_model_dir: str, ner_model_dir: str):
        """
        Args:
            intent_model_dir: path to fine-tuned intent classifier
            ner_model_dir:    path to fine-tuned NER model
        """
        device = 0 if torch.cuda.is_available() else -1  # GPU if available, else CPU

        # ── Intent classifier ─────────────────────────────────────
        self.intent_pipeline = pipeline(
            "text-classification",
            model=intent_model_dir,
            tokenizer=intent_model_dir,
            device=device,
            
            truncation=True,
        )

        # ── NER model ─────────────────────────────────────────────
        # aggregation_strategy="simple" merges B- and I- tokens into one span
        self.ner_pipeline = pipeline(
            "token-classification",
            model=ner_model_dir,
            tokenizer=ner_model_dir,
            aggregation_strategy="simple",
            device=device,
            

        )

        # Load label maps for reference
        #with open(f"{intent_model_dir}/intent_label2id.json") as f:
            #self.intent_label2id = json.load(f)
        #self.intent_id2label = {v: k for k, v in self.intent_label2id.items()}

    # ── Public interface ──────────────────────────────────────────────────────
    def parse(self, message: str) -> ParsedIntent:
        """
        Takes a preprocessed message string.
        Returns a ParsedIntent — never raises.
        """
        try:
            intent, confidence = self._classify_intent(message)
            entities = self._extract_entities(message)

            msg_lower = message.lower()

            query_words = ["how much", "price", "cost", "hw much", "how mch", "berapa", "mch"]
            submit_words = ["i buy", "i bought", "bought", "sell", "sold", "for", "at"]

            if entities.get("PRODUCT") and entities.get("LOCATION") and any(q in msg_lower for q in query_words):
                intent = "QUERY"
                confidence = max(confidence, 0.90)

            if entities.get("PRICE") and entities.get("PRODUCT") and any(s in msg_lower for s in submit_words):
                intent = "SUBMIT_PRICE"
                confidence = max(confidence, 0.90)

            return ParsedIntent(
                intent=intent,
                product=entities.get("PRODUCT"),
                unit=entities.get("UNIT"),
                location=entities.get("LOCATION"),
                price=entities.get("PRICE"),
                quantity=None,  # reserved for future quantity extraction
                confidence=confidence,
            )

        except Exception as e:
            # Never crash the pipeline — return UNKNOWN with zero confidence
            print(f"[AfroXLMRParser] Error: {e}")
            return ParsedIntent(
                intent="UNKNOWN", product=None, unit=None,
                location=None, price=None, quantity=None,
                confidence=0.0,
            )

    # ── Intent classification ─────────────────────────────────────────────────
    def _classify_intent(self, message: str):
        """Returns (intent_label: str, confidence: float)."""
        result = self.intent_pipeline(message)[0]
        label = result["label"]  # e.g. "QUERY"
        confidence = float(result["score"])
        return label, confidence

    # ── NER ───────────────────────────────────────────────────────────────────
    def _extract_entities(self, message: str) -> dict:
        """
        Returns a dict of {entity_type: value}.
        aggregation_strategy="simple" gives us merged spans like:
          {"entity_group": "PRODUCT", "word": "tomato", "score": 0.94, ...}

        For PRICE entities, we clean the string and convert to float.
        For all others, we return the word string directly.
        When multiple spans of the same type appear, we take the highest-score one.
        """
        raw_entities = self.ner_pipeline(message)
        result = {}
        scores = {}

        for ent in raw_entities:
            entity_type = ent["entity_group"]  # e.g. "PRODUCT"
            word = ent["word"].strip()
            score = float(ent["score"])

            # Only keep highest-confidence span per type
            if entity_type not in scores or score > scores[entity_type]:
                scores[entity_type] = score

                if entity_type == "PRICE":
                    result[entity_type] = self._parse_price(word)
                else:
                    result[entity_type] = word

        return result

    # ── Price parsing ─────────────────────────────────────────────────────────
    @staticmethod
    def _parse_price(price_str: str) -> Optional[float]:
        """
        Clean a price string and return a float.
        Handles: "₦850", "850", "850.00", "1,200"
        Returns None if unparseable.
        """
        cleaned = (price_str
                   .replace("₦", "")
                   .replace(",", "")
                   .replace(" ", "")
                   .strip())
        try:
            return float(cleaned)
        except ValueError:
            return None

if __name__ == "__main__":
        parser = AfroXLMRParser(
            intent_model_dir="./normalization/models/intent-model",
            ner_model_dir="./normalization/models/ner-model"
        )

        test_messages = [
            "abeg how much rice for wuse market",
            "i buy tomatoes 2 basket for 15000 at mile 12",
            "good morning",
            "how much is garri in yaba"
        ]

        for msg in test_messages:
            print("\nMessage:", msg)
            print(parser.parse(msg))