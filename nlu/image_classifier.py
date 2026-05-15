import io
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image


# ── Output schema ─────────────────────────────────────────────────────────────
@dataclass
class ImageClassificationResult:
    product: Optional[str]  # canonical product name, or None if rejected
    confidence: float  # 0.0 – 1.0
    top_matches: list  # [(product, score), ...] top 3
    accepted: bool  # True if confidence >= threshold
    raw_label: str  # exactly what CLIP returned


# ── Produce label list ────────────────────────────────────────────────────────
# These are what CLIP scores the image against.
# Written as natural descriptive phrases — CLIP understands language context.
# Add more as your product coverage grows.

PRODUCE_LABELS = [
    # Vegetables
    "fresh tomatoes",
    "red bell pepper tatashe",
    "scotch bonnet pepper rodo",
    "cayenne pepper shombo",
    "fresh green okra",
    "onions",
    "garlic bulbs",
    "fresh ginger root",
    "fluted pumpkin leaves ugwu",
    "water leaf",
    "bitter leaf",
    "garden eggs",
    "cucumber",
    "carrots",
    "cabbage",
    "spinach",
    "scent leaf efirin",
    "fresh corn maize",
    "sweet potatoes",
    "irish potatoes",

    # Tubers & staples
    "yam tubers",
    "cassava",
    "cocoyam",
    "plantain",
    "unripe plantain",
    "ripe yellow plantain",

    # Grains & legumes
    "white rice",
    "brown beans",
    "black eyed peas",
    "groundnuts peanuts",
    "melon seeds egusi",
    "ogbono seeds",
    "corn flour",

    # Proteins — fish
    "fresh catfish",
    "dried stockfish panla",
    "smoked fish",
    "fresh mackerel titus",
    "crayfish",
    "bonga fish",
    "dried fish",

    # Proteins — meat
    "raw chicken",
    "beef meat",
    "goat meat",
    "cow skin ponmo",
    "snail",
    "turkey",
    "eggs",

    # Oils & condiments
    "red palm oil",
    "groundnut oil bottle",
    "vegetable oil",
    "tomato paste tin",
    "seasoning cubes maggi knorr",
    "salt",
    "sugar",

    # Processed staples
    "garri cassava flakes",
    "corn pap ogi akamu",
    "semovita semolina",
    "wheat flour",
    "bread",
    "instant noodles indomie",
]

# Map CLIP label phrase → canonical product name
LABEL_TO_CANONICAL = {
    "fresh tomatoes": "tomato",
    "red bell pepper tatashe": "red bell pepper",
    "scotch bonnet pepper rodo": "scotch bonnet pepper",
    "cayenne pepper shombo": "cayenne pepper",
    "fresh green okra": "okra",
    "onions": "onion",
    "garlic bulbs": "garlic",
    "fresh ginger root": "ginger",
    "fluted pumpkin leaves ugwu": "fluted pumpkin leaf",
    "water leaf": "water leaf",
    "bitter leaf": "bitter leaf",
    "garden eggs": "garden egg",
    "cucumber": "cucumber",
    "carrots": "carrot",
    "cabbage": "cabbage",
    "spinach": "spinach",
    "scent leaf efirin": "scent leaf",
    "fresh corn maize": "maize",
    "sweet potatoes": "sweet potato",
    "irish potatoes": "irish potato",
    "yam tubers": "yam",
    "cassava": "cassava",
    "cocoyam": "cocoyam",
    "plantain": "plantain",
    "unripe plantain": "unripe plantain",
    "ripe yellow plantain": "ripe plantain",
    "white rice": "rice",
    "brown beans": "brown beans",
    "black eyed peas": "black-eyed peas",
    "groundnuts peanuts": "groundnut",
    "melon seeds egusi": "melon seeds",
    "ogbono seeds": "ogbono seeds",
    "corn flour": "corn flour",
    "fresh catfish": "catfish",
    "dried stockfish panla": "dried stockfish",
    "smoked fish": "smoked fish",
    "fresh mackerel titus": "mackerel",
    "crayfish": "crayfish",
    "bonga fish": "bonga fish",
    "dried fish": "dried fish",
    "raw chicken": "chicken",
    "beef meat": "beef",
    "goat meat": "goat meat",
    "cow skin ponmo": "cow skin",
    "snail": "snail",
    "turkey": "turkey",
    "eggs": "eggs",
    "red palm oil": "palm oil",
    "groundnut oil bottle": "groundnut oil",
    "vegetable oil": "vegetable oil",
    "tomato paste tin": "tomato paste",
    "seasoning cubes maggi knorr": "seasoning cube",
    "salt": "salt",
    "sugar": "sugar",
    "garri cassava flakes": "garri",
    "corn pap ogi akamu": "corn pap",
    "semovita semolina": "semolina",
    "wheat flour": "wheat flour",
    "bread": "bread",
    "instant noodles indomie": "instant noodles",
}


# ── Classifier ────────────────────────────────────────────────────────────────
class ImageClassifier:
    CONFIDENCE_THRESHOLD = 0.55  # below this → reject and ask user to clarify

    def __init__(self, confidence_threshold: float = None):
        self.threshold = confidence_threshold or self.CONFIDENCE_THRESHOLD
        self._model = None
        self._processor = None
        self._loaded = False

    def _load(self):
        """Lazy load CLIP — only on first use to keep startup fast."""
        if self._loaded:
            return
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32").to(self._device)
            self._processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32")
            self._loaded = True
            print("[ImageClassifier] CLIP model loaded")
        except ImportError:
            raise ImportError(
                "transformers and torch required. "
                "Run: pip install transformers torch Pillow"
            )

    # ── Public interface ──────────────────────────────────────────────────────
    def classify_bytes(self, image_bytes: bytes) -> ImageClassificationResult:
        """
        Classify a market produce image from raw bytes.
        This is what the pipeline calls with the WhatsApp image payload.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._classify(image)

    def classify_file(self, filepath: str) -> ImageClassificationResult:
        """Classify from a file path — useful for testing."""
        image = Image.open(filepath).convert("RGB")
        return self._classify(image)

    def classify_pil(self, image: Image.Image) -> ImageClassificationResult:
        """Classify from a PIL Image object."""
        return self._classify(image.convert("RGB"))

    # ── Core classification ───────────────────────────────────────────────────
    def _classify(self, image: Image.Image) -> ImageClassificationResult:
        import torch

        self._load()

        inputs = self._processor(
            text=PRODUCE_LABELS,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image  # shape: (1, n_labels)
            probs = logits.softmax(dim=1)[0]  # shape: (n_labels,)

        # Build ranked results
        scores = [(PRODUCE_LABELS[i], float(probs[i]))
                  for i in range(len(PRODUCE_LABELS))]
        scores.sort(key=lambda x: x[1], reverse=True)

        top_label, top_score = scores[0]
        canonical = LABEL_TO_CANONICAL.get(top_label, top_label)
        accepted = top_score >= self.threshold

        return ImageClassificationResult(
            product=canonical if accepted else None,
            confidence=round(top_score, 4),
            top_matches=[(LABEL_TO_CANONICAL.get(l, l), round(s, 4))
                         for l, s in scores[:3]],
            accepted=accepted,
            raw_label=top_label,
        )
