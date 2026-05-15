import os

files = {
    "median/main.py": [
        ("from preprocessor       import Preprocessor",                              "from nlu.preprocessor              import Preprocessor"),
        ("from base               import ParsedIntent",                              "from nlu.base                      import ParsedIntent"),
        ("from price_engine       import PriceEngine",                               "from median.price_engine           import PriceEngine"),
        ("from response_generator import ResponseGenerator",                         "from median.response_generator     import ResponseGenerator"),
        ("from db_interface       import BaseDBInterface, MockDBInterface",          "from median.db_interface           import BaseDBInterface, MockDBInterface"),
        ("from image_classifier   import ImageClassifier, ImageClassificationResult","from nlu.image_classifier          import ImageClassifier, ImageClassificationResult"),
        ("from afroxlmr_parser import AfroXLMRParser",                              "from normalization.afroxlmr_parser import AfroXLMRParser"),
    ],
    "median/api.py": [
        ("from main import AILayer",                                                 "from median.main import AILayer"),
    ],
    "median/price_engine.py": [
        ("from base import ParsedIntent",                                            "from nlu.base import ParsedIntent"),
        ("from db_interface import BaseDBInterface, PriceEntry, PriceSubmission",   "from median.db_interface import BaseDBInterface, PriceEntry, PriceSubmission"),
    ],
    "median/response_generator.py": [
        ("from price_engine import PriceEstimate",                                  "from median.price_engine import PriceEstimate"),
        ("from base import ParsedIntent",                                            "from nlu.base import ParsedIntent"),
    ],
    "median/integration_test.py": [
        ("from main import AILayer",                                                 "from median.main import AILayer"),
        ("from image_classifier import ImageClassificationResult",                   "from nlu.image_classifier import ImageClassificationResult"),
        ("from image_classifier import ImageClassifier",                             "from nlu.image_classifier import ImageClassifier"),
    ],
    "normalization/afroxlmr_parser.py": [
        ("from base import BaseNLUParser, ParsedIntent",                             "from nlu.base import BaseNLUParser, ParsedIntent"),
    ],
}

for filepath, replacements in files.items():
    filepath = filepath.replace("/", "\\")
    if not os.path.exists(filepath):
        print(f"SKIP (not found): {filepath}")
        continue
    content = open(filepath, encoding="utf-8").read()
    for old, new in replacements:
        content = content.replace(old, new)
    open(filepath, "w", encoding="utf-8").write(content)
    print(f"Fixed: {filepath}")

print("\nDone.")