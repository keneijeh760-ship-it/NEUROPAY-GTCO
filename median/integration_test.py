"""
integration_test.py
-------------------
End-to-end tests for the full pipeline.
Run this before handing off to the backend team.

Usage:
    python integration_test.py

All tests use MockDBInterface and MockParser so no trained models needed.
Green = ready to integrate. Any RED = fix before handoff.
"""

import sys
from datetime import datetime, timezone
from median.main import AILayer

# ── Test runner ───────────────────────────────────────────────────────────────
PASS = 0
FAIL = 0

def test(name: str, got: str, expect_contains: list[str],
         expect_not_contains: list[str] = None):
    global PASS, FAIL
    got_lower = got.lower()

    missing = [e for e in expect_contains
               if e.lower() not in got_lower]
    bad     = [e for e in (expect_not_contains or [])
               if e.lower() in got_lower]

    if not missing and not bad:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}")
        print(f"        Reply:   {got}")
        if missing:
            print(f"        Missing: {missing}")
        if bad:
            print(f"        Should not contain: {bad}")
        FAIL += 1


# ── Setup ─────────────────────────────────────────────────────────────────────
layer = AILayer(use_mock_parser=True)
ts    = datetime.now(timezone.utc).isoformat()

print("\n" + "="*60)
print("INTEGRATION TEST SUITE")
print("="*60)

# ── 1. Greeting ───────────────────────────────────────────────────────────────
print("\n[1] Greeting detection")
test("hello",         layer.process_message("hello",         "u1", ts), ["ask", "price"])
test("good morning",  layer.process_message("good morning",  "u1", ts), ["ask", "price"])
test("hi",            layer.process_message("hi",            "u1", ts), ["ask", "price"])

# ── 2. Query — data found ─────────────────────────────────────────────────────
print("\n[2] Price query — data found (mock DB has tomato @ Mile 12)")
r = layer.process_message("how much tomato for Mile 12", "u1", ts)
test("tomato Mile 12 query", r, ["₦", "tomato", "Mile 12"])

r = layer.process_message("how much yam for Oyingbo Market", "u1", ts)
test("yam oyingbo query",    r, ["₦", "yam", "Oyingbo"])

# ── 3. Query — no data ────────────────────────────────────────────────────────
print("\n[3] Price query — no data in DB")
r = layer.process_message("how much garlic for Wuse Market", "u1", ts)
# no-data template has two variants — check for common token across both
test("garlic no data", r, ["garlic"],
     expect_not_contains=["₦800"])

# ── 4. Submit price ───────────────────────────────────────────────────────────
print("\n[4] Price submission")
r = layer.process_message("tomato is ₦900 per basket for Mile 12", "u1", ts)
test("price submit", r, ["record", "✅"])

# ── 5. Pidgin normalization ───────────────────────────────────────────────────
print("\n[5] Pidgin / noisy input normalization")
r = layer.process_message("hw mch tomatoe 4 mile12", "u1", ts)
test("pidgin noisy query", r, ["₦"])

r = layer.process_message("abeg wetin be price of tomato for Mile 12", "u1", ts)
test("pidgin abeg query", r, ["₦"])

# ── 6. Low confidence / clarification ────────────────────────────────────────
print("\n[6] Clarification on unknown input")
r = layer.process_message("xyzzy football score now", "u1", ts)
test("unknown input", r, ["understand"])

# ── 7. Never crashes ─────────────────────────────────────────────────────────
print("\n[7] Robustness — never crashes")
edge_cases = ["", "   ", "!!???", "₦₦₦₦", "a", "1234567890"]
for case in edge_cases:
    try:
        r = layer.process_message(case, "u1", ts)
        test(f"edge case: '{case}'", r, [], [])  # just must not crash
    except Exception as e:
        test(f"edge case: '{case}'", f"CRASH: {e}", ["no crash"])




# ── 8. Image classification flow ─────────────────────────────────────────────
print("\n[8] Image flow — mock classifier injection")

# We can't run real CLIP in tests (no model downloaded)
# So we monkey-patch the classifier to simulate it

from nlu.image_classifier import ImageClassificationResult
from unittest.mock import MagicMock

def make_mock_classifier(product, confidence, accepted):
    clf = MagicMock()
    clf.classify_bytes.return_value = ImageClassificationResult(
        product     = product,
        confidence  = confidence,
        top_matches = [(product, confidence)],
        accepted    = accepted,
        raw_label   = product,
    )
    return clf

# Test A: image only, product identified → should query price
layer.image_classifier = make_mock_classifier("tomato", 0.87, True)
r = layer.process_message("", "u1", ts, image_bytes=b"fake_image_bytes")
# No location provided — system correctly says no data and asks for submission
test("image only — tomato identified", r, ["tomato"])

# Test B: image + location text → should use image product + text location
layer.image_classifier = make_mock_classifier("yam", 0.80, True)
r = layer.process_message("how much for Oyingbo Market", "u1", ts,
                           image_bytes=b"fake_image_bytes")
test("image + location text", r, ["₦", "yam", "Oyingbo"])

# Test C: image confidence too low → should ask user to clarify
layer.image_classifier = make_mock_classifier(None, 0.30, False)
r = layer.process_message("", "u1", ts, image_bytes=b"fake_image_bytes")
test("image low confidence", r, ["identify", "type"])

# Test D: image + text both have product → image fills in, text location used
layer.image_classifier = make_mock_classifier("pepper", 0.78, True)
r = layer.process_message("how much for Mile 12", "u1", ts,
                           image_bytes=b"fake_image_bytes")
test("image product + text location", r, ["Mile 12"])

# Restore original classifier
from nlu.image_classifier import ImageClassifier
layer.image_classifier = ImageClassifier()

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"RESULTS:  {PASS} passed  |  {FAIL} failed")
print("="*60 + "\n")

if FAIL > 0:
    print("Fix failures before handing off to backend team.\n")
    import sys; sys.exit(1)
else:
    print("All tests passed. Ready for backend integration.\n")