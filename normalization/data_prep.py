import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from collections import defaultdict
import json
import os

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "Davlan/afro-xlmr-base"
MAX_LENGTH = 128  # max tokens per message (WhatsApp msgs are short)
TEST_SIZE = 0.15  # 15% held out for evaluation
VAL_SIZE = 0.10  # 10% validation during training
RANDOM_SEED = 42

# ── Intent label map ──────────────────────────────────────────────────────────
INTENT_LABELS = ["QUERY", "SUBMIT_PRICE", "GREETING", "UNKNOWN"]
INTENT_LABEL2ID = {l: i for i, l in enumerate(INTENT_LABELS)}
INTENT_ID2LABEL = {i: l for l, i in INTENT_LABEL2ID.items()}

# ── NER label map (BIO scheme) ────────────────────────────────────────────────
# O        = outside any entity
# B-XXX    = beginning of entity XXX
# I-XXX    = continuation of entity XXX
NER_LABELS = [
    "O",
    "B-PRODUCT", "I-PRODUCT",
    "B-LOCATION", "I-LOCATION",
    "B-PRICE", "I-PRICE",
    "B-UNIT", "I-UNIT",
]
NER_LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}
NER_ID2LABEL = {i: l for l, i in NER_LABEL2ID.items()}


# ── Tokenizer ─────────────────────────────────────────────────────────────────
def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    return tokenizer


# ── BIO tagging ───────────────────────────────────────────────────────────────
def char_spans_to_bio_tags(text, tokenizer, entity_spans):
    """
    Convert character-level entity spans into BIO token labels.

    Args:
        text:         str — the raw message
        tokenizer:    HuggingFace tokenizer
        entity_spans: list of dicts, each with keys:
                        start (int), end (int), label (str e.g. "PRODUCT")

    Returns:
        token_labels: List[str] — one BIO label per token (including special tokens)

    How it works:
        1. Tokenize with return_offsets_mapping=True to get (char_start, char_end)
           for every token
        2. For each token, check if its char range overlaps any entity span
        3. Assign B-LABEL to the first overlapping token, I-LABEL to the rest
        4. Assign O to tokens outside all spans
        5. Special tokens ([CLS], [SEP]) get label -100 (ignored in loss)
    """
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        return_offsets_mapping=True,
    )
    offset_mapping = encoding["offset_mapping"]  # [(char_start, char_end), ...]
    token_labels = []

    # Track which entity span is "active" to correctly assign B vs I
    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
        # Special tokens have offset (0, 0)
        if char_start == 0 and char_end == 0:
            token_labels.append(-100)
            continue

        assigned = "O"
        for span in entity_spans:
            s_start = span["start"]
            s_end = span["end"]
            label = span["label"]  # e.g. "PRODUCT"

            # Token overlaps this span
            if char_start >= s_start and char_end <= s_end:
                # B- if this token starts at or right after the span start
                if char_start == s_start or (
                        token_idx > 0 and offset_mapping[token_idx - 1][1] <= s_start
                ):
                    assigned = f"B-{label}"
                else:
                    assigned = f"I-{label}"
                break  # spans should not overlap; take first match

        token_labels.append(assigned)

    return token_labels


# ── Row → entity spans ────────────────────────────────────────────────────────
def extract_entity_spans(row):
    """
    Build the entity_spans list from a dataset row's span columns.
    Skips any span where start/end are NaN (GREETING / UNKNOWN rows).
    """
    spans = []

    # Product span
    if pd.notna(row.get("product_span_start")) and pd.notna(row.get("product_span_end")):
        spans.append({
            "start": int(row["product_span_start"]),
            "end": int(row["product_span_end"]),
            "label": "PRODUCT",
        })

    # Location span
    if pd.notna(row.get("location_span_start")) and pd.notna(row.get("location_span_end")):
        spans.append({
            "start": int(row["location_span_start"]),
            "end": int(row["location_span_end"]),
            "label": "LOCATION",
        })

    # Price — inferred from price value position in string if not already a span column
    # The dataset does not have price_span_start/end so we do a simple string search
    if pd.notna(row.get("price")):
        price_str = str(int(row["price"]))
        idx = row["raw_message"].find(price_str)
        if idx != -1:
            spans.append({
                "start": idx,
                "end": idx + len(price_str),
                "label": "PRICE",
            })

    return spans


# ── Main preparation ──────────────────────────────────────────────────────────
def prepare_datasets(csv_path: str, save_dir: str = None):
    """
    Load CSV, build both intent and NER datasets, split into train/val/test.

    Returns:
        intent_dataset:  DatasetDict with train/val/test splits
        ner_dataset:     DatasetDict with train/val/test splits
        intent_label2id: dict
        ner_label2id:    dict
    """
    print(f"\n{'=' * 60}")
    print("DATA PREPARATION PIPELINE")
    print(f"{'=' * 60}")

    # ── 1. Load CSV ───────────────────────────────────────────────
    print(f"\n[1/5] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"      Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── 2. Validate intents ───────────────────────────────────────
    print("\n[2/5] Validating intents...")
    unknown_intents = set(df["intent"].unique()) - set(INTENT_LABELS)
    if unknown_intents:
        raise ValueError(f"Unexpected intent values found: {unknown_intents}. "
                         f"Expected: {INTENT_LABELS}")
    print(f"      Intent distribution:")
    for intent, count in df["intent"].value_counts().items():
        print(f"        {intent}: {count}")

    # ── 3. Load tokenizer ─────────────────────────────────────────
    print(f"\n[3/5] Loading tokenizer...")
    tokenizer = load_tokenizer()

    # ── 4. Build intent records ───────────────────────────────────
    print("\n[4/5] Building intent classification records...")
    intent_records = []
    for _, row in df.iterrows():
        encoding = tokenizer(
            row["raw_message"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )
        intent_records.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": INTENT_LABEL2ID[row["intent"]],
            "message_id": row["message_id"],
        })
    print(f"      Built {len(intent_records)} intent records")

    # ── 5. Build NER records ──────────────────────────────────────
    print("\n[5/5] Building NER records (BIO tagging)...")
    ner_records = []
    skipped = 0
    bio_errors = 0

    # Only NER-relevant rows (has at least one entity span)
    ner_df = df[df["intent"].isin(["QUERY", "SUBMIT_PRICE"])].copy()

    for _, row in ner_df.iterrows():
        entity_spans = extract_entity_spans(row)

        # Generate BIO tags
        bio_tags = char_spans_to_bio_tags(
            row["raw_message"], tokenizer, entity_spans
        )

        # Tokenize with padding for uniform tensor shape
        encoding = tokenizer(
            row["raw_message"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

        # Pad bio_tags to MAX_LENGTH with -100
        padded_labels = bio_tags[:MAX_LENGTH]
        while len(padded_labels) < MAX_LENGTH:
            padded_labels.append(-100)

        # Convert string labels to ints (-100 stays as -100)
        int_labels = [
            NER_LABEL2ID[t] if isinstance(t, str) else t
            for t in padded_labels
        ]

        ner_records.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": int_labels,
            "message_id": row["message_id"],
        })

    print(f"      Built {len(ner_records)} NER records")
    if skipped:
        print(f"      Skipped {skipped} rows (no valid spans)")

    # ── 6. Train / val / test splits ─────────────────────────────
    print(f"\n[+] Splitting datasets...")

    def split_dataset(records, label="dataset"):
        n = len(records)
        n_test = int(n * TEST_SIZE)
        n_val = int(n * VAL_SIZE)
        n_train = n - n_test - n_val

        # Shuffle deterministically
        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.permutation(n)

        train_records = [records[i] for i in indices[:n_train]]
        val_records = [records[i] for i in indices[n_train:n_train + n_val]]
        test_records = [records[i] for i in indices[n_train + n_val:]]

        print(f"    {label}: train={len(train_records)}, "
              f"val={len(val_records)}, test={len(test_records)}")

        return DatasetDict({
            "train": Dataset.from_list(train_records),
            "val": Dataset.from_list(val_records),
            "test": Dataset.from_list(test_records),
        })

    intent_dataset = split_dataset(intent_records, "intent")
    ner_dataset = split_dataset(ner_records, "ner")

    # ── 7. Optionally save to disk ────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        intent_dataset.save_to_disk(os.path.join(save_dir, "intent_dataset"))
        ner_dataset.save_to_disk(os.path.join(save_dir, "ner_dataset"))

        # Save label maps alongside
        with open(os.path.join(save_dir, "intent_label2id.json"), "w") as f:
            json.dump(INTENT_LABEL2ID, f, indent=2)
        with open(os.path.join(save_dir, "ner_label2id.json"), "w") as f:
            json.dump(NER_LABEL2ID, f, indent=2)

        print(f"\n    Datasets saved to: {save_dir}")

    print(f"\n{'=' * 60}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'=' * 60}\n")

    return intent_dataset, ner_dataset, INTENT_LABEL2ID, NER_LABEL2ID


# ── Quick sanity check ────────────────────────────────────────────────────────
def sanity_check(csv_path: str, n_samples: int = 5):
    """
    Print BIO tag output for N sample rows so you can visually verify
    the tagging looks correct before committing to a full training run.
    """
    tokenizer = load_tokenizer()
    df = pd.read_csv(csv_path)
    sample = df[df["intent"].isin(["QUERY", "SUBMIT_PRICE"])].sample(
        n_samples, random_state=RANDOM_SEED
    )

    print(f"\n{'=' * 60}")
    print("BIO TAGGING SANITY CHECK")
    print(f"{'=' * 60}")

    for _, row in sample.iterrows():
        spans = extract_entity_spans(row)
        bio = char_spans_to_bio_tags(row["raw_message"], tokenizer, spans)

        encoding = tokenizer(
            row["raw_message"],
            max_length=MAX_LENGTH,
            truncation=True,
            return_offsets_mapping=True,
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        print(f"\n  Message:  {row['raw_message']}")
        print(f"  Intent:   {row['intent']}")
        print(f"  Entities: {spans}")
        print(f"  {'TOKEN':<20} {'BIO TAG'}")
        print(f"  {'-' * 35}")
        for token, tag in zip(tokens, bio):
            if tag != -100:
                print(f"  {token:<20} {tag}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    csv_path = sys.argv[1] if len(sys.argv) > 1 else \
        r".\normalization\nigeria_market_whatsapp_ai_dataset_500_v3_training_ready.csv"

    save_dir = r".\normalization"

    # Run sanity check first
    sanity_check(csv_path, n_samples=3)

    # Then full preparation
    prepare_datasets(csv_path, save_dir=save_dir)
