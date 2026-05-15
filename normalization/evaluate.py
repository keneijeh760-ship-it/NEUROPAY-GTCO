import argparse
import json
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import classification_report as sk_report
from seqeval.metrics import classification_report as seq_report
import os

INTENT_LABELS = ["QUERY", "SUBMIT_PRICE", "GREETING", "UNKNOWN"]
NER_LABELS = [
    "O",
    "B-PRODUCT", "I-PRODUCT",
    "B-LOCATION", "I-LOCATION",
    "B-PRICE", "I-PRICE",
    "B-UNIT", "I-UNIT",
]
NER_ID2LABEL = {i: l for i, l in enumerate(NER_LABELS)}


# ── Intent evaluation ─────────────────────────────────────────────────────────
def evaluate_intent(data_dir, model_dir):
    print(f"\n{'=' * 60}")
    print("INTENT CLASSIFIER — TEST SET EVALUATION")
    print(f"{'=' * 60}\n")

    dataset = load_from_disk(os.path.join(data_dir, "intent_dataset"))
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    args = TrainingArguments(output_dir="/tmp/eval_intent", report_to="none",
                             per_device_eval_batch_size=32)
    trainer = Trainer(model=model, args=args)

    predictions = trainer.predict(dataset["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    print(sk_report(labels, preds, target_names=INTENT_LABELS, digits=3))

    # Per-class breakdown
    correct = sum(p == l for p, l in zip(preds, labels))
    print(f"Overall accuracy: {correct}/{len(labels)} = {correct / len(labels):.3f}")


# ── NER evaluation ────────────────────────────────────────────────────────────
def evaluate_ner(data_dir, model_dir):
    print(f"\n{'=' * 60}")
    print("NER MODEL — TEST SET EVALUATION")
    print(f"{'=' * 60}\n")

    dataset = load_from_disk(os.path.join(data_dir, "ner_dataset"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    args = TrainingArguments(output_dir="/tmp/eval_ner", report_to="none",
                             per_device_eval_batch_size=32)
    trainer = Trainer(
        model=model,
        args=args,

    )

    predictions = trainer.predict(dataset["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(preds, labels):
        tl, tp = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            tl.append(NER_ID2LABEL[l])
            tp.append(NER_ID2LABEL[p])
        true_labels.append(tl)
        true_preds.append(tp)

    print(seq_report(true_labels, true_preds, digits=3))


# ── Interpreting your scores ──────────────────────────────────────────────────
def print_score_guide():
    print("""
── How to read your scores ───────────────────────────────────

INTENT CLASSIFIER (accuracy + F1 macro):
  >= 0.90   Excellent — ship it
  0.80–0.89 Good — acceptable for hackathon, improve post-launch
  0.70–0.79 Weak — check class balance, consider more GREETING/UNKNOWN examples
  < 0.70    Problem — likely a data or training issue

NER (seqeval F1 per entity):
  >= 0.80   Excellent
  0.65–0.79 Acceptable — PRODUCT and LOCATION are most important
  < 0.65    Retrain — check span indices in dataset, likely still corrupted

Priority: PRODUCT F1 > LOCATION F1 > PRICE F1 > UNIT F1
The system degrades gracefully on missing entities but
not on wrong entities — so precision matters more than recall.
""")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./prepared_data")
    parser.add_argument("--intent_model_dir", default="./models/intent")
    parser.add_argument("--ner_model_dir", default="./models/ner")
    args = parser.parse_args()

    evaluate_intent(args.data_dir, args.intent_model_dir)
    evaluate_ner(args.data_dir, args.ner_model_dir)
    print_score_guide()