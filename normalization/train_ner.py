import os
import json
import argparse
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import classification_report, f1_score

MODEL_CHECKPOINT = "Davlan/afro-xlmr-base"

NER_LABELS = [
    "O",
    "B-PRODUCT", "I-PRODUCT",
    "B-LOCATION", "I-LOCATION",
    "B-PRICE", "I-PRICE",
    "B-UNIT", "I-UNIT",
]
NER_LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}
NER_ID2LABEL = {i: l for l, i in NER_LABEL2ID.items()}


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """
    seqeval expects lists of label-name lists (one per sentence).
    We convert int predictions → string labels, stripping -100 (special tokens).
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_labels = []
    true_preds = []

    for pred_seq, label_seq in zip(preds, labels):
        true_label_seq = []
        true_pred_seq = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:  # skip special/padding tokens
                continue
            true_label_seq.append(NER_ID2LABEL[l])
            true_pred_seq.append(NER_ID2LABEL[p])
        true_labels.append(true_label_seq)
        true_preds.append(true_pred_seq)

    return {
        "f1": f1_score(true_labels, true_preds),
    }


# ── Train ─────────────────────────────────────────────────────────────────────
def train(data_dir: str, output_dir: str):
    print(f"\n{'=' * 60}")
    print("NER MODEL FINE-TUNING")
    print(f"{'=' * 60}\n")

    dataset = load_from_disk(os.path.join(data_dir, "ner_dataset"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    print(f"Train: {len(dataset['train'])}  Val: {len(dataset['val'])}  "
          f"Test: {len(dataset['test'])}")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(NER_LABELS),
        id2label=NER_ID2LABEL,
        label2id=NER_LABEL2ID,
    )

    # DataCollator handles dynamic padding per batch
    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ── Evaluate on test set ──────────────────────────────────────
    print("\n── Test Set Evaluation ──")
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

    print(classification_report(true_labels, true_preds, digits=3))

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "ner_label2id.json"), "w") as f:
        json.dump(NER_LABEL2ID, f, indent=2)

    print(f"\nModel saved to {output_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./prepared_data")
    parser.add_argument("--output_dir", default="./models/ner")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)
