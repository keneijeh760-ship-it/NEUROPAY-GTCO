import os
import json
import argparse
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_CHECKPOINT = "Davlan/afro-xlmr-base"

INTENT_LABELS = ["QUERY", "SUBMIT_PRICE", "GREETING", "UNKNOWN"]
INTENT_LABEL2ID = {l: i for i, l in enumerate(INTENT_LABELS)}
INTENT_ID2LABEL = {i: l for l, i in INTENT_LABEL2ID.items()}


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# ── Train ─────────────────────────────────────────────────────────────────────
def train(data_dir: str, output_dir: str):
    print(f"\n{'=' * 60}")
    print("INTENT CLASSIFIER FINE-TUNING")
    print(f"{'=' * 60}\n")

    # Load prepared datasets
    dataset = load_from_disk(os.path.join(data_dir, "intent_dataset"))
    print(f"Train: {len(dataset['train'])}  Val: {len(dataset['val'])}  "
          f"Test: {len(dataset['test'])}")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(INTENT_LABELS),
        id2label=INTENT_ID2LABEL,
        label2id=INTENT_LABEL2ID,
    )

    # Training arguments
    # These are conservative defaults suitable for a 500-row dataset.
    # Increase num_train_epochs if validation F1 is still climbing at epoch 5.
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
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",  # disable wandb / tensorboard unless you want them
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ── Evaluate on test set ──────────────────────────────────────
    print("\n── Test Set Evaluation ──")
    predictions = trainer.predict(dataset["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    print(classification_report(
        labels, preds,
        target_names=INTENT_LABELS,
        digits=3
    ))

    # Save model + label map
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    with open(os.path.join(output_dir, "intent_label2id.json"), "w") as f:
        json.dump(INTENT_LABEL2ID, f, indent=2)

    print(f"\nModel saved to {output_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./prepared_data")
    parser.add_argument("--output_dir", default="./models/intent")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)
