# NEUROPAY-GTCO — Market Intelligence & Trust ML Microservice

**Squad Hackathon 3.0 · Challenge #1 — “Proof of Life”**  
**This repository:** [NEUROPAY-GTCO](https://github.com/keneijeh760-ship-it/NEUROPAY-GTCO)  
**Consuming application:** **BuyWise / Confam** — FastAPI backend + WhatsApp router (separate app repo) that calls this service over HTTP.  

This repository is **not** a thin wrapper around a general-purpose LLM. It implements a **multi-stage, model-backed inference pipeline**: fine-tuned **Afro-XLMR** encoders for intent + token-level NER, **CLIP** zero-shot visual grounding over a curated Nigerian produce label space, **deterministic** statistical layers (IQR filtering, weighted medians, exponential recency decay, reputation weighting), and explicit **confidence / sanity gates**—all exposed as a **standalone FastAPI** service that the BuyWise backend can call over HTTP with bounded latency and **no requirement** to ship raw chat payloads to third-party generative APIs for core NLU.

---

## Table of contents

1. [Architectural overview](#1-architectural-overview)  
2. [Why this beats a standard “LLM API call”](#2-why-this-beats-a-standard-llm-api-call-the-technical-edge)  
3. [Tech stack & pipeline mechanics](#3-tech-stack--pipeline-mechanics)  
4. [Training logistics & datasets](#4-training-logistics--datasets)  
5. [Isolated microservice run guide](#5-isolated-microservice-run-guide)  
6. [HTTP API reference](#6-http-api-reference)  
7. [Integration contract (BuyWise backend)](#7-integration-contract-buywise-backend)  
8. [Limitations & research roadmap](#8-limitations--research-roadmap)

---

## 1. Architectural overview

### 1.1 Role in the ecosystem

| Aspect | Description |
|--------|-------------|
| **Service type** | **Python FastAPI** inference microservice (`median/api.py`), suitable for **process isolation** (separate venv/container/VM from the main BuyWise FastAPI API). |
| **Ingress** | **JSON** (`POST /parse`, `POST /process`) and **multipart** (`POST /process/image`) from the BuyWise router after WhatsApp (or web chat) normalizes media to bytes/text. |
| **Core objective** | Transform **unstructured** consumer utterances and **optional images** into **structured semantic objects** (`ParsedIntent`) and **price intelligence** (`PriceEstimate`) that support **trust-relevant decisions**: whether reported prices are **internally consistent** with historical crowdsourced distributions, whether NLU confidence warrants a **clarification** instead of a numeric claim, and whether **vision-only** product hypotheses exceed a calibrated softmax threshold. |
| **“Proof of Life” mapping** | In opaque informal markets, “proof” is often **aggregated behavioral evidence** (who said what, when, with what reputation) rather than a single binary document check. This service encodes that as **reputation-weighted, recency-decayed, outlier-filtered** price surfaces plus **multimodal grounding** (text NER + image CLIP) to reduce **counterfeit / phantom listing** style misinformation at the **signal** layer. |

### 1.2 End-to-end control flow

High-level execution path (see `median/main.py` — class `AILayer`):

```text
                    ┌──────────────────────────────────────────────┐
                    │           FastAPI (median/api.py)           │
                    │  /process  /process/image  /parse  /health  │
                    └────────────────────┬─────────────────────────┘
                                         │ single shared AILayer instance
                                         ▼
┌──────────────┐   ┌─────────────────────┴──────────────────────┐
│ Stage 1      │   │ Stage 2a              │ Stage 2b (opt.)    │
│ Preprocessor │──►│ NLU (AfroXLMR or Mock)│ ImageClassifier    │
│ (rules+lex)  │   │ intent + NER spans    │ CLIP ViT-B/32      │
└──────────────┘   └───────────┬───────────┴──────────┬─────────┘
                               │                      │
                               │  ParsedIntent + gate │
                               ▼                      │
                    ┌──────────────────────┐          │
                    │ Stage 3 PriceEngine  │◄─────────┘
                    │ IQR + weights + DB   │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Stage 4 ResponseGen  │
                    │ template + slots     │
                    └──────────────────────┘
```

**Failure isolation:** `AILayer.process_message` catches top-level exceptions and returns a safe user string; `AfroXLMRParser.parse` degrades to `UNKNOWN` with zero confidence on model errors—**the service stays up** under adversarial or corrupted inputs.

---

## 2. Why this beats a standard “LLM API call” (the technical edge)

### 2.1 Deterministic vs probabilistic generative stacks

| Dimension | Typical “one-shot LLM” wrapper | NEUROPAY-GTCO approach |
|-----------|-------------------------------|------------------------|
| **Output schema** | Free-form prose; fragile JSON mode | First-class **`ParsedIntent`** / **`ParseResponse`** with **typed fields** (`intent`, `product`, `location`, `price`, `unit`, `confidence`). |
| **Hallucination surface** | Model may invent markets, prices, or units | **Price numerals** for NER come from **token spans** + `_parse_price` cleaning; **aggregate prices** come from **DB-backed** rows + **IQR** filtering, not from language model imagination. |
| **Latency tail** | Large autoregressive decoding | **Encoder-only** inference (intent ≤128 tokens, NER same bound) + optional **single forward** CLIP pass; no autoregressive loop for NLU. |
| **Auditability** | Black-box prompt | **Interpretable stages**: preprocessor diffs, pipeline logits/scores, engine filters (documentable for judges). |

### 2.2 Customized modeling (localized fraud / market semantics)

- **Intent + NER** are fine-tuned from **`Davlan/afro-xlmr-base`** (`normalization/train_intent.py`, `normalization/train_ner.py`) — a multilingual encoder **aligned with African language varieties**, not a generic English-only BERT.
- **Heuristic arbitration layer** inside `AfroXLMRParser.parse` **deterministically boosts** intent when **entity spans + lexical triggers** align (e.g. PRODUCT+LOCATION + query words ⇒ force `QUERY`; PRODUCT+PRICE + submit lexicon ⇒ force `SUBMIT_PRICE`). This mitigates **classifier miscalibration** on short WhatsApp-style utterances.
- **Vision** uses **`openai/clip-vit-base-patch32`** with a **hand-engineered ontology** (`PRODUCE_LABELS`, `LABEL_TO_CANONICAL`) mapping **natural language prompts** (“fresh tomatoes”, “garri cassava flakes”) to **canonical SKUs**—classic **zero-shot transfer** without fine-tuning CLIP on your GPU budget, while still being a **real vision model forward pass**, not OCR pasted into ChatGPT.

### 2.3 Latency, privacy, and operational security

- **No mandatory third-party generative API** for the documented core path: models run **in your process** (GPU optional, CPU fallback via Hugging Face `pipeline` device map).
- **BuyWise** may still call OpenRouter / etc. for **product UX** outside this repo; this microservice remains the **grounded NLU + pricing math** boundary.
- **CORS** is permissive on the FastAPI app for hackathon demos—**tighten in production** to the BuyWise origin only.

---

## 3. Tech stack & pipeline mechanics

### 3.1 Core dependencies (`median/requirement.txt`)

| Package | Role |
|---------|------|
| **torch ≥2** | CLIP + transformer inference; CUDA optional. |
| **transformers ≥4.40** | `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `CLIPModel`, `pipeline` factory. |
| **datasets** | `load_from_disk` for prepared intent/NER arrow datasets. |
| **seqeval** | Span-level NER metrics during training. |
| **scikit-learn** | Classification metrics (`accuracy`, macro **F1**). |
| **pandas / numpy** | Dataset prep & vector manipulations in training scripts. |
| **fastapi / uvicorn** | ASGI service boundary. |
| **prophet** | Declared for **time-series extensions** (not wired into the hot `PriceEngine` path in the current tree—safe to omit at install time if you patch requirements for minimal images). |

**Imaging:** **Pillow** is imported by `nlu/image_classifier.py` — install explicitly if you use image endpoints (`pip install Pillow`).

### 3.2 Stage 1 — Preprocessor (`nlu/preprocessor.py`)

- **Deterministic normalization**: abbreviation expansion (Pidgin / chat shorthand), **noise stripping** (“lol”, “chai”), **leet-style digit substitutions** (`4` → “for”), and a large **product alias graph** (e.g. regional names for tomato, pepper, garri).
- **Outputs** `PreprocessedMessage` with **`cleaned`** text fed to transformers; **`noise_level`** and **`language`** hints for downstream analytics (currently used for telemetry-style fields, not a separate model).

### 3.3 Stage 2a — NLP (`normalization/afroxlmr_parser.py`)

| Component | Implementation detail |
|-----------|-------------------------|
| **Intent** | `pipeline("text-classification", model=intent_model_dir, truncation=True, max_length=128)` → label + softmax score. |
| **NER** | `pipeline("token-classification", model=ner_model_dir, aggregation_strategy="simple")` → merged spans (`PRODUCT`, `LOCATION`, `PRICE`, `UNIT`). |
| **Span disambiguation** | For duplicate entity types, **highest softmax span wins**. |
| **Price parsing** | `_parse_price` strips `₦`, commas, whitespace → `float`. |
| **Rule overrides** | Keyword sets for **QUERY** vs **SUBMIT_PRICE** when entities co-occur—reduces **false UNKNOWN** and **wrong intent** on edge templates common in Nigerian market chat. |

**Confidence gate:** `ParsedIntent.CONFIDENCE_GATE = 0.65` (`nlu/base.py`) — if below gate **and** `product` is missing, `ResponseGenerator.generate_clarification()` fires before expensive DB work.

### 3.4 Stage 2b — Computer vision (`nlu/image_classifier.py`)

| Item | Detail |
|------|--------|
| **Model** | `openai/clip-vit-base-patch32` (ViT-B/32 image encoder + text tower). |
| **Inference** | `torch.no_grad()`, single batch, **softmax over full label list** → ranked `(label, probability)` pairs. |
| **Thresholding** | `CONFIDENCE_THRESHOLD = 0.55` — sub-threshold predictions are **rejected** (`accepted=False`) to avoid false product injection. |
| **Integration** | `AILayer._handle_image` merges CLIP product into `ParsedIntent`, blending confidence with `max(text_conf, image_conf)` when text lacked product. |

**Important:** This is **produce recognition for market pricing**, not document liveness detection. It still constitutes **real computer vision** with measurable calibration knobs (threshold, label set cardinality ~70+ entries).

### 3.5 Stage 3 — Price engine & statistical anomaly layer (`median/price_engine.py`)

| Mechanism | Purpose |
|-----------|---------|
| **IQR filter (`_iqr_filter`)** | Classic **1.5×IQR** rule on price samples before aggregation—drops extreme spam / fat-finger values when ≥4 points exist. |
| **Weighted median (`_weighted_median`)** | Robust central tendency under skewed informal-market distributions. |
| **Weights (`_compute_weights`)** | `exp(-days_old × decay) × reputation_score` — **exponential recency decay** per product category (`DECAY_FACTORS` for vegetable vs staple vs protein…). |
| **Confidence bands (`_compute_confidence`)** | Discretizes `high | medium | low | no_data` from **(n_points, freshest_hours)** thresholds—explicit **data sufficiency** signal for judges. |
| **Submit sanity (`_sanity_check`)** | Rejects absurd submissions vs 30-day historical median band (`0.1× … 5×`) for same unit bucket—**adversarial price injection** mitigation. |
| **Unit normalization** | `median/normalizers.py` bridges colloquial units to comparison space. |

The `BaseDBInterface` abstraction (`median/db_interface.py`) allows **MockDBInterface** (seeded Lagos ministry–style references + synthetic noise) or a **production adapter** implemented by BuyWise against PostgreSQL.

### 3.6 Stage 4 — Response generation (`median/response_generator.py`)

- **Template-controlled** natural language with **randomized polite variants** (reduces repetitive bot feel) but **bounded** slot filling—no open-ended generation, so **WYSIWYG** alignment with `PriceEstimate` fields.

### 3.7 Development mode — Mock NLU (`median/main.py` → `_MockParser`)

When `USE_MOCK_PARSER=true` (default in `median/api.py`), the service avoids loading heavy transformer weights—useful for **CI / judge laptops without GPU**. Production demos for **AI Technical Depth** should set **`USE_MOCK_PARSER=false`** and ship the **fine-tuned checkpoint directories**.

---

## 4. Training logistics & datasets

### 4.1 Dataset artifacts

| Asset | Location / notes |
|-------|-------------------|
| **Intent HF dataset** | `normalization/intent_dataset/` (train/val/test shards via `datasets`). |
| **NER HF dataset** | `normalization/ner_dataset/`. |
| **Source CSV (example)** | `normalization/nigeria_market_whatsapp_ai_dataset_500_v3_training_ready.csv` — WhatsApp-flavored Nigerian market language. |
| **Label maps** | `normalization/intent_label2id.json`, `normalization/ner_label2id.json`. |

### 4.2 Intent training (`normalization/train_intent.py`)

- **Base checkpoint:** `Davlan/afro-xlmr-base`.  
- **Head:** `AutoModelForSequenceClassification` with labels `QUERY | SUBMIT_PRICE | GREETING | UNKNOWN`.  
- **Optimization:** AdamW-style defaults via Hugging Face `TrainingArguments` (`lr=2e-5`, `weight_decay=0.01`, `warmup_ratio=0.1`), **epoch-level eval**, **`load_best_model_at_end`** with **`f1_macro`** as selection metric, **`EarlyStoppingCallback(patience=2)`**.  
- **Metrics:** accuracy + **macro-F1** (important under class imbalance).

### 4.3 NER training (`normalization/train_ner.py`)

- **BIO schema** for `PRODUCT`, `LOCATION`, `PRICE`, `UNIT`.  
- **Data collator:** `DataCollatorForTokenClassification` for dynamic padding.  
- **Evaluation:** `seqeval` **entity-level F1** + classification report.

### 4.4 Preprocessing & adversarial hygiene

- **Preprocessor** reduces **orthographic adversarial variance** (abbreviations, Pidgin orthography) *before* tokenization—stabilizes tokenizer alignment for NER.  
- **Parser try/except** returns low-confidence `UNKNOWN` instead of crashing on **malformed UTF-8 / emoji bombs**.  
- **PriceEngine** combines **IQR**, **historical median band sanity**, and **reputation weighting**—multi-signal **consistency checking** rather than trusting any single user string.

### 4.5 Fairness & bias (engineering stance)

- Models inherit **socio-linguistic biases** from training distributions; the service **surfaces confidence** and **asks for clarification** instead of fabricating prices when uncertain.  
- **Mock seeds** include **high-reputation ministry-style** anchors vs lower-reputation synthetic rows—demonstrates how **reputation enters the math**, not just the narrative.

---

## 5. Isolated microservice run guide

### 5.1 Prerequisites

- **Python 3.10+** recommended (tested mindset: 3.11).  
- **Git** + ~4–8 GB free disk for **torch** wheels + optional model cache.  
- **GPU**: optional; CLIP + XLMR-small batches run on **CPU** for demos but slower.

### 5.2 Installation

From the repository root (`NEUROPAY-GTCO/`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r median/requirement.txt
pip install Pillow
```

> **CUDA:** install the **torch build matching your NVIDIA driver** from PyTorch’s index if you need GPU (`--index-url https://download.pytorch.org/whl/cu124` etc.).

### 5.3 Model directories

Fine-tuned weights are **not** committed to git in many setups. Train locally (`python normalization/train_intent.py …`, `train_ner.py …`) or copy checkpoints into:

```text
normalization/models/intent-model/
normalization/models/ner-model/
```

### 5.4 Environment variables

| Variable | Purpose |
|----------|---------|
| **`USE_MOCK_PARSER`** | `true` (default): `_MockParser` — no Afro-XLMR load. `false`: load real checkpoints. |
| **`INTENT_MODEL_DIR`** | Path to intent classifier (default `./normalization/models/intent-model`). |
| **`NER_MODEL_DIR`** | Path to NER model (default `./normalization/models/ner-model`). |
| **`PYTHONPATH`** | Must include repo root so `median.api:app` resolves imports (`from median.main import AILayer`, `from nlu…`). |

### 5.5 Launch (Uvicorn)

**Bind on 8001** when BuyWise already uses **8000**:

```powershell
cd NEUROPAY-GTCO
$env:PYTHONPATH = "."
$env:USE_MOCK_PARSER = "true"
uvicorn median.api:app --host 0.0.0.0 --port 8001 --reload
```

**Full ML mode:**

```powershell
$env:PYTHONPATH = "."
$env:USE_MOCK_PARSER = "false"
$env:INTENT_MODEL_DIR = "./normalization/models/intent-model"
$env:NER_MODEL_DIR = "./normalization/models/ner-model"
uvicorn median.api:app --host 0.0.0.0 --port 8001
```

Smoke tests:

```http
GET http://127.0.0.1:8001/health
GET http://127.0.0.1:8001/docs
```

### 5.6 `curl` — structured parse (`/parse`)

```bash
curl -s -X POST "http://127.0.0.1:8001/parse" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"how much is garri in yaba\"}"
```

Expected shape:

```json
{
  "intent": "QUERY",
  "product": "garri",
  "unit": null,
  "location": "yaba",
  "price": null,
  "quantity": null,
  "confidence": 0.9
}
```

*(Exact values depend on mock vs trained weights.)*

### 5.7 `curl` — full conversational reply (`/process`)

```bash
curl -s -X POST "http://127.0.0.1:8001/process" \
  -H "Content-Type: application/json" \
  -d "{\"raw_text\": \"how much is garri in yaba\", \"user_id\": \"judge-demo-1\", \"timestamp\": \"2026-05-16T12:00:00Z\"}"
```

Returns:

```json
{ "reply": "…natural language estimate or clarification…" }
```

### 5.8 `curl` — multimodal (`/process/image`)

```bash
curl -s -X POST "http://127.0.0.1:8001/process/image" \
  -F user_id=judge-demo-2 \
  -F timestamp=2026-05-16T12:00:00Z \
  -F raw_text="how much for this" \
  -F image=@./sample_market.jpg
```

---

## 6. HTTP API reference

| Method | Path | Body | Response |
|--------|------|------|----------|
| `GET` | `/health` | — | `{ "status": "ok" }` |
| `GET` | `/` | — | Service metadata + endpoint index |
| `POST` | `/parse` | `{ "message": "<text>" }` | `ParseResponse` (structured) |
| `POST` | `/process` | `{ "raw_text", "user_id", "timestamp" }` | `{ "reply": "<text>" }` |
| `POST` | `/process/image` | `multipart/form-data` | `{ "reply": "<text>" }` |

**Image MIME allowlist:** `image/jpeg`, `image/png`, `image/webp`, `image/jpg`.

---

## 7. Integration contract (BuyWise backend)

BuyWise should set:

```env
ML_API_URL=http://127.0.0.1:8001
```

- **`POST /parse`** — when the backend wants **structured JSON** to merge with its **Postgres** `price_reports` and **rule-based submit detector** (recommended for auditable persistence).  
- **`POST /process`** — when the backend wants a **finished user reply string** with **all four internal stages** executed server-side.  
- **Never** block the BuyWise event loop on ML—use **async HTTP** or a **worker queue** at scale.

---

## 8. Limitations & research roadmap

| Gap | Mitigation / next step |
|-----|-------------------------|
| **CLIP label set coverage** | Expand `PRODUCE_LABELS` + map table; optional **few-shot linear probe** on frozen CLIP features. |
| **Intent misfires on SUBMIT_PRICE** | BuyWise implements **pre-ML conversational submit detection**; continue **data collection** + **hard-negative mining** in `train_intent.py`. |
| **Prophet unused** | Wire optional **per-SKU time-series** forecasting for anomaly alerts (`price_t` vs Prophet band). |
| **DB mock in default path** | Implement `BaseDBInterface` against **live Postgres** for production. |
| **Liveness / document fraud** | Out of current scope; would require **separate CV models** (texture CNNs, frequency analysis) — candidate **Challenge #1** extension track. |

---

## Citation & team context

Built as the **dedicated ML inference plane** for **BuyWise**’s Squad Hackathon 3.0 submission under **Challenge #1 — Proof of Life**, emphasizing **measurable trust signals** (confidence gates, IQR, reputation, recency) rather than **ungrounded generative prose**.

For repository layout of the **host application**, see the BuyWise / Confam monorepo README.

---

*README version: 2026-05 — aligned to in-tree sources: `median/api.py`, `median/main.py`, `normalization/afroxlmr_parser.py`, `nlu/preprocessor.py`, `nlu/image_classifier.py`, `median/price_engine.py`.*
