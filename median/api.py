"""
api.py
------
FastAPI wrapper around the AI/ML layer.
Accepts text messages and optional image uploads.

Endpoints:
    POST /process       — text only
    POST /process/image — text + image (multipart form)
    GET  /health        — health check

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from median.main import AILayer
import os

app = FastAPI(title="Market Price AI Layer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────
class TextRequest(BaseModel):
    raw_text:  str
    user_id:   str
    timestamp: str

class MessageResponse(BaseModel):
    reply: str


# ── Initialise AI layer once at startup ───────────────────────────────────────
USE_MOCK = os.getenv("USE_MOCK_PARSER", "true").lower() == "true"

layer = AILayer(
    intent_model_dir = os.getenv("INTENT_MODEL_DIR", "./normalization/models/intent-model"),
    ner_model_dir    = os.getenv("NER_MODEL_DIR", "./normalization/models/ner-model"),
    use_mock_parser  = USE_MOCK,
)
print(f"AI Layer started — mock_parser={USE_MOCK}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/process", response_model=MessageResponse)
def process_text(request: TextRequest):
    """
    Text-only messages. Backend team calls this for normal WhatsApp messages.
    """
    if not request.raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text cannot be empty")

    reply = layer.process_message(
        raw_text  = request.raw_text,
        user_id   = request.user_id,
        timestamp = request.timestamp,
    )
    return MessageResponse(reply=reply)


@app.post("/process/image", response_model=MessageResponse)
async def process_image(
    user_id:   str        = Form(...),
    timestamp: str        = Form(...),
    raw_text:  str        = Form(""),         # optional — user may send no text
    image:     UploadFile = File(...),
):
    """
    Text + image messages.
    Backend team calls this when a WhatsApp message contains an image.

    Form fields:
        user_id:   str   — sender ID
        timestamp: str   — ISO 8601
        raw_text:  str   — message text (can be empty if image only)
        image:     file  — the image attachment (JPEG/PNG)

    WhatsApp delivers images as separate media objects.
    The backend team fetches the media bytes and posts them here.
    """
    # Validate image type
    if image.content_type not in ("image/jpeg", "image/png",
                                   "image/webp", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {image.content_type}. "
                   f"Use JPEG or PNG."
        )

    image_bytes = await image.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Image file is empty")

    reply = layer.process_message(
        raw_text    = raw_text,
        user_id     = user_id,
        timestamp   = timestamp,
        image_bytes = image_bytes,
    )
    return MessageResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "service":   "Market Price AI Layer",
        "version":   "1.0.0",
        "endpoints": {
            "text only":   "POST /process",
            "with image":  "POST /process/image",
            "health":      "GET  /health",
            "docs":        "GET  /docs",
        }
    }


# ── /parse endpoint (for web app) ────────────────────────────────────────────
class ParseRequest(BaseModel):
    message: str

class ParseResponse(BaseModel):
    intent:     str
    product:    Optional[str]
    unit:       Optional[str]
    location:   Optional[str]
    price:      Optional[float]
    quantity:   Optional[float]
    confidence: float


@app.post("/parse", response_model=ParseResponse)
def parse_message(request: ParseRequest):
    """
    Returns structured parsed intent — no reply string generated.
    Backend team uses this to query their DB and build their own response.

    Used by the web app. Source should be stored as 'web' on the backend.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    # Stage 1 — preprocess
    preprocessed = layer.preprocessor.process(request.message)

    # Stage 2 — NLU parse
    parsed = layer.parser.parse(preprocessed.cleaned)

    return ParseResponse(
        intent     = parsed.intent,
        product    = parsed.product,
        unit       = parsed.unit,
        location   = parsed.location,
        price      = parsed.price,
        quantity   = parsed.quantity,
        confidence = parsed.confidence,
    )