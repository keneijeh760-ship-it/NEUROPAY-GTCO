# Market Price AI Layer — Backend Integration Guide

This README explains how the backend team should connect to the ML/NLU layer for the web-based market price app.

The ML layer takes a user's natural-language market message, such as:

```txt
how much is garri in yaba
```

and converts it into structured data:

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

The backend can then use this structured response to query the database, save price submissions, or return a fallback response.

---

## 1. What the ML Layer Does

The ML layer is responsible for:

1. Cleaning/preprocessing the user's text.
2. Detecting the user's intent.
3. Extracting market entities such as product, location, price, unit, and quantity.
4. Returning structured JSON to the backend.

The backend should not directly use the ML model files. It should call the FastAPI endpoint.

---

## 2. Main Endpoint for the Web App

For the web app, the backend should use:

```txt
POST /parse
```

This endpoint returns structured intent/entity data only. It does not generate a final user-facing reply.

### Request Body

```json
{
  "message": "how much is garri in yaba"
}
```

### Response Body

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

---

## 3. Possible Intents

The `intent` field tells the backend what the user is trying to do.

| Intent | Meaning | Backend Action |
|---|---|---|
| `QUERY` | User is asking for the price of a product in a location. | Search the database using `product` and `location`. |
| `SUBMIT_PRICE` | User is submitting a price they saw or paid. | Save `product`, `location`, `price`, and metadata to the database. |
| `GREETING` | User sent a greeting. | Return a welcome/help message. |
| `UNKNOWN` | The ML layer could not understand the message. | Ask the user to rephrase. |

---

## 4. Response Fields

| Field | Type | Description |
|---|---|---|
| `intent` | string | One of `QUERY`, `SUBMIT_PRICE`, `GREETING`, or `UNKNOWN`. |
| `product` | string or null | Product extracted from the message, e.g. `garri`, `tomatoes`, `rice`. |
| `unit` | string or null | Unit if detected, e.g. `basket`, `kg`, `mudu`. May be null for now. |
| `location` | string or null | Market/location extracted from the message, e.g. `yaba`, `mile 12`, `wuse market`. |
| `price` | number or null | Price extracted from the message. Mostly present for `SUBMIT_PRICE`. |
| `quantity` | number or null | Quantity if detected. May be null for now. |
| `confidence` | number | ML confidence score between `0.0` and `1.0`. |

---

## 5. Backend Logic

### If intent is `QUERY`

Example user message:

```txt
how much is garri in yaba
```

ML response:

```json
{
  "intent": "QUERY",
  "product": "garri",
  "location": "yaba",
  "price": null,
  "confidence": 0.9
}
```

Backend should:

1. Check that `product` and `location` are not null.
2. Search the price database for matching product/location.
3. Return latest, average, median, or recommended price to the frontend.
4. If no data exists, return a "no data yet" response and invite the user to submit a price.

Pseudo-code:

```python
if parsed["intent"] == "QUERY":
    product = parsed["product"]
    location = parsed["location"]

    if not product or not location:
        return "Please include both product and market/location."

    result = find_price(product, location)

    if result:
        return f"{product} in {location} is around ₦{result.average_price}"
    else:
        return f"No price data yet for {product} in {location}."
```

---

### If intent is `SUBMIT_PRICE`

Example user message:

```txt
i buy tomatoes for 15000 at mile 12
```

ML response:

```json
{
  "intent": "SUBMIT_PRICE",
  "product": "tomatoes",
  "location": "mile 12",
  "price": 15000.0,
  "unit": null,
  "quantity": null,
  "confidence": 0.9
}
```

Backend should:

1. Check that `product`, `location`, and `price` are present.
2. Save the submitted price to the database.
3. Store `source` as `"web"`.
4. Return a success message to the frontend.

Suggested database record:

```json
{
  "raw_message": "i buy tomatoes for 15000 at mile 12",
  "product": "tomatoes",
  "location": "mile 12",
  "price": 15000.0,
  "unit": null,
  "quantity": null,
  "confidence": 0.9,
  "source": "web",
  "created_at": "2026-05-15T12:00:00Z"
}
```

Pseudo-code:

```python
if parsed["intent"] == "SUBMIT_PRICE":
    if not parsed["product"] or not parsed["location"] or parsed["price"] is None:
        return "Please include product, market/location, and price."

    save_price_report(
        raw_message=message,
        product=parsed["product"],
        location=parsed["location"],
        price=parsed["price"],
        unit=parsed["unit"],
        quantity=parsed["quantity"],
        confidence=parsed["confidence"],
        source="web"
    )

    return "Price submitted successfully."
```

---

### If intent is `GREETING`

Example:

```txt
good morning
```

Backend should return a simple onboarding/help message:

```txt
Welcome! Ask for a market price like: "how much is garri in yaba"
```

---

### If intent is `UNKNOWN`

Example:

```txt
send me the link
```

Backend should ask the user to rephrase:

```txt
I could not understand that. Try something like: "how much is rice in wuse market"
```

---

## 6. Confidence Gate

The ML layer returns a `confidence` score.

Recommended backend behavior:

| Confidence | Suggested Action |
|---|---|
| `>= 0.65` | Trust the parse and continue. |
| `< 0.65` | Ask the user to clarify, unless required fields were clearly extracted. |

Example:

```python
if parsed["confidence"] < 0.65 and not parsed["product"]:
    return "Please rephrase your message with the product and location."
```

Note: Some rule-based overrides may return `0.9` confidence for obvious messages like:

```txt
how much is garri in yaba
```

---

## 7. Running the ML API Locally

From the project root:

```powershell
$env:USE_MOCK_PARSER="false"
$env:INTENT_MODEL_DIR="./normalization/models/intent-model"
$env:NER_MODEL_DIR="./normalization/models/ner-model"
$env:PYTHONPATH="."
uvicorn median.api:app --reload
```

If `api.py` is in the project root instead of the `median` folder, run:

```powershell
uvicorn api:app --reload
```

If you are not sure where `api.py` is, run:

```powershell
Get-ChildItem -Recurse -Filter api.py
```

Then use the correct module path:

| Location of `api.py` | Uvicorn command |
|---|---|
| `./api.py` | `uvicorn api:app --reload` |
| `./median/api.py` | `uvicorn median.api:app --reload` |

---

## 8. Testing in FastAPI Docs

After starting the server, open:

```txt
http://127.0.0.1:8000/docs
```

Go to:

```txt
POST /parse
```

Click **Try it out** and test:

```json
{
  "message": "how much is garri in yaba"
}
```

Expected response:

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

Test another one:

```json
{
  "message": "i buy tomatoes for 15000 at mile 12"
}
```

Expected response:

```json
{
  "intent": "SUBMIT_PRICE",
  "product": "tomatoes",
  "unit": null,
  "location": "mile 12",
  "price": 15000.0,
  "quantity": null,
  "confidence": 0.9
}
```

---

## 9. Backend Calling the ML API

### Python Example

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/parse",
    json={"message": "how much is garri in yaba"}
)

parsed = response.json()
print(parsed)
```

### JavaScript/TypeScript Example

```ts
const response = await fetch("http://127.0.0.1:8000/parse", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    message: "how much is garri in yaba"
  })
});

const parsed = await response.json();
console.log(parsed);
```

### Spring Boot WebClient Example

```java
WebClient client = WebClient.create("http://127.0.0.1:8000");

Map<String, String> request = Map.of(
    "message", "how much is garri in yaba"
);

Mono<ParseResponse> response = client.post()
    .uri("/parse")
    .bodyValue(request)
    .retrieve()
    .bodyToMono(ParseResponse.class);
```

Example DTO:

```java
public class ParseResponse {
    public String intent;
    public String product;
    public String unit;
    public String location;
    public Double price;
    public Double quantity;
    public Double confidence;
}
```

---

## 10. Other API Endpoints

The API may also expose these endpoints:

| Endpoint | Method | Purpose | Use for Web App? |
|---|---|---|---|
| `/parse` | `POST` | Returns structured intent/entities. | Yes, main endpoint. |
| `/process` | `POST` | Returns a generated reply string after running the full AI layer. | Optional. Use only if backend wants ML layer to generate final text. |
| `/process/image` | `POST` | Accepts image upload and text. | Not needed unless image-based product detection is enabled. |
| `/health` | `GET` | Health check. | Useful for deployment checks. |
| `/` | `GET` | Service info. | Optional. |
| `/docs` | `GET` | FastAPI Swagger UI. | Useful for testing. |

---

## 11. `/process` Endpoint

This endpoint returns a final reply string instead of structured JSON.

### Request

```json
{
  "raw_text": "how much is garri in yaba",
  "user_id": "user-123",
  "timestamp": "2026-05-15T12:00:00Z"
}
```

### Response

```json
{
  "reply": "No price info for garri for yaba in our system."
}
```

Use `/process` only if the backend wants the ML layer to generate the final user-facing response.

For the current web app, `/parse` is recommended because it gives the backend more control.

---

## 12. `/process/image` Endpoint

This endpoint is for text + image input.

It expects multipart form data:

| Field | Type | Required |
|---|---|---|
| `user_id` | string | Yes |
| `timestamp` | string | Yes |
| `raw_text` | string | No |
| `image` | file | Yes |

This is optional and not needed for the current web flow unless image classification is enabled.

---

## 13. `/health` Endpoint

Use this to check whether the ML API is running.

```txt
GET /health
```

Expected response:

```json
{
  "status": "ok"
}
```

---

## 14. Model Files Required

Do not delete these folders:

```txt
normalization/models/intent-model
normalization/models/ner-model
```

The API must be started with:

```powershell
$env:INTENT_MODEL_DIR="./normalization/models/intent-model"
$env:NER_MODEL_DIR="./normalization/models/ner-model"
```

---

## 15. Common Errors

### Error: `Could not import module "api"`

This means your `uvicorn` command is pointing to the wrong module.

Find `api.py`:

```powershell
Get-ChildItem -Recurse -Filter api.py
```

If it is inside `median`, use:

```powershell
uvicorn median.api:app --reload
```

If it is in the root folder, use:

```powershell
uvicorn api:app --reload
```

---

### Error: `No module named 'nlu'`

Run from the project root and set:

```powershell
$env:PYTHONPATH="."
```

Then start the server again.

---

### Wrong predictions from `/parse`

Make sure mock mode is off:

```powershell
$env:USE_MOCK_PARSER="false"
```

If mock mode is on, the app will use the mock parser instead of the trained AfroXLMR models.

---

## 16. Recommended Backend Integration Flow

```txt
User types message in web app
        ↓
Frontend sends message to backend
        ↓
Backend sends message to ML API: POST /parse
        ↓
ML API returns intent + entities
        ↓
Backend applies business logic:
    - QUERY → search DB
    - SUBMIT_PRICE → save DB
    - GREETING → welcome message
    - UNKNOWN → clarification message
        ↓
Backend returns final response to frontend
```

---

## 17. Example End-to-End Flow

User enters:

```txt
how much is garri in yaba
```

Backend calls:

```txt
POST http://127.0.0.1:8000/parse
```

Request:

```json
{
  "message": "how much is garri in yaba"
}
```

ML response:

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

Backend searches database:

```sql
SELECT *
FROM price_reports
WHERE product = 'garri'
  AND location = 'yaba'
ORDER BY created_at DESC;
```

Backend returns to frontend:

```txt
Garri in Yaba is currently around ₦X based on recent reports.
```

---

## 18. Final Notes for Backend Team

- Use `/parse` for the web app.
- Store submitted prices with `source = "web"`.
- Do not call the model files directly from backend.
- Run the ML API as a separate service.
- Make sure `USE_MOCK_PARSER=false` when testing the real trained model.
- Use `product + location` for price queries.
- Use `product + location + price` for price submissions.
- Ask for clarification when important fields are missing.
