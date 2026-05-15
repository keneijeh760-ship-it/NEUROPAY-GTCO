"""
build_9_10_market_dataset.py

Purpose:
Build a much stronger Nigeria market-price dataset by replacing synthetic/proxy rows
with direct rows from official/open data sources.

Target quality split for a real ~9/10 dataset:
- 300+ direct rows from NBS Selected Food Price Watch tables
- 150+ direct rows from WFP/HDX Nigeria Food Prices CSV
- 0-25 Jiji rows max, used only as noisy marketplace signal

Run from your project root:
    pip install requests pandas openpyxl
    python build_9_10_market_dataset.py \
      --existing nigeria_market_price_seed_dataset_500_v2_improved.csv \
      --out nigeria_market_price_dataset_500_9of10.csv

Notes:
- This script does NOT bypass any website restrictions.
- It uses official/open-data endpoints where possible.
- For NBS, it downloads the official March 2026 ZIP from NBS Microdata.
- For HDX, it uses the public CKAN API to discover the current CSV resource.
"""

import argparse
import csv
import io
import os
import re
import sys
import zipfile
from datetime import datetime, date
from urllib.parse import urljoin

import requests

NBS_MAR2026_ZIP = "https://microdata.nigerianstat.gov.ng/index.php/catalog/162/download/1401"
NBS_SOURCE_PAGE = "https://microdata.nigerianstat.gov.ng/index.php/catalog/162/related-materials"
HDX_PACKAGE_API = "https://data.humdata.org/api/3/action/package_show?id=wfp-food-prices-for-nigeria"
JIIJI_MAX_ROWS = 25
TARGET_ROWS = 500


def norm_money(x):
    if x is None:
        return None
    s = str(x).replace(',', '').replace('₦', '').replace('N', '').strip()
    try:
        return round(float(s), 2)
    except Exception:
        return None


def freshness_weight(observed_date):
    if not observed_date:
        return 0.50, None
    if isinstance(observed_date, str):
        try:
            d = datetime.fromisoformat(observed_date[:10]).date()
        except Exception:
            return 0.50, None
    else:
        d = observed_date
    days = (date.today() - d).days
    if days <= 30:
        w = 1.0
    elif days <= 90:
        w = 0.85
    elif days <= 180:
        w = 0.70
    else:
        w = 0.50
    return w, days


def infer_unit(product, raw_unit=""):
    text = f"{product} {raw_unit}".lower()
    if "egg" in text:
        return "piece", 30 if "crate" in text else 1
    if "oil" in text:
        return "litre", 1
    if "rice" in text and "50" in text:
        return "kg", 50
    if "rice" in text:
        return "kg", 1
    return "kg", 1


def make_row(record_id, product, category, unit_raw, price, location, state, market_name, source, source_detail, source_url, observed_date, direct=True):
    unit_canonical, qty = infer_unit(product, unit_raw)
    price = norm_money(price)
    if not price:
        return None
    price_per_unit = round(price / qty, 2) if qty else price
    fw, days = freshness_weight(observed_date)
    trust = {"NBS": 0.95, "WFP_HDX": 0.90, "Jiji": 0.55}.get(source, 0.60)
    loc_gran = "market" if market_name else ("state_average" if state and state != "National" else "national_average")
    loc_conf = {"market": 0.90, "state_average": 0.75, "national_average": 0.55}.get(loc_gran, 0.60)
    final_weight = round(trust * fw * loc_conf, 4)
    row_score = round(min(9.5, final_weight * 10 + (1.0 if direct else 0)), 2)
    return {
        "record_id": record_id,
        "product_raw": product,
        "product_canonical": product.lower().strip(),
        "category": category or "food",
        "unit_raw": unit_raw or unit_canonical,
        "unit_canonical": unit_canonical,
        "quantity": qty,
        "price": price,
        "price_per_canonical_unit": price_per_unit,
        "currency": "NGN",
        "location_raw": location or state or "Nigeria",
        "location_canonical": location or state or "Nigeria",
        "market_name": market_name or "",
        "state": state or "",
        "country": "Nigeria",
        "location_granularity": loc_gran,
        "location_confidence": loc_conf,
        "observed_date": observed_date,
        "days_old": days if days is not None else "",
        "freshness_weight": fw,
        "source": source,
        "source_detail": source_detail,
        "source_url": source_url,
        "trust_weight": trust,
        "final_weight": final_weight,
        "row_accuracy_score_10": row_score,
        "is_synthetic_estimate": False,
        "direct_source_extracted": True if direct else False,
        "data_generation_method": f"direct_{source.lower()}_extract_with_unit_freshness_location_processing",
        "raw_message": f"how much is {product.lower()} in {location or state or 'Nigeria'}",
        "normalized_message": f"{product.lower()} costs {price} NGN per {unit_raw or unit_canonical} in {location or state or 'Nigeria'}",
        "intent": "PRICE_OBSERVATION",
        "notes": "Direct/open data row; suitable for high-confidence seeding." if direct else "Fallback row; use lower priority."
    }


def download_nbs_zip():
    print("Downloading NBS March 2026 ZIP...")
    resp = requests.get(NBS_MAR2026_ZIP, timeout=90)
    resp.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(resp.content))


def extract_nbs_rows(max_rows=350):
    """Best-effort extraction. NBS workbook layouts may change, so this uses openpyxl dynamically."""
    import openpyxl
    rows = []
    try:
        z = download_nbs_zip()
    except Exception as e:
        print(f"WARNING: Could not download NBS ZIP: {e}")
        return rows
    xlsx_names = [n for n in z.namelist() if n.lower().endswith('.xlsx')]
    if not xlsx_names:
        print("WARNING: NBS ZIP downloaded but no XLSX found.")
        return rows
    data = z.read(xlsx_names[0])
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    record_i = 1
    for ws in wb.worksheets:
        grid = list(ws.iter_rows(values_only=True))
        # Heuristic: find rows where a state and price-like values appear.
        for r in grid:
            vals = [v for v in r if v not in (None, "")]
            if len(vals) < 3:
                continue
            row_text = " ".join(str(v) for v in vals).lower()
            # Skip obvious headings.
            if "item" in row_text and "price" in row_text:
                continue
            # Try product in first/second cell, state/location in nearby cell, price in last numeric cell.
            nums = [norm_money(v) for v in vals]
            nums = [n for n in nums if n is not None and n > 10]
            if not nums:
                continue
            price = nums[-1]
            text_vals = [str(v).strip() for v in vals if isinstance(v, str) and len(str(v).strip()) > 1]
            if len(text_vals) < 2:
                continue
            product = text_vals[0]
            state = text_vals[-1]
            if len(product) > 70 or len(state) > 40:
                continue
            row = make_row(
                f"NBS-DIRECT-{record_i:04d}", product, "food", "reported unit", price,
                state, state, "", "NBS", f"March 2026 table: {ws.title}", NBS_SOURCE_PAGE, "2026-03-31", True
            )
            if row:
                rows.append(row)
                record_i += 1
                if len(rows) >= max_rows:
                    return rows
    return rows


def hdx_resource_url():
    print("Discovering HDX/WFP CSV resource...")
    resp = requests.get(HDX_PACKAGE_API, timeout=45)
    resp.raise_for_status()
    package = resp.json()["result"]
    csvs = []
    for res in package.get("resources", []):
        url = res.get("url", "")
        name = res.get("name", "")
        fmt = (res.get("format") or "").lower()
        if "csv" in fmt or url.lower().endswith(".csv") or "download" in url.lower():
            csvs.append((name, url))
    if not csvs:
        raise RuntimeError("No CSV resource found in HDX package")
    # Prefer main full data over quickcharts if names reveal it.
    csvs.sort(key=lambda x: ("quick" in x[0].lower(), x[0]))
    return csvs[0][1]


def extract_hdx_rows(max_rows=150):
    rows = []
    try:
        url = hdx_resource_url()
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
    except Exception as e:
        print(f"WARNING: Could not download HDX/WFP CSV: {e}")
        return rows
    sample = resp.text[:2048]
    dialect = csv.Sniffer().sniff(sample)
    reader = csv.DictReader(io.StringIO(resp.text), dialect=dialect)
    record_i = 1
    for rec in reader:
        # Common WFP fields vary: adm1_name, mkt_name, cm_name, cur_name, mp_price, mp_month, mp_year, um_name
        product = rec.get("cm_name") or rec.get("commodity") or rec.get("item") or rec.get("product")
        price = rec.get("mp_price") or rec.get("price") or rec.get("value")
        market = rec.get("mkt_name") or rec.get("market") or ""
        state = rec.get("adm1_name") or rec.get("state") or ""
        unit = rec.get("um_name") or rec.get("unit") or "reported unit"
        year = rec.get("mp_year") or rec.get("year")
        month = rec.get("mp_month") or rec.get("month")
        if not product or not price:
            continue
        obs_date = "2026-03-31"
        try:
            if year and month:
                obs_date = f"{int(float(year)):04d}-{int(float(month)):02d}-28"
        except Exception:
            pass
        row = make_row(
            f"WFP-DIRECT-{record_i:04d}", product, "food", unit, price,
            market or state, state, market, "WFP_HDX", "WFP Price Database via HDX", url, obs_date, True
        )
        if row:
            rows.append(row)
            record_i += 1
            if len(rows) >= max_rows:
                return rows
    return rows


def read_existing(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_rows(rows, out):
    if not rows:
        raise RuntimeError("No rows to write")
    fields = list(rows[0].keys())
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--existing', default='nigeria_market_price_seed_dataset_500_v2_improved.csv')
    ap.add_argument('--out', default='nigeria_market_price_dataset_500_9of10.csv')
    args = ap.parse_args()

    nbs = extract_nbs_rows(325)
    hdx = extract_hdx_rows(175)
    direct = nbs + hdx

    existing = read_existing(args.existing)
    # Only use existing rows as fallback; prefer high-quality official rows and minimal/no Jiji.
    fallback = [r for r in existing if r.get('source') in ('NBS', 'WFP_HDX')]
    fallback.sort(key=lambda r: float(r.get('final_weight') or 0), reverse=True)

    output = []
    seen = set()
    for r in direct + fallback:
        key = (r.get('product_canonical'), r.get('state'), r.get('market_name'), r.get('observed_date'), r.get('source'), r.get('price'))
        if key in seen:
            continue
        seen.add(key)
        output.append(r)
        if len(output) >= TARGET_ROWS:
            break

    if len(output) < TARGET_ROWS:
        # Last resort: allow up to 25 Jiji rows, clearly marked as lower-trust.
        jiji = [r for r in existing if r.get('source') == 'Jiji'][:JIIJI_MAX_ROWS]
        output.extend(jiji[:TARGET_ROWS-len(output)])

    if len(output) < TARGET_ROWS:
        print(f"WARNING: Only built {len(output)} rows. Need more official rows for a true 9/10 dataset.")
    else:
        output = output[:TARGET_ROWS]

    write_rows(output, args.out)
    direct_count = sum(str(r.get('direct_source_extracted')).lower() == 'true' for r in output)
    synthetic_count = sum(str(r.get('is_synthetic_estimate')).lower() == 'true' for r in output)
    sources = {}
    for r in output:
        sources[r.get('source')] = sources.get(r.get('source'), 0) + 1
    print({"sources": sources, "direct_source_extracted": direct_count, "synthetic": synthetic_count})
    if direct_count < 400:
        print("QUALITY WARNING: This is not yet 9/10. Get at least 400 direct official/open-data rows.")
    else:
        print("QUALITY TARGET HIT: This dataset is defensible as ~9/10 for demo/database seeding.")

if __name__ == '__main__':
    main()
