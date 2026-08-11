"""
validation.py
=============
Normalisasi nilai field + validasi ringkas hasil ekstraksi NuExtract3-GGUF.

Diadaptasi dari notebook ``doc-validation/cek_nuextract3_gguf.ipynb``
(normalisasi tanggal/currency + aturan PASSED/FAILED).
"""

from __future__ import annotations

import difflib
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# NORMALISASI
# ---------------------------------------------------------------------------
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR", "Rp": "IDR"}


def _normalise_date(value: str) -> str:
    value = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", value, flags=re.IGNORECASE)
    value = " ".join(str(value).split())
    formats = (
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d",
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%d-%b-%Y",
        "%d-%b-%y", "%Y%m%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return str(value)


def _normalise_amount(value) -> str:
    """Bersihkan nilai uang/angka menjadi string angka sederhana."""
    s = str(value)
    for symbol in _CURRENCY_SYMBOLS:
        s = s.replace(symbol, "")
    s = s.strip()
    if re.search(r"\d\.\d{3},\d{2}", s):          # 1.234.567,89
        s = s.replace(".", "").replace(",", ".")
    elif re.search(r"\d,\d{3}\.\d{2}", s):        # 1,234,567.89
        s = s.replace(",", "")
    else:
        s = s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if m:
        num = float(m.group())
        return str(int(num)) if num.is_integer() else str(num)
    return str(value).strip()


def normalise_field(field_name: str, raw_value) -> str:
    if raw_value is None:
        return ""
    if re.search(r"(?i)(date|issued|due|expiry|issued|start|end|tanggal|tgl)", field_name):
        return _normalise_date(str(raw_value))
    if any(kw in field_name.lower() for kw in ("amount", "total", "subtotal", "tax", "price", "fee", "cost", "value", "sum")):
        return _normalise_amount(raw_value)
    if isinstance(raw_value, bool):
        return "true" if raw_value else "false"
    return str(raw_value).strip()


def normalise_fields(fields: dict) -> dict:
    norm = {k: normalise_field(k, v) for k, v in fields.items() if v is not None and v != ""}
    # Jika due_date sama persis dengan po_date, itu duplikat (model menyalin
    # tanggal PO) — dokumen tidak punya nilai jatuh tempo eksplisit.
    if norm.get("due_date") and norm.get("due_date") == norm.get("po_date"):
        norm["due_date"] = ""
    return {k: v for k, v in norm.items() if v != ""}


# ---------------------------------------------------------------------------
# KOREKSI FIELD KODE/NOMOR DARI OCR
# ---------------------------------------------------------------------------
_CODE_HINTS = ("number", "code", "reference")
# Panjang minimum baris OCR & confidence minimum agar layak dijadikan sumber.
_OCR_MIN_LEN = 8
_OCR_MIN_CONF = 0.95
_OCR_CUTOFF = 0.92


def _is_code_field(key: str) -> bool:
    k = key.lower()
    return any(h in k for h in _CODE_HINTS)


def _find_npwp_from_ocr(ocr_lines: list[dict]) -> str | None:
    """Ambil nomor NPWP dari baris OCR berlabel 'NPWP' (yang paling andal)."""
    for ln in ocr_lines:
        text = ln.get("text") or ""
        if "NPWP" in text.upper():
            digits = re.sub(r"\D", "", text)
            if 12 <= len(digits) <= 20:
                return digits
    return None


def correct_codes_from_ocr(fields: dict, ocr_lines: list[dict]) -> dict:
    """Perbaiki field kode/nomor yang salah, memakai baris OCR ber-conf tinggi.

    Alasan: PaddleOCR sering lebih akurat untuk kode/nomor daripada NuExtract
    yang membaca ulang gambar beresolusi rendah (mis. ``POISSP-...`` vs
    ``POBSP-...``). Hanya diterapkan pada field bertipe kode, dan hanya bila
    baris OCR hampir identik (edit distance kecil) dengan confidence >= 0.95 —
    jadi nilai model yang sudah benar TIDAK dirusak oleh OCR.
    """
    if not fields or not ocr_lines:
        return fields

    candidates = []
    for ln in ocr_lines:
        text = re.sub(r"^[\W_]+", "", (ln.get("text") or "").strip())  # buang ":" di depan
        if len(text) >= _OCR_MIN_LEN and float(ln.get("confidence", 0.0)) >= _OCR_MIN_CONF:
            candidates.append(text)
    if not candidates:
        return fields

    out = dict(fields)
    for key, val in fields.items():
        if val is None:
            continue
        # NPWP: ambil langsung dari baris berlabel NPWP di OCR.
        if "tax_number" in key.lower():
            npwp = _find_npwp_from_ocr(ocr_lines)
            if npwp:
                out[key] = npwp
            continue
        if not _is_code_field(key):
            continue
        sval = str(val).strip()
        if not sval:
            continue
        matches = difflib.get_close_matches(sval, candidates, n=1, cutoff=_OCR_CUTOFF)
        if matches and matches[0] != sval:
            out[key] = matches[0]
    return out


# ---------------------------------------------------------------------------
# VALIDASI
# ---------------------------------------------------------------------------
REQUIRED_FIELDS: dict[str, list[str]] = {
    "invoice": ["invoice_number", "invoice_date", "supplier_name", "total_amount"],
    "purchase_order": ["po_number", "po_date", "supplier_name", "buyer_name", "total_amount"],
    "delivery_order": ["do_number", "do_date", "supplier_name", "recipient_name", "delivery_address"],
    "kwitansi": ["kwitansi_date", "company_name", "invoice_reference", "total_value"],
    "tax_invoice": ["tax_invoice_number", "invoice_date", "supplier_name", "total_amount", "tax_amount"],
}

# Field bernilai numerik yang wajib > 0.
_POSITIVE_FIELDS = ("amount", "total", "subtotal", "tax", "price", "value", "items", "quantity")


def validate_document(doc_type: str, norm_fields: dict) -> dict:
    """Validasi ringkas: field wajib + aturan angka positif & tanggal valid."""
    field_errors: list[dict] = []
    for f in REQUIRED_FIELDS.get(doc_type, []):
        if not norm_fields.get(f):
            field_errors.append({"field": f, "code": "MISSING_REQUIRED_FIELD", "message": f"Field wajib '{f}' tidak terisi"})

    rules: list[dict] = []
    # 1. positive_amounts
    positive_ok = True
    for key, value in norm_fields.items():
        if any(kw in key.lower() for kw in _POSITIVE_FIELDS):
            try:
                if float(str(value).replace(",", "")) <= 0:
                    positive_ok = False
                    break
            except (TypeError, ValueError):
                pass
    rules.append({"rule": "positive_amounts", "passed": positive_ok})

    # 2. dates_valid
    dates_ok = True
    for key, value in norm_fields.items():
        if re.search(r"(?i)(date|issued|due|expiry|start|end|tanggal|tgl)", key) and value:
            try:
                datetime.strptime(str(value), "%Y-%m-%d")
            except ValueError:
                dates_ok = False
                break
    rules.append({"rule": "dates_valid", "passed": dates_ok})

    status = "PASSED" if not field_errors and all(r["passed"] for r in rules) else "FAILED"
    return {"status": status, "field_errors": field_errors, "rules": rules}
