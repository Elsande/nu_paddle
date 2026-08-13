"""fusion.py — Fusion Layer (rule-based, tanpa LLM).

Menggabungkan hasil ekstraksi beberapa sumber (qwen3-vl, glm, nu-extract) menjadi
satu nilai per field, sesuai plan.md pasal 2.5 & 4.4:

- Field yang hanya punya satu nilai unik -> ``consensus``.
- Field dengan nilai BEDA -> pilih berdasarkan prioritas jenis field
  (numeric / id_code / free_text), dengan opsi override bila selisih
  confidence melampaui ``CONFIDENCE_OVERRIDE_THRESHOLD``.

Catatan: paddleocr-vl 1.6 tidak masuk fusion per-field — ia memberi baris OCR
yang dipakai untuk koreksi field kode/nomor lewat ``validation.correct_codes_from_ocr``.
"""

from __future__ import annotations

from typing import Any

# Selisih confidence minimal agar sumber ber-conf tertinggi boleh "mengalahkan"
# sumber prioritas utama (plan.md 4.4).
CONFIDENCE_OVERRIDE_THRESHOLD = 0.3

# Petunjuk nama field -> jenis field.
_ID_CODE_HINTS = ("number", "code", "reference", "npwp", "tax_number", "cf_code")
_NUMERIC_HINTS = (
    "amount", "total", "subtotal", "tax", "price", "quantity",
    "value", "items", "date", "tanggal", "tgl",
)
_FREE_TEXT_HINTS = ("name", "address", "description", "catatan", "materai")

# Prioritas sumber per jenis field (plan.md 4.4). paddleocr-vl & nu-extract
# tidak dimasukkan: paddleocr-vl lewat correct_codes_from_ocr, nu-extract
# sementara off.
FIELD_PRIORITY: dict[str, list[str]] = {
    "numeric": ["qwen3", "glm"],
    "id_code": ["qwen3", "glm"],
    "free_text": ["qwen3", "glm"],
}


def classify_field(field_name: str) -> str:
    """Klasifikasikan jenis field: ``numeric`` | ``id_code`` | ``free_text``."""
    k = field_name.lower()
    if any(h in k for h in _ID_CODE_HINTS):
        return "id_code"
    if any(h in k for h in _NUMERIC_HINTS):
        return "numeric"
    if any(h in k for h in _FREE_TEXT_HINTS):
        return "free_text"
    return "free_text"


def _norm(value: Any) -> str:
    """Normalisasi nilai untuk perbandingan (lowercase, hilangkan spasi ganda)."""
    return " ".join(str(value).strip().lower().split()) if value is not None else ""


def _avg(values: list[float | None]) -> float:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def fuse_sources(sources: list[dict]) -> dict[str, dict]:
    """Gabungkan hasil beberapa sumber.

    Parameters
    ----------
    sources : list[dict]
        Tiap elemen: ``{"name": str, "fields": dict, "confidences": dict}``.

    Returns
    -------
    dict
        ``{field: {"value", "confidence", "source_used", "had_conflict",
        "alternatives"?}}``.
    """
    per_field: dict[str, list[tuple[str, Any, float | None]]] = {}
    for src in sources:
        name = src.get("name", "?")
        fields = src.get("fields") or {}
        confs = src.get("confidences") or {}
        for key, value in fields.items():
            per_field.setdefault(key, []).append((name, value, confs.get(key)))

    fused: dict[str, dict] = {}
    for key, entries in per_field.items():
        non_null = [(n, v, c) for (n, v, c) in entries if v is not None and str(v).strip() != ""]
        if not non_null:
            fused[key] = {
                "value": None,
                "confidence": 0.0,
                "source_used": "none",
                "had_conflict": False,
            }
            continue

        unique = {_norm(v) for _, v, _ in non_null}
        if len(unique) <= 1:
            fused[key] = {
                "value": non_null[0][1],
                "confidence": _avg([c for _, _, c in non_null]),
                "source_used": "consensus",
                "had_conflict": False,
            }
            continue

        # Konflik: pilih berdasar prioritas jenis field.
        ftype = classify_field(key)
        avail_names = {n for n, _, _ in non_null}
        order = [s for s in FIELD_PRIORITY.get(ftype, []) if s in avail_names]
        if not order:
            order = [n for n, _, _ in non_null]
        default = order[0]

        best = max(non_null, key=lambda e: e[2] if e[2] is not None else 0.0)
        best_src, best_conf = best[0], best[2]
        winner = default
        if best_src in order and best_src != default:
            default_conf = next((c for n, _, c in non_null if n == default), None) or 0.0
            if (best_conf or 0.0) - default_conf > CONFIDENCE_OVERRIDE_THRESHOLD:
                winner = best_src

        fused[key] = {
            "value": next(v for n, v, _ in non_null if n == winner),
            "confidence": next(c for n, _, c in non_null if n == winner) or 0.0,
            "source_used": winner,
            "had_conflict": True,
            "alternatives": {n: v for n, v, _ in non_null if n != winner},
        }
    return fused
