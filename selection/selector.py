"""Selektor selective inference — kapan teks PaddleOCR ikut dikirim ke NuExtract.

Prinsip:
- NuExtract3-GGUF (role="main") SELALU dijalankan dengan gambar dokumen.
- PaddleOCR (role="support") memberi teks pendukung; TIDAK ADA aturan yang
  membatasi: selama OCR menghasilkan teks, teks tersebut SELALU dikirim ke
  NuExtract sebagai konteks. Satu-satunya pengecualian: OCR kosong/gagal.

Catatan: confidence tiap baris tetap disimpan di hasil (``ocr.lines``) sebagai
informasi, bukan sebagai aturan pemblokiran.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Heuristik tulisan tangan / coretan (proxy murah, tanpa API).
# Port dari AI-Document selection/selector.py aturan 5, disesuaikan dengan
# baris OCR PaddleOCR (text + bbox + confidence) — bukan parsing_blocks.
# ---------------------------------------------------------------------------
# Aktifkan heuristik tulisan tangan/coretan (eksperimental).
ENABLE_HANDWRITING_HEURISTIC = True
# Baris teks dengan panjang di rentang ini dicek sebagai kandidat coretan.
SUSPICIOUS_MIN_CHARS = 2
SUSPICIOUS_MAX_CHARS = 40
# Rasio karakter alfanumerik di bawah ini => "banyak simbol" (mencurigakan).
SUSPICIOUS_MAX_CLEAN_RATIO = 0.55
# Confidence OCR di bawah ini dianggap tidak andal (kemungkinan tulisan
# tangan). Fragmen logo/header ber-conf ~0.4-0.5 TIDAK boleh memicu — tulisan
# tangan asli umumnya memberi confidence jauh di bawah ini.
LOW_CONFIDENCE_THRESHOLD = 0.35

# Karakter "bersih" utk heuristik: alfanumerik + tanda baca umum yang wajar di
# dokumen cetak (koma, titik, persen, kurung, dsb.). Karakter di luar ini
# (simbol/unicode aneh dari tulisan tangan yang garbled) dihitung "kotor".
_CLEAN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " .,:;()[]/\\-_%+=@"
)


def _clean_ratio(text: str) -> float:
    """Fraksi karakter "bersih" (alfanumerik + tanda baca umum) dalam teks."""
    if not text:
        return 1.0
    clean = sum(1 for c in text if c in _CLEAN_CHARS)
    return clean / len(text)


def _has_alpha(text: str) -> bool:
    """Setidaknya ada satu huruf alfabet (memfilter angka/tanggal murni)."""
    return any(c.isalpha() for c in text)


def detect_handwriting_heuristic(lines: list[dict] | None) -> list[dict]:
    """Baris OCR yang mencurigakan (proxy tulisan tangan/coretan).

    Baris dianggap mencurigakan bila:
    - teks pendek (2-40 char) + rasio simbol tinggi + mengandung huruf, ATAU
    - confidence rendah (< ``LOW_CONFIDENCE_THRESHOLD``).
    Mengembalikan daftar baris mencurigakan (kosong bila dokumen bersih).
    """
    if not ENABLE_HANDWRITING_HEURISTIC or not lines:
        return []
    suspicious: list[dict] = []
    for ln in lines:
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        conf = float(ln.get("confidence") or 0.0)
        low_conf = conf < LOW_CONFIDENCE_THRESHOLD
        symbolic = (
            SUSPICIOUS_MIN_CHARS <= len(text) <= SUSPICIOUS_MAX_CHARS
            and _clean_ratio(text) < SUSPICIOUS_MAX_CLEAN_RATIO
            and _has_alpha(text)
        )
        if low_conf or symbolic:
            suspicious.append(ln)
    return suspicious


@dataclass
class OCRDecision:
    """Hasil keputusan selektor untuk satu dokumen."""

    include_text: bool = False
    reason: str = ""
    filtered_text: str = ""
    stats: dict = field(default_factory=dict)


def evaluate(
    filtered_text: str,
    lines: list[dict] | None = None,
    kept_lines: list[dict] | None = None,
) -> OCRDecision:
    """Putuskan apakah teks OCR pendukung dikirim ke NuExtract.

    Tidak ada aturan pemblokiran: teks selalu disertakan selama tidak kosong.
    """
    kept_lines = kept_lines or []
    lines = lines or kept_lines

    stats = {
        "total_lines": len(lines),
        "kept_lines": len(kept_lines),
    }
    if kept_lines:
        stats["avg_confidence"] = round(
            sum(ln.get("confidence", 0.0) for ln in kept_lines) / len(kept_lines), 3
        )

    text = (filtered_text or "").strip()
    if not text:
        return OCRDecision(False, "OCR gagal / teks kosong", "", stats)

    stats["word_count"] = len(text.split())
    return OCRDecision(True, "teks OCR disertakan (tanpa aturan pemblokiran)", text, stats)
