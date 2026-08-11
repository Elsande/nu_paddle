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
