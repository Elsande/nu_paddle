"""VLM API — GLM-4.6V-Flash (Z.AI) untuk tulisan tangan / coretan.

Model VLM dipanggil LEWAT API OpenAI-compatible (gateway lokal perusahaan),
bukan dimuat ke mesin ini — cukup HTTP client ringan. Tugas utamanya
(role="support") menangani apa yang tidak bisa PaddleOCR:

1. Membaca TULISAN TANGAN (handwriting) di atas nilai cetak.
2. Memahami CORETAN (ciretan/cross-out): mana kata/angka yang dicoret dan mana
   PEMBETULAN yang ditulis di sebelahnya -> hasil akhir memakai nilai pembetulan.

Dipanggil SELEKTIF oleh ``run_batch`` hanya bila heuristik
(``selection.selector.detect_handwriting_heuristic``) menandai dokumen
mencurigakan, lalu dibagi jadi 3 langkah:

- ``detect_handwriting`` : konfirmasi YA/TIDAK (biaya murah, max_tokens kecil).
- ``read_handwriting``   : daftar koreksi + catatan tulisan tangan + teks bersih.
- ``review_fields``      : koreksi field JSON hasil NuExtract memakai nilai
  pembetulan (nilai asli tetap disimpan utk audit trail).

Kegagalan API tidak boleh menggagalkan dokumen: cukup di-catch dan dicatat di
``result["vlm"]["error"]`` (degrade halus, anti miss).
"""

from __future__ import annotations

import base64
import gc
import io
import json
import re
import time
from typing import Any, Optional

from config import (
    VLM_API_BASE_URL,
    VLM_API_DISABLE_THINKING,
    VLM_API_JPEG_QUALITY,
    VLM_API_KEY,
    VLM_API_MAX_NEW_TOKENS,
    VLM_API_MAX_SIDE,
    VLM_API_MODEL,
    VLM_API_READ_MAX_SIDE,
    VLM_API_TEMPERATURE,
    VLM_API_TIMEOUT,
)
from models.base import BaseExtractionModel, ModelResult
from models.nuextract_gguf_model import extract_json


# ---------------------------------------------------------------------------
# PROMPT (Bahasa Indonesia, fokus tulisan tangan + coretan)
# ---------------------------------------------------------------------------
DETECT_PROMPT = (
    "Periksa dokumen gambar ini dengan teliti. Apakah ada (1) tulisan tangan, "
    "(2) kata/angka yang dicoret (coretan/garis coret), atau (3) pembetulan "
    "tulisan tangan di samping nilai yang dicoret?\n"
    "Jawab STRICT dengan dua baris: baris pertama 'YA' atau 'TIDAK', baris "
    "kedua alasan singkat dalam satu kalimat. Bila tidak ada satu pun indikasi, "
    "jawab 'TIDAK'."
)

READ_PROMPT = (
    "Baca seluruh dokumen ini dengan teliti, termasuk teks KETIK (cetak), "
    "TULISAN TANGAN, dan CORETAN di atas teks ketik.\n"
    "Aturan interpretasi WAJIB:\n"
    "1. Bila kata teks KETIK dicoret (garis coret / tanda X / scribble) dan ada "
    "TULISAN TANGAN di dekatnya (atas/bawah/samping) -> itu PEMBETULAN: teks "
    "ketik yang dicoret DIGANTI oleh tulisan tangan tersebut. Yang dicoret "
    "SELALU teks ketik asli; tulisan tangan adalah HASIL pembetulan.\n"
    "2. Bila ada TULISAN TANGAN di sela-sela teks ketik TANPA ada kata ketik "
    "yang dicoret -> itu TAMBAHAN kata (insertion), bukan pembetulan.\n"
    "Kembalikan HANYA JSON valid dengan format:\n"
    '{"corrections": [{"crossed_out": string, "corrected": string, "location": string}], '
    '"insertions": [{"inserted": string, "location": string}], '
    '"handwritten_notes": [string], "cleaned_text": string}\n'
    "- corrections: tiap PEMBETULAN; crossed_out = teks KETIK yang dicoret "
    "(nilai asli), corrected = TULISAN TANGAN penggantinya, location = posisi "
    "(mis. 'poin III.1', 'baris 2.7.4').\n"
    "- insertions: tiap TAMBAHAN kata tulisan tangan yang TIDAK menggantikan "
    "teks ketik yang dicoret; inserted = kata tambahan, location = posisinya.\n"
    "- handwritten_notes: tulisan tangan lain yang tidak jelas fungsi/coretan "
    "tanpa pembetulan.\n"
    "- cleaned_text: seluruh teks dokumen hasil AKHIR — teks ketik dengan nilai "
    "pembetulan MENGGANTIKAN yang dicoret, dan TAMBAHAN kata diselipkan di "
    "posisinya.\n"
    "Bila tidak ada coretan, corrections = []. Bila tidak ada tambahan, "
    "insertions = []."
)

REVIEW_PROMPT = (
    "Berikut hasil ekstraksi JSON dari dokumen:\n{fields_json}\n\n"
    "Periksa ulang terhadap gambar dokumen. Bila suatu field memakai nilai yang "
    "DICORET padahal ada pembetulan tulisan tangan di sebelahnya, perbaiki "
    "memakai nilai PEMBETULAN. Jangan mengubah nilai yang sudah benar.\n"
    "Kembalikan HANYA JSON valid dengan format:\n"
    '{{"changes": [{{"field": string, "old_value": string, "new_value": string, '
    '"reason": string}}]}}\n'
    "Hanya field yang berubah; bila semuanya sudah benar, changes = []."
)


class VLMApiModel(BaseExtractionModel):
    """GLM-4.6V-Flash via API OpenAI-compatible (tulisan tangan/coretan)."""

    name = "GLM-4.6V-Flash (API)"
    role = "support"

    # max_tokens khusus untuk deteksi YA/TIDAK (GLM memakai reasoning dulu).
    _DETECT_MAX_NEW_TOKENS = 1024

    @staticmethod
    def _strip_box_tags(text: str) -> str:
        """Buang penanda <|begin_of_box|> / <|end_of_box|> dari output model."""
        return re.sub(r"<\|(?:begin|end)_of_box\|>", "", text).strip()

    def __init__(
        self,
        base_url: str = VLM_API_BASE_URL,
        api_key: str = VLM_API_KEY,
        model: str = VLM_API_MODEL,
        timeout: int = VLM_API_TIMEOUT,
        max_new_tokens: int = VLM_API_MAX_NEW_TOKENS,
        temperature: float = VLM_API_TEMPERATURE,
        disable_thinking: bool = VLM_API_DISABLE_THINKING,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._disable_thinking = disable_thinking
        self._client: Any = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI  # import lambat (model tidak dimuat lokal)

        # max_retries=1 + timeout terbatas agar error "server mati" cepat
        # terlihat, bukan menggantung berulang kali.
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=1,
        )

    def unload(self) -> None:
        self._client = None
        gc.collect()

    # -- util gambar -------------------------------------------------------
    def _encode_image(self, image_path: str, max_side: int = VLM_API_MAX_SIDE) -> str:
        """Encode gambar jadi data URI JPEG (fit sisi <= max_side)."""
        from PIL import Image

        with Image.open(image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            side = max(w, h)
            if side > max_side:
                scale = max_side / side
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=VLM_API_JPEG_QUALITY)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    # -- inference ---------------------------------------------------------
    def _chat(
        self,
        image_path: str,
        text_prompt: str,
        max_new_tokens: int | None = None,
        max_side: int = VLM_API_MAX_SIDE,
    ) -> str:
        """Satu panggilan chat completion dengan gambar + prompt. Return teks."""
        self.load()
        if self._client is None:
            raise RuntimeError("Client VLM API tidak tersedia (panggil load() dulu).")

        image_url = self._encode_image(image_path, max_side=max_side)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_new_tokens or self._max_new_tokens,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        # Coba dengan param GLM tambahan; bila gateway menolak (400), ulangi
        # tanpa param tambahan (temperature tetap dipertahankan bila aman).
        for attempt in (0, 1):
            if attempt == 0 and self._disable_thinking:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            elif attempt == 1:
                kwargs.pop("extra_body", None)
            try:
                completion = self._client.chat.completions.create(**kwargs)
                message = completion.choices[0].message
                # GLM lokal bisa mengembalikan jawaban di reasoning_content
                # dengan content kosong (terpotong). Fallback ke reasoning.
                content = (message.content or "").strip()
                if not content:
                    content = (getattr(message, "reasoning_content", None) or "").strip()
                return content
            except Exception:
                if attempt == 0:
                    continue
                raise
        return ""  # pragma: no cover (loop di atas selalu raise di attempt terakhir)

    def run(self, image_path: str, **kwargs) -> ModelResult:
        """Kontrak wajib: ekstraksi teks umum dari gambar (jarang dipakai langsung)."""
        t0 = time.time()
        try:
            text = self._chat(image_path, READ_PROMPT)
            return ModelResult(text, self.name, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    # -- langkah tulisan tangan / coretan ---------------------------------
    def detect_handwriting(self, image_path: str) -> ModelResult:
        """Konfirmasi YA/TIDAK ada tulisan tangan/coretan. extra["detected"]."""
        t0 = time.time()
        try:
            raw = self._chat(image_path, DETECT_PROMPT, max_new_tokens=self._DETECT_MAX_NEW_TOKENS)
            text = self._strip_box_tags(raw)
            has_ya = bool(re.search(r"\bYA\b", text, re.IGNORECASE))
            has_tidak = bool(re.search(r"\bTIDAK\b", text, re.IGNORECASE))
            detected = has_ya and not has_tidak
            return ModelResult(
                text,
                self.name,
                time.time() - t0,
                extra={"detected": detected, "raw": raw},
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    def read_handwriting(self, image_path: str) -> ModelResult:
        """Baca coretan + pembetulan. extra["corrections"], extra["cleaned_text"]."""
        t0 = time.time()
        try:
            # Resolusi lebih tinggi agar tulisan tangan kecil terbaca.
            text = self._chat(image_path, READ_PROMPT, max_side=VLM_API_READ_MAX_SIDE)
            data = extract_json(text)
            if not isinstance(data, dict):
                raise ValueError("Output READ bukan objek JSON.")
            return ModelResult(
                text,
                self.name,
                time.time() - t0,
                extra={
                    "corrections": data.get("corrections") or [],
                    "insertions": data.get("insertions") or [],
                    "handwritten_notes": data.get("handwritten_notes") or [],
                    "cleaned_text": data.get("cleaned_text") or "",
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    def review_fields(
        self,
        image_path: str,
        fields: dict,
        doc_type: str = "invoice",
        ocr_text: str = "",
    ) -> ModelResult:
        """Koreksi field JSON hasil NuExtract. extra["changes"] (list dict)."""
        t0 = time.time()
        try:
            prompt = REVIEW_PROMPT.format(
                fields_json=json.dumps(fields, ensure_ascii=False, indent=2)
            )
            if ocr_text:
                prompt += f"\n\nTeks OCR pendukung:\n{ocr_text[:2000]}"
            text = self._chat(image_path, prompt)
            data = extract_json(text)
            if not isinstance(data, dict):
                raise ValueError("Output REVIEW bukan objek JSON.")
            changes = data.get("changes") or []
            if not isinstance(changes, list):
                changes = []
            return ModelResult(
                text,
                self.name,
                time.time() - t0,
                extra={"changes": changes, "doc_type": doc_type},
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))
