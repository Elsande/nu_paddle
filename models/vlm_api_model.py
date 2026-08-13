"""VLM API — paddleocr-vl 1.6, qwen3-vl-30b, glm-4.6v-flash (SATU gateway).

Ketiga model dipanggil LEWAT API OpenAI-compatible yang SAMA (gateway lokal
perusahaan, ``config.VLM_API_BASE_URL``) — yang membedakan hanya nama model.
Model TIDAK dimuat ke mesin ini; cukup HTTP client ringan.

Kelas:
- ``OpenAIVLModel`` : base generik untuk semua model VLM OpenAI-compatible.
  Punya ``_chat()`` (gambar + prompt) + ``extract_document()`` (ekstraksi
  JSON terstruktur sesuai schema per jenis dokumen) + langkah tulisan
  tangan/coretan (detect/read/review).
- ``VLMApiModel``  : GLM-4.6V-Flash (nama model dari ``config.GLM_VL_MODEL``).
- ``Qwen3VLModel`` : Qwen3-VL-30B (nama model dari ``config.QWEN3_VL_MODEL``).

Kegagalan API tidak boleh menggagalkan dokumen: cukup di-catch dan dicatat
(degrade halus, anti miss).
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
    GLM_VL_MODEL,
    QWEN3_VL_MODEL,
    VLM_API_BASE_URL,
    VLM_API_DISABLE_THINKING,
    VLM_API_JPEG_QUALITY,
    VLM_API_KEY,
    VLM_API_MAX_NEW_TOKENS,
    VLM_API_MAX_SIDE,
    VLM_API_READ_MAX_SIDE,
    VLM_API_TEMPERATURE,
    VLM_API_TIMEOUT,
)
from extraction.schemas import build_schema, build_instructions
from models.base import BaseExtractionModel, ModelResult
from models.nuextract_gguf_model import extract_json


# ---------------------------------------------------------------------------
# PROMPT EKSTRAKSI TERSTRUKTUR (dipakai qwen3-vl & glm — SAMA PERSIS)
# ---------------------------------------------------------------------------
EXTRACT_SYSTEM_PROMPT = (
    "Kamu adalah sistem ekstraksi dokumen bisnis Indonesia. Tugasmu HANYA "
    "membaca dan mengekstrak informasi yang benar-benar terlihat di gambar. "
    "Jangan menebak, menghitung, atau melengkapi data yang tidak tertulis. "
    "Jika ada tulisan tangan (handwriting), coretan, atau catatan tambahan, "
    "ekstrak TERPISAH ke 'anotasi_tulisan_tangan'. Jika ada teks yang dicoret "
    "dan diperbaiki dengan tulisan tangan, gunakan nilai PEMBETULAN. Jika suatu "
    "field tidak terbaca atau tidak ada, isi null. Jangan menerjemahkan istilah "
    "Indonesia. Output HARUS berupa JSON valid saja — tidak ada teks pembuka, "
    "penutup, atau penjelasan di luar JSON."
)

EXTRACT_USER_PROMPT = (
    "Berikut gambar dokumen bisnis jenis '{doc_type}'. Ekstrak seluruh informasi "
    "yang terlihat sesuai template JSON di bawah ini (JANGAN mengubah nama key):\n\n"
    "{schema}\n\n"
    "Kembalikan HANYA JSON valid dengan format:\n"
    '{{"<nama_field>": {{"value": <nilai atau null>, "confidence": 0.0-1.0}}, '
    '"...": "...", "line_items": [objek biasa tanpa value/confidence], '
    '"anotasi_tulisan_tangan": [{{"teks": string, "lokasi_referensi": string, '
    '"jenis": "koreksi|catatan_tambahan|paraf|tidak_diketahui", "confidence": 0.0}}]}}\n'
    "- Untuk tiap field template, isi objek {\"value\", \"confidence\"}.\n"
    "- line_items (jika ada di template) diisi array objek polos.\n"
    "- anotasi_tulisan_tangan: daftar tulisan tangan/coretan (kosong bila tidak ada)."
)

# ---------------------------------------------------------------------------
# PROMPT TULISAN TANGAN / CORETAN (dipakai GLM — selektif)
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


class OpenAIVLModel(BaseExtractionModel):
    """Base generik: model VLM yang dipanggil via API OpenAI-compatible.

    Semua model (paddleocr-vl 1.6, qwen3-vl-30b, glm-4.6v-flash) memakai
    gateway yang sama; subclass hanya mengganti nama model + nama tampilan.
    """

    name = "OpenAI VLM API"
    role = "support"
    _default_model: str = ""

    # max_tokens khusus untuk deteksi YA/TIDAK (model memakai reasoning dulu).
    _DETECT_MAX_NEW_TOKENS = 1024

    def __init__(
        self,
        model: str | None = None,
        base_url: str = VLM_API_BASE_URL,
        api_key: str = VLM_API_KEY,
        timeout: int = VLM_API_TIMEOUT,
        max_new_tokens: int = VLM_API_MAX_NEW_TOKENS,
        temperature: float = VLM_API_TEMPERATURE,
        disable_thinking: bool = VLM_API_DISABLE_THINKING,
    ) -> None:
        self._model = model or self._default_model
        self._base_url = base_url
        self._api_key = api_key
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
    @staticmethod
    def _strip_box_tags(text: str) -> str:
        """Buang penanda <|begin_of_box|> / <|end_of_box|> dari output model."""
        return re.sub(r"<\|(?:begin|end)_of_box\|>", "", text).strip()

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
        system_prompt: str | None = None,
    ) -> str:
        """Satu panggilan chat completion dengan gambar + prompt. Return teks."""
        self.load()
        if self._client is None:
            raise RuntimeError("Client VLM API tidak tersedia (panggil load() dulu).")

        image_url = self._encode_image(image_path, max_side=max_side)
        user_content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": text_prompt},
        ]
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_new_tokens or self._max_new_tokens,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        # Coba dengan param tambahan (thinking disabled); bila gateway menolak
        # (400), ulangi tanpa param tambahan (temperature tetap dipertahankan).
        for attempt in (0, 1):
            if attempt == 0 and self._disable_thinking:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            elif attempt == 1:
                kwargs.pop("extra_body", None)
            try:
                completion = self._client.chat.completions.create(**kwargs)
                message = completion.choices[0].message
                # Model lokal bisa mengembalikan jawaban di reasoning_content
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
        """Kontrak wajib: teks umum dari gambar (jarang dipakai langsung)."""
        t0 = time.time()
        try:
            text = self._chat(image_path, READ_PROMPT)
            return ModelResult(text, self.name, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    # -- ekstraksi terstruktur (qwen3-vl & glm) ----------------------------
    def extract_document(
        self,
        image_path: str,
        doc_type: str = "invoice",
        ocr_text: str = "",
        max_side: int = VLM_API_MAX_SIDE,
    ) -> ModelResult:
        """Ekstrak field JSON terstruktur dari *image_path*.

        Output ModelResult.fields = {field: value}. extra["confidences"] =
        {field: 0.0-1.0 | None}, extra["handwritten_notes"], extra["doc_type"].
        """
        t0 = time.time()
        try:
            schema = build_schema(doc_type)
            instructions = build_instructions(doc_type, lang="id")
            prompt = EXTRACT_USER_PROMPT.format(doc_type=doc_type, schema=schema)
            if instructions:
                prompt += f"\n\nPetunjuk lapangan:\n{instructions}"
            if ocr_text:
                prompt += f"\n\nTeks OCR pendukung (konteks saja):\n{ocr_text[:4000]}"

            text = self._chat(
                image_path,
                prompt,
                max_side=max_side,
                system_prompt=EXTRACT_SYSTEM_PROMPT,
            )
            data = extract_json(text)
            if not isinstance(data, dict):
                raise ValueError("Output ekstraksi bukan objek JSON.")

            fields: dict[str, Any] = {}
            confidences: dict[str, float | None] = {}
            for key, value in data.items():
                if key == "anotasi_tulisan_tangan":
                    continue
                if isinstance(value, dict) and "value" in value:
                    conf = value.get("confidence")
                    confidences[key] = float(conf) if isinstance(conf, (int, float)) else None
                    fields[key] = value.get("value")
                else:
                    fields[key] = value

            return ModelResult(
                text,
                self.name,
                time.time() - t0,
                fields=fields,
                extra={
                    "confidences": confidences,
                    "handwritten_notes": data.get("anotasi_tulisan_tangan") or [],
                    "doc_type": doc_type,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    # -- langkah tulisan tangan / coretan (selektif) -----------------------
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
        """Koreksi field JSON hasil fusion. extra["changes"] (list dict)."""
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


class VLMApiModel(OpenAIVLModel):
    """GLM-4.6V-Flash via API OpenAI-compatible (tulisan tangan/coretan)."""

    name = "GLM-4.6V-Flash (API)"
    role = "support"
    _default_model = GLM_VL_MODEL


class Qwen3VLModel(OpenAIVLModel):
    """Qwen3-VL-30B via API OpenAI-compatible (ekstraksi + tulisan tangan)."""

    name = "Qwen3-VL-30B (API)"
    role = "support"
    _default_model = QWEN3_VL_MODEL
