"""Implementasi NuExtract3-GGUF sebagai model UTAMA (role="main").

NuExtract3 (5B, vision-language) memproduksi JSON terstruktur sesuai schema
yang diberikan. Di sini dijalankan via llama.cpp (GGUF Q4_K_M + mmproj) karena
ringan untuk CPU-only RAM 12GB (download ~3.4GB, RAM inferensi ~3-4GB).

Logika diambil dari ``doc-validation/extractors/adapters/nuextract3_gguf_adapter.py``
yang sudah terbukti jalan; diadaptasi ke kontrak ``BaseExtractionModel``.

PENTING:
- Gambar WAJIB di-resize dulu (``fit_image_for_vision``) sebelum dikirim ke
  model — CLIP/mmproj llama.cpp segfault (core dump) pada gambar > ~1 MP.
- Teks OCR pendukung opsional; gambar tetap jadi sinyal utama.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any

from config import (
    ENABLE_THINKING,
    GGUF_MODEL_PATH,
    GGUF_MMPROJ_PATH,
    MAX_NEW_TOKENS,
    N_CTX,
    N_GPU_LAYERS,
    N_THREADS,
    TEMPERATURE,
)
from extraction.schemas import build_instructions, build_schema
from models.base import BaseExtractionModel, ModelResult

# ---------------------------------------------------------------------------
# CHAT TEMPLATE NuExtract3 (disalin dari model repo — GGUF tidak meng-embed-nya)
# ---------------------------------------------------------------------------
_CHAT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "extraction" / "nuextract3_chat_template.jinja"
_DEFAULT_CHAT_TEMPLATE = _CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")

# Batas aman gambar untuk encoder CLIP llama.cpp.
VISION_MAX_PIXELS = 900_000
VISION_MAX_SIDE = 1000


# ---------------------------------------------------------------------------
# UTIL GAMBAR
# ---------------------------------------------------------------------------
def fit_image_for_vision(image_bytes: bytes) -> bytes:
    """Downscale *image_bytes* agar aman untuk model vision (< ~1 MP)."""
    if not image_bytes:
        return image_bytes
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    if width * height <= VISION_MAX_PIXELS and max(width, height) <= VISION_MAX_SIDE:
        return image_bytes

    scale = min(1.0, VISION_MAX_PIXELS / (width * height), VISION_MAX_SIDE / max(width, height))
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    img = img.convert("RGB").resize((new_width, new_height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UTIL JSON
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Any:
    """Ambil blok JSON pertama ({...} atau [...]) dari output model."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start == -1:
        raise ValueError("Output model tidak mengandung JSON.")

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Blok JSON tidak lengkap pada output model.")


# ---------------------------------------------------------------------------
# MTMD CHAT HANDLER — paksa template NuExtract + schema/instructions
# ---------------------------------------------------------------------------
def _get_mtmd_handler_class():
    """Return chat handler untuk rendering prompt terstruktur NuExtract."""
    from llama_cpp.llama_chat_format import MTMDChatHandler

    class _NuExtractMTMDChatHandler(MTMDChatHandler):
        def __init__(
            self,
            clip_model_path: str,
            template: str,
            instructions: str = "",
            enable_thinking: bool = False,
            verbose: bool = True,
            use_gpu: bool = False,
        ) -> None:
            super().__init__(
                clip_model_path=clip_model_path,
                verbose=verbose,
                use_gpu=use_gpu,
            )
            self._nu_template = template
            self._nu_instructions = instructions
            self._nu_enable_thinking = enable_thinking

        def _get_chat_template(self, llama_model: Any) -> str:
            return _DEFAULT_CHAT_TEMPLATE

        def __call__(self, *, llama: Any, messages: list, **kwargs: Any):
            kwargs["template"] = self._nu_template
            kwargs["enable_thinking"] = self._nu_enable_thinking
            if self._nu_instructions:
                kwargs["instructions"] = self._nu_instructions
            return super().__call__(llama=llama, messages=messages, **kwargs)

    return _NuExtractMTMDChatHandler


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
class NuExtract3GGUFModel(BaseExtractionModel):
    """NuExtract3-GGUF sebagai ekstraktor terstruktur (model utama)."""

    name = "NuExtract3-GGUF"
    role = "main"

    def __init__(
        self,
        model_path: str = str(GGUF_MODEL_PATH),
        mmproj_path: str = str(GGUF_MMPROJ_PATH),
        n_ctx: int = N_CTX,
        n_threads: int = N_THREADS,
        n_gpu_layers: int = N_GPU_LAYERS,
        temperature: float = TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        enable_thinking: bool = ENABLE_THINKING,
    ) -> None:
        self._model_path = model_path
        self._mmproj_path = mmproj_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._enable_thinking = enable_thinking
        self._llm: Any = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        if self._llm is not None:
            return

        for label, path in (("model", self._model_path), ("mmproj", self._mmproj_path)):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"File GGUF {label} tidak ditemukan: {path}. "
                    "Pastikan model sudah ada di data/models/nuextract3/."
                )

        from llama_cpp import Llama

        handler_cls = _get_mtmd_handler_class()
        handler = handler_cls(
            clip_model_path=self._mmproj_path,
            template=build_schema("invoice"),
            enable_thinking=self._enable_thinking,
            verbose=False,
            use_gpu=False,
        )
        self._llm = Llama(
            model_path=self._model_path,
            chat_handler=handler,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads if self._n_threads > 0 else None,
            n_gpu_layers=self._n_gpu_layers,
            verbose=False,
        )

    def unload(self) -> None:
        self._llm = None

    # -- inference ---------------------------------------------------------
    def run(self, image_path: str, doc_type: str = "invoice", ocr_text: str = "", **kwargs) -> ModelResult:
        """Ekstrak field terstruktur dari *image_path*.

        Parameters
        ----------
        image_path:
            Gambar hasil preprocessing (PNG/JPG).
        doc_type:
            Jenis dokumen (invoice, purchase_order, delivery_order, kwitansi, tax_invoice).
        ocr_text:
            Teks OCR pendukung (opsional). Hanya konteks, bukan sumber utama.
        """
        t0 = time.time()
        self.load()
        try:
            image_bytes = Path(image_path).read_bytes()
            image_bytes = fit_image_for_vision(image_bytes)
            b64 = base64.b64encode(image_bytes).decode("ascii")

            content: list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
            ocr_text = (ocr_text or "").strip()
            if ocr_text:
                content.append({"type": "text", "text": self._prepare_document_text(ocr_text)})

            # Schema & instructions per jenis dokumen.
            self._set_handler_template(doc_type)

            response = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": content}],
                temperature=self._temperature,
                max_tokens=self._max_new_tokens,
            )
            output_text = (response["choices"][0]["message"].get("content") or "").strip()
            fields = extract_json(output_text)

            elapsed = time.time() - t0
            return ModelResult(
                output_text,
                self.name,
                elapsed,
                fields=fields if isinstance(fields, dict) else None,
                extra={"doc_type": doc_type, "used_ocr_text": bool(ocr_text)},
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    # -- helpers -----------------------------------------------------------
    def _set_handler_template(self, doc_type: str) -> None:
        """Perbarui schema/instructions pada handler sesuai jenis dokumen."""
        handler = getattr(self._llm, "chat_handler", None)
        if handler is None:
            return
        handler._nu_template = build_schema(doc_type)
        handler._nu_instructions = build_instructions(doc_type, lang="id")

    @staticmethod
    def _prepare_document_text(document_text: str, n_ctx: int = N_CTX) -> str:
        """Trim teks OCR panjang agar prompt tetap dalam context window."""
        if not document_text:
            return ""
        reserve_tokens = 1024
        max_tokens = max(1024, n_ctx - reserve_tokens)
        max_chars = max(2048, int(max_tokens * 2.5))
        if len(document_text) <= max_chars:
            return document_text
        midpoint = max_chars // 2
        prefix = document_text[:midpoint].rstrip()
        suffix = document_text[-(max_chars - midpoint):].lstrip()
        return f"{prefix}\n...[truncated]...\n{suffix}"
