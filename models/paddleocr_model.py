"""Implementasi PaddleOCR sebagai model pendukung (role="support").

Tidak menghasilkan hasil akhir; hanya memberi teks + confidence + bbox sebagai
konteks pendukung untuk NuExtract3-GGUF (model utama). Logika diambil dari
project ``paddleocr/invoice-doc-ai`` (detect-then-recognize + CLAHE + spatial
sort), yang sudah terbukti bekerja di mesin ini.

PENTING:
- Engine di-load sekali (singleton lazy) untuk hemat RAM.
- PaddleX dipatch ke mode offline supaya memakai model lokal
  ``~/.paddlex/official_models`` (tidak pernah download dari internet).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from config import OCR_CONFIDENCE_THRESHOLD, PADDLEOCR_PARAMS
from models.base import BaseExtractionModel, ModelResult

_LOCAL_MODELS_ROOT = Path(os.path.expanduser("~/.paddlex/official_models"))

# ---------------------------------------------------------------------------
# OFFLINE PATCH — Paksa PaddleX memakai model lokal (mencegah download).
# ---------------------------------------------------------------------------
def _find_local_model(name: str) -> Path:
    names = (name,) if isinstance(name, str) else tuple(name)
    for n in names:
        for candidate in (_LOCAL_MODELS_ROOT / n, _LOCAL_MODELS_ROOT / n.replace("_infer", "")):
            if candidate.exists():
                return candidate
        base = n.replace("_infer", "")
        sub = _LOCAL_MODELS_ROOT / base / f"{base}_infer"
        if sub.exists():
            return sub
        for p in _LOCAL_MODELS_ROOT.iterdir():
            if p.is_dir() and (n in p.name or base in p.name or p.name in n):
                return p
    raise FileNotFoundError(
        f"[OFFLINE] Model '{names[0]}' tidak ditemukan di {_LOCAL_MODELS_ROOT}. "
        "Download model dulu (satu kali) agar offline patch berfungsi."
    )


def _patch_paddlex_offline() -> None:
    try:
        import importlib

        om_mod = importlib.import_module("paddlex.inference.utils.official_models")

        def _patched_local_path(self, model_name):
            return _find_local_model(model_name)

        def _patched_get_path(self, model_name, *, model_formats=None):
            return _find_local_model(model_name)

        def _patched_build_hosters(self):
            return []

        if hasattr(om_mod, "_ModelManager"):
            om_mod._ModelManager._get_model_local_path = _patched_local_path
            om_mod._ModelManager.get_model_path = _patched_get_path
            om_mod._ModelManager._build_hosters = _patched_build_hosters
        if hasattr(om_mod, "official_models"):
            inst = om_mod.official_models
            inst._get_model_local_path = lambda *a, **kw: _patched_local_path(inst, *a, **kw)
            inst.get_model_path = lambda *a, **kw: _patched_get_path(inst, *a, **kw)
            inst._build_hosters = _patched_build_hosters
    except Exception:
        pass


_patch_paddlex_offline()


# ---------------------------------------------------------------------------
# HELPER GAMBAR
# ---------------------------------------------------------------------------
def _apply_clahe(image_np: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """CLAHE pada channel Luminance (YCrCb). Input/output: OpenCV BGR."""
    if image_np is None or image_np.size == 0:
        return image_np
    if image_np.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image_np)
    ycrcb = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)
    y_chan, cr_chan, cb_chan = cv2.split(ycrcb)
    y_enhanced = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(y_chan)
    merged = cv2.merge((y_enhanced, cr_chan, cb_chan))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def _sort_boxes_spatially(lines_data: list[dict]) -> list[dict]:
    """Urutkan baris OCR berdasarkan Y (baris) lalu X (kolom) — urutan baca wajar."""
    valid_boxes: list[tuple] = []
    invalid_boxes: list[dict] = []
    for line in lines_data:
        bbox = line.get("bbox")
        if bbox is not None and len(bbox) > 0:
            try:
                ys = [pt[1] for pt in bbox]
                xs = [pt[0] for pt in bbox]
                valid_boxes.append((min(ys), min(xs), max(ys) - min(ys), line))
            except Exception:
                invalid_boxes.append(line)
        else:
            invalid_boxes.append(line)

    if not valid_boxes:
        return invalid_boxes

    valid_boxes.sort(key=lambda item: item[0])
    sorted_lines: list[dict] = []
    current_group = [valid_boxes[0]]

    for item in valid_boxes[1:]:
        prev_item = current_group[-1]
        tolerance = prev_item[2] * 0.5 if prev_item[2] > 0 else 10
        if abs(item[0] - prev_item[0]) <= tolerance:
            current_group.append(item)
        else:
            current_group.sort(key=lambda x: x[1])
            sorted_lines.extend(x[3] for x in current_group)
            current_group = [item]

    current_group.sort(key=lambda x: x[1])
    sorted_lines.extend(x[3] for x in current_group)
    return sorted_lines + invalid_boxes


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
class PaddleOCRModel(BaseExtractionModel):
    """PaddleOCR sebagai OCR pendukung: teks + confidence + bbox per baris."""

    name = "PaddleOCR"
    role = "support"

    def __init__(self, params: Optional[dict] = None, confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD):
        self._engine: Any = None
        self._params: dict = dict(PADDLEOCR_PARAMS, **(params or {}))
        self.confidence_threshold = confidence_threshold

    def load(self) -> None:
        if self._engine is not None:
            return
        try:
            import paddle

            if hasattr(paddle, "disable_signal_handler"):
                paddle.disable_signal_handler()
        except Exception:
            pass
        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(**self._params)

    def _predict_image(self, img) -> tuple[list[dict], list[dict]]:
        """CLAHE -> predict -> spatial sort. Kembalikan (semua baris, baris tersaring)."""
        clahe_img = _apply_clahe(img)
        pred_results = list(self._engine.predict(clahe_img))

        lines: list[dict] = []
        for result in pred_results:
            if hasattr(result, "get"):
                rec_texts = result.get("rec_texts", [])
                rec_scores = result.get("rec_scores", [])
                rec_polys = result.get("rec_polys", result.get("dt_polys", []))
            else:
                rec_texts = getattr(result, "rec_texts", [])
                rec_scores = getattr(result, "rec_scores", [])
                rec_polys = getattr(result, "rec_polys", getattr(result, "dt_polys", []))

            for i, text in enumerate(rec_texts):
                text_str = str(text).strip()
                if not text_str:
                    continue
                score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                poly = rec_polys[i] if (rec_polys is not None and i < len(rec_polys)) else None
                bbox = poly.tolist() if hasattr(poly, "tolist") else (list(poly) if poly is not None else None)
                lines.append({"bbox": bbox, "text": text_str, "confidence": score})

        lines = _sort_boxes_spatially(lines)
        kept = [ln for ln in lines if ln["confidence"] >= self.confidence_threshold]
        return lines, kept

    def run(self, image_path: str, **kwargs) -> ModelResult:
        t0 = time.time()
        self.load()
        try:
            raw_img = cv2.imread(image_path)
            if raw_img is None:
                return ModelResult("", self.name, 0.0, error=f"Gambar tidak terbaca: {image_path}")

            lines, kept = self._predict_image(raw_img)
            filtered_text = "\n".join(ln["text"] for ln in kept)
            elapsed = time.time() - t0
            return ModelResult(
                filtered_text,
                self.name,
                elapsed,
                extra={
                    "lines": lines,
                    "kept_lines": kept,
                    "total_lines": len(lines),
                    # Seluruh baris (tanpa filter) — "ambil semua dari gambar".
                    "all_lines": [{"text": ln["text"], "confidence": ln["confidence"]} for ln in lines],
                    "all_text": "\n".join(ln["text"] for ln in lines),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    def run_header(self, image_path: str, top_ratio: float = 0.22, scale: float = 2.0) -> ModelResult:
        """OCR pas kedua khusus area HEADER (logo/kop): crop atas + upscale 2x.

        Nama perusahaan sering berada di dalam logo beresolusi rendah yang tidak
        terbaca pada OCR normal; upscale membuat teks logo bisa terbaca (mis.
        ``PT.INTI SOLUSINDO ABADI``). Hasilnya digabung sebagai konteks tambahan.
        """
        t0 = time.time()
        self.load()
        try:
            raw_img = cv2.imread(image_path)
            if raw_img is None:
                return ModelResult("", self.name, 0.0, error=f"Gambar tidak terbaca: {image_path}")
            h, w = raw_img.shape[:2]
            crop = raw_img[0 : int(h * top_ratio), :]
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            lines, kept = self._predict_image(crop)
            filtered_text = "\n".join(ln["text"] for ln in kept)
            return ModelResult(
                filtered_text,
                self.name,
                time.time() - t0,
                extra={
                    "lines": lines,
                    "kept_lines": kept,
                    "total_lines": len(lines),
                    "all_lines": [{"text": ln["text"], "confidence": ln["confidence"]} for ln in lines],
                    "all_text": "\n".join(ln["text"] for ln in lines),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    def unload(self) -> None:
        self._engine = None
