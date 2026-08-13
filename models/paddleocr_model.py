"""paddleocr-vl 1.6 via API — pengganti PaddleOCR lokal (role="support").

Tidak ada lagi model PaddleOCR yang dimuat lokal. paddleocr-vl 1.6 dipanggil
lewat gateway OpenAI-compatible yang SAMA dengan qwen3-vl & glm
(``config.VLM_API_BASE_URL``), nama model dari ``config.PADDLEOCR_VL_MODEL``.

Output model (JSON ``{"texts": [...]}``) di-parse menjadi baris OCR dengan
struktur yang sama seperti dulu PaddleOCR lokal: text + bbox + confidence,
lalu di-sort spasial. Interface ``run()`` / ``run_region()`` / ``run_header()``
dipertahankan agar `run_batch._zone_ocr` tidak perlu diubah perilakunya.

Bila model tidak mengembalikan bbox/confidence, digunakan default netral
(``PADDLEOCR_VL_DEFAULT_CONFIDENCE``) supaya heuristik tulisan tangan tidak
menjadi salah trigger dan koreksi kode OCR tidak asal diterapkan.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Optional

import cv2
import numpy as np

from config import (
    LAYOUT_ZONE_MARGIN,
    LAYOUT_ZONE_MAX_DIM,
    OCR_CONFIDENCE_THRESHOLD,
    PADDLEOCR_VL_DEFAULT_CONFIDENCE,
    PADDLEOCR_VL_SYSTEM_PROMPT,
    PADDLEOCR_VL_USER_PROMPT,
    VLM_API_MAX_SIDE,
)
from models.base import ModelResult
from models.nuextract_gguf_model import extract_json
from models.vlm_api_model import OpenAIVLModel


# ---------------------------------------------------------------------------
# HELPER GAMBAR
# ---------------------------------------------------------------------------
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


class PaddleOCRVLApiModel(OpenAIVLModel):
    """paddleocr-vl 1.6 via API: teks + confidence + bbox per baris OCR."""

    name = "PaddleOCR-VL 1.6 (API)"
    role = "support"

    def __init__(
        self,
        confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
        max_side: int = VLM_API_MAX_SIDE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.max_side = max_side

    # -- parsing -----------------------------------------------------------
    def _parse_ocr_lines(self, text: str) -> list[dict]:
        """Parse output model (JSON ``{"texts": [...]}``) menjadi baris OCR.

        Degrade halus: bila output bukan JSON sesuai format, baris teks mentah
        dipakai apa adanya (bbox=None, confidence=default netral).
        """
        lines: list[dict] = []
        try:
            data = extract_json(text)
            if isinstance(data, dict) and isinstance(data.get("texts"), list):
                for item in data["texts"]:
                    t = str(item.get("text") or "").strip()
                    if not t:
                        continue
                    bbox = item.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        bbox = None
                    else:
                        try:
                            bbox = [float(v) for v in bbox]
                        except (TypeError, ValueError):
                            bbox = None
                    conf = item.get("confidence")
                    if isinstance(conf, (int, float)):
                        conf = float(conf)
                    else:
                        conf = PADDLEOCR_VL_DEFAULT_CONFIDENCE
                    lines.append({"text": t, "bbox": bbox, "confidence": conf})
        except Exception:
            lines = []

        if not lines:
            for ln in (text or "").splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(
                        {
                            "text": ln,
                            "bbox": None,
                            "confidence": PADDLEOCR_VL_DEFAULT_CONFIDENCE,
                        }
                    )
        return _sort_boxes_spatially(lines)

    @staticmethod
    def _save_temp_image(crop: np.ndarray) -> str:
        """Simpan crop (BGR ndarray) ke file PNG sementara."""
        fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="paddleocr_vl_")
        os.close(fd)
        cv2.imwrite(tmp_path, crop)
        return tmp_path

    # -- inference ---------------------------------------------------------
    def _predict_image(self, image_path: str) -> tuple[list[dict], list[dict]]:
        """paddleocr-vl via API + spatial sort. (semua baris, baris tersaring)."""
        text = self._chat(
            image_path,
            PADDLEOCR_VL_USER_PROMPT,
            max_side=self.max_side,
            system_prompt=PADDLEOCR_VL_SYSTEM_PROMPT,
        )
        lines = self._parse_ocr_lines(text)
        kept = [ln for ln in lines if ln["confidence"] >= self.confidence_threshold]
        return lines, kept

    def run(self, image_path: str, **kwargs) -> ModelResult:
        t0 = time.time()
        self.load()
        try:
            lines, kept = self._predict_image(image_path)
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

    def run_region(self, image_path: str, box, scale: float = 1.0, label: str = "", margin: int | None = None) -> ModelResult:
        """OCR pada satu wilayah (zona layout): crop bbox (+margin) lalu upscale.

        Parameters
        ----------
        box : iterable [x0, y0, x1, y1]
            Bounding box zona (koordinat gambar asli).
        scale : float
            Faktor perbesaran crop sebelum OCR (logo kecil butuh scale > 1).
        label : str
            Nama zona (untuk disimpan di hasil).
        margin : int
            Padding sekitar bbox; default dari config.LAYOUT_ZONE_MARGIN.
        """
        t0 = time.time()
        self.load()
        try:
            raw_img = cv2.imread(image_path)
            if raw_img is None:
                return ModelResult("", self.name, 0.0, error=f"Gambar tidak terbaca: {image_path}")
            h, w = raw_img.shape[:2]
            x0, y0, x1, y1 = [int(v) for v in box]
            pad = LAYOUT_ZONE_MARGIN if margin is None else margin
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
            if x1 <= x0 or y1 <= y0:
                return ModelResult("", self.name, time.time() - t0, extra={"label": label, "box": list(box), "scale": scale})

            crop = raw_img[y0:y1, x0:x1]
            if scale > 0 and scale != 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            # Cap ukuran hasil upscale -> batasi runtime OCR zona.
            max_dim = max(crop.shape[:2])
            if max_dim > LAYOUT_ZONE_MAX_DIM:
                f = LAYOUT_ZONE_MAX_DIM / max_dim
                crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

            tmp_path = self._save_temp_image(crop)
            try:
                text = self._chat(
                    tmp_path,
                    PADDLEOCR_VL_USER_PROMPT,
                    max_side=self.max_side,
                    system_prompt=PADDLEOCR_VL_SYSTEM_PROMPT,
                )
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            lines = self._parse_ocr_lines(text)
            kept = [ln for ln in lines if ln["confidence"] >= self.confidence_threshold]
            filtered_text = "\n".join(ln["text"] for ln in kept)
            return ModelResult(
                filtered_text,
                self.name,
                time.time() - t0,
                extra={
                    "label": label,
                    "box": list(box),
                    "scale": scale,
                    "lines": lines,
                    "kept_lines": kept,
                    "total_lines": len(lines),
                    "all_lines": [{"text": ln["text"], "confidence": ln["confidence"]} for ln in lines],
                    "all_text": "\n".join(ln["text"] for ln in lines),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    def run_header(self, image_path: str, top_ratio: float = 0.22, scale: float = 2.0) -> ModelResult:
        """OCR pas kedua khusus area HEADER (logo/kop) — fallback tanpa layout.

        Crop bagian atas halaman + upscale agar teks di dalam logo terbaca.
        Didelegasikan ke :meth:`run_region` dengan bbox atas (satu jalur kode).
        """
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            return ModelResult("", self.name, 0.0, error=f"Gambar tidak terbaca: {image_path}")
        h, w = raw_img.shape[:2]
        return self.run_region(
            image_path,
            [0, 0, w, int(h * top_ratio)],
            scale=scale,
            label="header_fallback",
        )

    def unload(self) -> None:
        super().unload()
