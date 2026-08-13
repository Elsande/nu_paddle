#!/usr/bin/env python3
"""
test_paddleocr_vl_api.py
========================
Benchmark LATENCY paddleocr-vl 1.6 via API (gateway OpenAI-compatible).

Pemakaian:
    python test_paddleocr_vl_api.py                          # semua contoh
    python test_paddleocr_vl_api.py "contoh invoice/.../INV.pdf"
    python test_paddleocr_vl_api.py --model <nama-model> --base-url http://host:port/v1

Mengukur berapa lama tiap panggilan OCR bekerja + menampilkan cuplikan teks.
Model tidak dimuat lokal; cukup HTTP client. Tidak ada dependensi berat.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from models.registry import AVAILABLE_MODELS

SUPPORTED = config.SUPPORTED_IMAGE_EXTS | {".pdf"}


def discover(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p) for p in paths if Path(p).suffix.lower() in SUPPORTED]
    docs: list[Path] = []
    if config.SAMPLE_DIR.is_dir():
        for p in sorted(config.SAMPLE_DIR.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED:
                docs.append(p)
    return docs


def to_image(path: Path) -> str:
    """PDF -> gambar (halaman 1, temp); selain itu dipakai apa adanya."""
    if path.suffix.lower() == ".pdf":
        from preprocessing.pdf_to_image import pdf_to_image

        img = pdf_to_image(str(path))
        tmp = Path(tempfile.gettempdir()) / f"test_paddleocr_vl_{path.stem}.png"
        from preprocessing.pillow_utils import save_array_as_image

        save_array_as_image(img, str(tmp))
        return str(tmp)
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark latency paddleocr-vl 1.6 via API")
    ap.add_argument("images", nargs="*", help="path gambar/PDF (default: semua contoh)")
    ap.add_argument("--model", default=config.PADDLEOCR_VL_MODEL, help="nama model di gateway")
    ap.add_argument("--base-url", default=config.VLM_API_BASE_URL, help="gateway OpenAI-compatible")
    ap.add_argument("--api-key", default=config.VLM_API_KEY)
    ap.add_argument("--max-side", type=int, default=config.VLM_API_MAX_SIDE)
    args = ap.parse_args()

    docs = discover(args.images)
    if not docs:
        print(f"[SKIP] Tidak ada dokumen ditemukan di {config.SAMPLE_DIR}")
        return

    print(f"[INFO] Gateway : {args.base_url}")
    print(f"[INFO] Model   : {args.model}")
    print(f"[INFO] {len(docs)} dokumen akan di-OCR\n")

    model = AVAILABLE_MODELS["PaddleOCR-VL 1.6 (API)"](
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_side=args.max_side,
    )
    model.load()

    times: list[float] = []
    for i, doc in enumerate(docs, 1):
        image = to_image(doc)
        print(f"\n[{i}/{len(docs)}] {doc.name}")
        t0 = time.time()
        res = model.run(image)
        dt = time.time() - t0
        times.append(dt)
        print(f"    elapsed : {dt:.2f} s")
        if res.error:
            print(f"    ERROR   : {res.error}")
            continue
        extra = res.extra or {}
        print(f"    baris   : {extra.get('total_lines', 0)}")
        text = (extra.get("all_text") or "").strip()
        snippet = text[:300].replace("\n", " ⏎ ")
        print(f"    cuplikan: {snippet if snippet else '(kosong)'}")

    if times:
        total = sum(times)
        print(
            f"\n{'=' * 60}\nRINGKASAN LATENCY (paddleocr-vl 1.6 via API)\n{'=' * 60}"
            f"\n  total : {total:.2f} s"
            f"\n  avg   : {total / len(times):.2f} s / dokumen"
            f"\n  min   : {min(times):.2f} s"
            f"\n  max   : {max(times):.2f} s"
        )


if __name__ == "__main__":
    main()
