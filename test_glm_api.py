#!/usr/bin/env python3
"""
test_glm_api.py
===============
Benchmark LATENCY glm-4.6v-flash via API (gateway OpenAI-compatible).

Pemakaian:
    python test_glm_api.py                                 # semua contoh
    python test_glm_api.py "contoh invoice/.../INV.pdf"
    python test_glm_api.py --doc-type invoice --model zai-org/glm-4.6v-flash

Mengukur berapa lama tiap ekstraksi bekerja + menampilkan field hasil JSON.
Model tidak dimuat lokal; cukup HTTP client.
"""

from __future__ import annotations

import argparse
import json
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
        tmp = Path(tempfile.gettempdir()) / f"test_glm_{path.stem}.png"
        from preprocessing.pillow_utils import save_array_as_image

        save_array_as_image(img, str(tmp))
        return str(tmp)
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark latency glm-4.6v-flash via API")
    ap.add_argument("images", nargs="*", help="path gambar/PDF (default: semua contoh)")
    ap.add_argument("--doc-type", default=None, help="invoice/purchase_order/delivery_order/kwitansi/tax_invoice")
    ap.add_argument("--model", default=config.GLM_VL_MODEL, help="nama model di gateway")
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
    print(f"[INFO] {len(docs)} dokumen akan diekstrak\n")

    model = AVAILABLE_MODELS["GLM-4.6V-Flash (API)"](
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    model.load()

    times: list[float] = []
    for i, doc in enumerate(docs, 1):
        doc_type = args.doc_type or config.detect_document_type(doc)
        image = to_image(doc)
        print(f"\n[{i}/{len(docs)}] {doc.name} (jenis: {doc_type})")
        t0 = time.time()
        res = model.extract_document(image, doc_type=doc_type, max_side=args.max_side)
        dt = time.time() - t0
        times.append(dt)
        print(f"    elapsed : {dt:.2f} s")
        if res.error:
            print(f"    ERROR   : {res.error}")
            continue
        print(f"    fields  : {json.dumps(res.fields, ensure_ascii=False)[:500]}")
        confs = (res.extra or {}).get("confidences") or {}
        if confs:
            shown = {k: v for k, v in list(confs.items())[:5]}
            print(f"    conf    : {json.dumps(shown)}")

    if times:
        total = sum(times)
        print(
            f"\n{'=' * 60}\nRINGKASAN LATENCY (glm-4.6v-flash via API)\n{'=' * 60}"
            f"\n  total : {total:.2f} s"
            f"\n  avg   : {total / len(times):.2f} s / dokumen"
            f"\n  min   : {min(times):.2f} s"
            f"\n  max   : {max(times):.2f} s"
        )


if __name__ == "__main__":
    main()
