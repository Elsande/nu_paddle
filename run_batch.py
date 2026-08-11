"""
run_batch.py
============
Entry point nu-paddle: batch extraction (CLI) + UI Gradio.

Pemakaian:
    python run_batch.py                        # batch SEMUA dokumen contoh
    python run_batch.py [file1 file2 ...]      # batch dokumen tertentu
    python run_batch.py --ui                   # buka UI Gradio (browser)

Alur per dokumen:
    preprocess (AI-Document) -> PaddleOCR (support) -> selector ->
    NuExtract3-GGUF (main: gambar + ocr_text + schema) ->
    normalisasi + validasi -> results/<jenis>_<nama>.json

Model dimuat SEKALI (singleton) lalu dipakai ulang — untuk CLI maupun UI.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Pastikan root proyek ada di sys.path (jalankan dari folder nu-paddle).
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from models.registry import AVAILABLE_MODELS
from preprocessing.pipeline import preprocess
from preprocessing.pdf_to_image import pdf_to_image
from preprocessing.pillow_utils import save_array_as_image
from selection.selector import evaluate
from validation import correct_codes_from_ocr, normalise_fields, validate_document

SUPPORTED_EXTS = config.SUPPORTED_IMAGE_EXTS | {".pdf"}


# ---------------------------------------------------------------------------
# MODEL SINGLETON — dimuat SEKALI per proses, dipakai ulang (CLI & UI).
# Hanya memakai models.registry (aturan AGENT.md: tanpa import class spesifik).
# ---------------------------------------------------------------------------
_OCR: object | None = None
_EXTRACTOR: object | None = None
_LAYOUT: object | None = None


def get_ocr_model():
    global _OCR
    if _OCR is None:
        _OCR = AVAILABLE_MODELS["PaddleOCR"]()
        _OCR.load()
    return _OCR


def get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = AVAILABLE_MODELS["NuExtract3-GGUF"]()
        _EXTRACTOR.load()
    return _EXTRACTOR


def get_layout_model():
    """LayoutModel (PP-DocLayoutV3) singleton — dimuat sekali per proses."""
    global _LAYOUT
    if _LAYOUT is None:
        from models.layout_model import LayoutModel

        _LAYOUT = LayoutModel()
        _LAYOUT.load()
    return _LAYOUT


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _load_initial_array(input_path: str):
    """Render PDF halaman pertama / muat gambar sebagai array RGB."""
    if Path(input_path).suffix.lower() == ".pdf":
        return pdf_to_image(input_path)
    from PIL import Image

    import numpy as np

    with Image.open(input_path) as im:
        return np.asarray(im.convert("RGB"))


def prepare_image(input_path: str):
    """preprocess() wajib; jika gagal quality gate & tidak strict -> render mentah.

    Returns (processed_image_path | None, meta_dict).
    """
    res = preprocess(input_path, binarize=False)
    meta = {
        "preprocessed": res.processed_image_path is not None,
        "passed_quality_gate": res.passed_quality_gate,
        "blur_score": res.blur_score,
        "resolution": list(res.resolution) if res.resolution else None,
        "reject_reason": res.reject_reason,
    }
    if res.processed_image_path:
        return res.processed_image_path, meta

    # Quality gate gagal. Strict -> tolak; tidak strict -> tetap proses (anti miss).
    if config.STRICT_QUALITY_GATE:
        return None, meta
    try:
        image = _load_initial_array(input_path)
        out_dir = os.path.join(tempfile.gettempdir(), "nupaddle_fallback", uuid.uuid4().hex)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "processed.png")
        save_array_as_image(image, out_path)
        meta["preprocessed"] = True
        meta["preprocessed_skipped"] = True
        return out_path, meta
    except Exception as exc:  # noqa: BLE001
        meta["fallback_error"] = str(exc)
        return None, meta


def discover_documents(paths: list[str] | None = None) -> list[Path]:
    """Cari SEMUA dokumen contoh (atau pakai path yang diberikan user)."""
    if paths:
        return [Path(p) for p in paths if Path(p).suffix.lower() in SUPPORTED_EXTS]
    docs: list[Path] = []
    if config.SAMPLE_DIR.is_dir():
        for p in sorted(config.SAMPLE_DIR.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                docs.append(p)
    return docs


# ---------------------------------------------------------------------------
# PROSES SATU DOKUMEN
# ---------------------------------------------------------------------------
def _zone_ocr(image_path: str, ocr_model) -> dict:
    """OCR konteks tambahan: band header/logo (run_header) + zona tabel (layout).

    - Header/logo SELALU pakai wide-band crop (``run_header``): nama perusahaan
      di dalam logo terbukti lebih akurat dibaca lewat band lebar daripada bbox
      zona — teks logo sering melewati batas zona ``header_image``/``header``.
    - Zona ``table`` (PP-DocLayoutV3) dicrop + upscale (kapasitas dibatasi)
      agar angka kecil di tabel terbaca lebih baik.

    Returns
    -------
    dict
        ``zones`` (meta per zona, termasuk band header), ``all_lines``,
        ``kept_lines``.
    """
    zones_meta: list[dict] = []
    all_lines: list[dict] = []
    kept_lines: list[dict] = []

    # 1) Band header/logo — selalu dijalankan.
    header_res = ocr_model.run_header(image_path)
    hextra = header_res.extra or {}
    hctx = "\n".join(ln["text"] for ln in hextra.get("kept_lines") or [])
    zones_meta.append(
        {
            "label": "header",
            "box": None,
            "scale": config.LAYOUT_HEADER_FALLBACK_SCALE,
            "context_text": f"[ZONA: header]\n{hctx}" if hctx else "",
            "text": hextra.get("all_text", ""),
            "lines": hextra.get("all_lines", []),
            "elapsed_seconds": round(header_res.elapsed_seconds, 2),
            "error": header_res.error,
        }
    )
    all_lines.extend(hextra.get("all_lines") or [])
    kept_lines.extend(hextra.get("kept_lines") or [])

    # 2) Zona tabel dari deteksi layout (jika tersedia).
    # Model layout dimuat per kebutuhan lalu DI-UNLOAD segera: menghemat RAM
    # (~0.5GB) agar tidak menggeser peak melewati batas saat NuExtract berjalan.
    try:
        layout = get_layout_model()
        layout_zones = layout.detect_zones(image_path)
    except Exception:  # noqa: BLE001
        layout_zones = []
    finally:
        try:
            get_layout_model().unload()
        except Exception:  # noqa: BLE001
            pass
        gc.collect()

    for zone in layout_zones:
        if zone.label not in config.LAYOUT_ZONE_TARGETS:
            continue
        scale = config.LAYOUT_ZONE_SCALE.get(zone.label, 1.0)
        zres = ocr_model.run_region(image_path, zone.box, scale=scale, label=zone.label)
        extra = zres.extra or {}
        context = "\n".join(ln["text"] for ln in extra.get("kept_lines") or [])
        zones_meta.append(
            {
                "label": zone.label,
                "box": zone.box,
                "scale": scale,
                "context_text": f"[ZONA: {zone.label}]\n{context}" if context else "",
                "text": extra.get("all_text", ""),
                "lines": extra.get("all_lines", []),
                "elapsed_seconds": round(zres.elapsed_seconds, 2),
                "error": zres.error,
            }
        )
        all_lines.extend(extra.get("all_lines") or [])
        kept_lines.extend(extra.get("kept_lines") or [])

    return {
        "zones": zones_meta,
        "all_lines": all_lines,
        "kept_lines": kept_lines,
    }


def process_one(doc: Path, ocr_model, extractor) -> dict:
    t_start = time.time()
    doc_type = config.detect_document_type(doc)

    image_path, prep_meta = prepare_image(str(doc))
    if image_path is None:
        return {
            "document_id": str(uuid.uuid4()),
            "file_name": doc.name,
            "document_type": doc_type,
            "adapter": "nuextract3_gguf",
            "preprocessing": prep_meta,
            "fields": {},
            "normalised": {},
            "confidence": 0.0,
            "processing_time_ms": 0.0,
            "error": prep_meta.get("reject_reason") or "Gagal menyiapkan gambar",
            "validation": {"status": "FAILED", "field_errors": [], "rules": []},
            "ocr": {"included": False, "reason": "no image", "stats": {}},
        }

    # 1) PaddleOCR (support) — halaman penuh.
    ocr_res = ocr_model.run(image_path)
    ocr_extra = ocr_res.extra or {}

    # 1b) OCR konteks tambahan: band header/logo (run_header) + zona tabel (layout).
    zone_data = _zone_ocr(image_path, ocr_model)
    zones_meta = zone_data["zones"]
    zone_all_lines = zone_data["all_lines"]
    zone_kept_lines = zone_data["kept_lines"]
    header_meta = next((z for z in zones_meta if z.get("label") == "header"), {})

    # Gabung teks konteks: halaman penuh + zona berlabel.
    combined_text = "\n".join(ln["text"] for ln in ocr_extra.get("kept_lines") or [])
    combined_lines = list(ocr_extra.get("lines") or []) + list(zone_all_lines)
    combined_kept = list(ocr_extra.get("kept_lines") or []) + list(zone_kept_lines)
    zone_contexts = [z["context_text"] for z in zones_meta if z.get("context_text")]
    if zone_contexts:
        combined_text = f"{combined_text}\n" + "\n".join(zone_contexts)

    decision = evaluate(
        combined_text,
        lines=combined_lines,
        kept_lines=combined_kept,
    )
    ocr_text = decision.filtered_text if decision.include_text else ""

    # 2) NuExtract3-GGUF (main)
    # OCR/layout sudah tidak dibutuhkan di fase ini: unload + gc dulu supaya
    # peak RAM saat NuExtract dimuat/dijalankan tetap di bawah batas.
    try:
        ocr_model.unload()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    ext = extractor.run(image_path, doc_type=doc_type, ocr_text=ocr_text)

    fields = ext.fields if isinstance(ext.fields, dict) else {}
    # 3) Koreksi field kode/nomor dari OCR (PaddleOCR lebih akurat untuk kode).
    all_lines = list(ocr_extra.get("all_lines") or []) + list(zone_all_lines)
    fields = correct_codes_from_ocr(fields, all_lines)
    normalised = normalise_fields(fields)
    validation = validate_document(doc_type, normalised)
    elapsed_ms = round((time.time() - t_start) * 1000, 2)

    result = {
        "document_id": str(uuid.uuid4()),
        "file_name": doc.name,
        "document_type": doc_type,
        "adapter": "nuextract3_gguf",
        "preprocessing": prep_meta,
        "fields": fields,
        "normalised": normalised,
        "confidence": 0.9,
        "processing_time_ms": elapsed_ms,
        "validation": validation,
        "ocr": {
            "included": decision.include_text,
            "reason": decision.reason,
            "stats": decision.stats,
            "elapsed_seconds": round(ocr_res.elapsed_seconds, 2),
            "zones_elapsed_seconds": round(
                sum(z.get("elapsed_seconds", 0.0) for z in zones_meta), 2
            ),
            # "Ambil semua dari gambar": seluruh baris OCR (tanpa filter) ikut
            # disimpan agar tidak ada nomor/teks penting yang terlewat.
            "all_text": ocr_extra.get("all_text", ""),
            "lines": ocr_extra.get("all_lines", []),
            # OCR per-zona layout (band header + tabel).
            "zones": zones_meta,
            # Band header/logo (untuk nama perusahaan di dalam logo).
            "header_lines": header_meta.get("lines", []),
            "header_text": header_meta.get("text", ""),
        },
        "error": ext.error,
    }

    out_name = f"{doc_type}_{doc.stem}.json"
    out_path = config.RESULTS_DIR / out_name
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# BATCH (CLI & UI memakai fungsi yang sama)
# ---------------------------------------------------------------------------
def run(paths: list[str] | None = None) -> list[dict]:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = discover_documents(paths)
    if not documents:
        print(f"[SKIP] Tidak ada dokumen ditemukan di {config.SAMPLE_DIR}")
        return []

    print(f"[INFO] {len(documents)} dokumen akan diproses\n")
    for i, doc in enumerate(documents, 1):
        shown = doc.relative_to(config.SAMPLE_DIR) if str(doc).startswith(str(config.SAMPLE_DIR)) else doc
        print(f"  {i:>2}. {shown}")

    ocr_model = get_ocr_model()
    extractor = get_extractor()
    print("\n[INFO] Model siap (NuExtract3-GGUF main + PaddleOCR support). Memulai ekstraksi...\n")

    results: list[dict] = []
    for doc in documents:
        print(f"\n{'=' * 70}\nDOKUMEN: {doc.name}")
        result = process_one(doc, ocr_model, extractor)
        results.append(result)
        _print_result(result)

    # Ringkasan.
    print(f"\n{'=' * 70}\nRINGKASAN\n{'=' * 70}")
    n_ok = sum(1 for r in results if r["validation"]["status"] == "PASSED")
    for r in results:
        print(f"  [{r['validation']['status']:>6}] {r['document_type']:<16} {r['file_name']}")
    print(f"\n  PASSED: {n_ok}/{len(results)}   ->  hasil tersimpan di {config.RESULTS_DIR}")
    return results


def _print_result(result: dict) -> None:
    print(f"  jenis      : {result['document_type']}")
    print(f"  status     : {result['validation']['status']}")
    print(f"  waktu      : {result['processing_time_ms']} ms")
    if result.get("error"):
        print(f"  ERROR      : {result['error']}")
    ocr = result.get("ocr", {})
    print(f"  OCR support: {'Ya' if ocr.get('included') else 'Tidak'} ({ocr.get('reason', '')})")
    for key, value in (result.get("normalised") or {}).items():
        print(f"  {key:<22}: {value}")
    for fe in result["validation"].get("field_errors") or []:
        print(f"  [WARN] {fe['message']}")


# ---------------------------------------------------------------------------
# UI GRADIO (mirror pola AI-Document/app.py)
# ---------------------------------------------------------------------------
def _summarise(results: list[dict]) -> str:
    """Tabel ringkasan hasil dalam markdown."""
    if not results:
        return "_Tidak ada hasil._"
    n_ok = sum(1 for r in results if r["validation"]["status"] == "PASSED")
    header = f"**{n_ok}/{len(results)} dokumen PASSED** — hasil lengkap tersimpan di `{config.RESULTS_DIR}`\n\n"
    lines = ["| Dokumen | Jenis | Status | Waktu | Field utama |", "|---|---|---|---|---|"]
    for r in results:
        status = r["validation"]["status"]
        norm = r.get("normalised") or {}
        fields = " · ".join(f"{k}={v}" for k, v in list(norm.items())[:4])
        if not fields:
            fields = r.get("error") or "(kosong)"
        lines.append(
            f"| {r['file_name']} | {r['document_type']} | {status} | "
            f"{r['processing_time_ms'] / 1000:.0f}s | {fields} |"
        )
    return header + "\n".join(lines)


def build_ui():
    """Bangun UI Gradio. Hanya impor models.registry (aturan AGENT.md)."""
    import gradio as gr

def _resolve_upload_paths(files) -> list[str]:
    """Konversi output gr.File (str / FileData) ke path absolut yang valid."""
    paths = []
    for f in files or []:
        if isinstance(f, str):
            p = f
        elif hasattr(f, "path"):  # gr.FileData / gradio >=5
            p = f.path
        elif hasattr(f, "name"):
            p = f.name
        else:
            p = str(f)
        if Path(p).suffix.lower() in SUPPORTED_EXTS:
            paths.append(p)
    return paths


def build_ui():
    """Bangun UI Gradio. Hanya impor models.registry (aturan AGENT.md)."""
    import gradio as gr

    def extract_all() -> tuple[str, list[dict]]:
        try:
            results = run(None)  # SEMUA dokumen contoh
            return _summarise(results), results
        except Exception as exc:  # noqa: BLE001
            return f"_ERROR: {exc}_", [{"error": str(exc)}]

    def extract_uploaded(files) -> tuple[str, list[dict]]:
        paths = _resolve_upload_paths(files)
        if not paths:
            return "_Tidak ada dokumen valid pada upload. Format: PDF/JPG/PNG/BMP/TIF/WEBP._", []
        try:
            results = run(paths)
            return _summarise(results), results
        except Exception as exc:  # noqa: BLE001
            return f"_ERROR: {exc}_", [{"error": str(exc)}]

    with gr.Blocks(title="nu-paddle — Ekstraksi Dokumen") as demo:
        gr.Markdown(
            "# nu-paddle — Ekstraksi Dokumen\n"
            "**NuExtract3-GGUF** (model utama) + **PaddleOCR** (support) dipakai "
            "otomatis — tidak perlu memilih model. Upload dokumen, hasil seluruh "
            "field + teks OCR lengkap langsung keluar. Setiap dokumen melewati "
            "quality check lalu preprocessing AI-Document."
        )
        files = gr.File(
            file_count="multiple",
            label="Upload dokumen (PDF/gambar) — otomatis diproses",
        )
        with gr.Row():
            btn_all = gr.Button("Ekstrak Semua Contoh", variant="primary")
            btn_clear = gr.Button("Bersihkan")
        summary = gr.Markdown("_Upload dokumen atau klik 'Ekstrak Semua Contoh' untuk memulai._")
        details = gr.JSON(label="Hasil detail")

        # Auto-ekstrak begitu file di-upload (hasil sebelumnya langsung diganti,
        # tidak menampilkan hasil dokumen lain).
        # concurrency_limit=1 -> satu ekstraksi berjalan, sisanya antri (anti OOM).
        files.upload(extract_uploaded, inputs=files, outputs=[summary, details], concurrency_limit=1)
        btn_all.click(extract_all, outputs=[summary, details], concurrency_limit=1)
        btn_clear.click(
            lambda: ("_Siap. Upload dokumen untuk memulai._", None),
            outputs=[summary, details],
        )

    return demo

    return demo


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="nu-paddle: batch + UI ekstraksi dokumen")
    parser.add_argument("docs", nargs="*", help="path dokumen tertentu (opsional; default: semua contoh)")
    parser.add_argument("--ui", action="store_true", help="buka UI Gradio di browser")
    parser.add_argument("--host", default="127.0.0.1", help="host untuk --ui (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="port untuk --ui (default 7860)")
    parser.add_argument("--share", action="store_true", help="buat link publik sementara (gradio --share)")
    args = parser.parse_args()

    if args.ui:
        demo = build_ui()
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)
        return

    run(args.docs or None)


if __name__ == "__main__":
    main()
