"""Tahap 1: Konversi PDF -> gambar.

Fase 1 HANYA menangani halaman pertama dokumen PDF (multi-halaman di luar
scope). Render dilakukan lewat PyMuPDF (pymupdf) yang self-contained dan
tidak butuh binary poppler tambahan.

Zoom render dipilih agar lebar output berada di sekitar MIN_RENDER_WIDTH
piksel, cukup untuk keterbacaan OCR tanpa terlalu besar sehingga lambat.
"""

from __future__ import annotations

import numpy as np
import pymupdf  # PyMuPDF


# Lebar target (piksel) hasil render halaman PDF.
MIN_RENDER_WIDTH = 1800


def pdf_to_image(pdf_path: str) -> np.ndarray:
    """Render halaman pertama PDF menjadi array numpy ber-channel RGB."""
    doc = pymupdf.open(pdf_path)
    try:
        if doc.page_count < 1:
            raise ValueError("PDF tidak memiliki halaman sama sekali.")
        page = doc.load_page(0)

        # Hitung zoom agar lebar render >= MIN_RENDER_WIDTH.
        base_width = page.rect.width
        zoom = max(1.0, MIN_RENDER_WIDTH / base_width)
        matrix = pymupdf.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8)
        image = image.reshape(pix.height, pix.width, pix.n)
        # pix.n == 3 untuk RGB, == 1 untuk grayscale. Normalisasi ke RGB.
        if pix.n == 1:
            image = np.repeat(image, 3, axis=2)
        return image
    finally:
        doc.close()
