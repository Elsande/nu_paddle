"""
extraction/schemas.py
=====================
Schema JSON dan instruksi per jenis dokumen untuk NuExtract3-GGUF.

Diambil dari notebook ``doc-validation/cek_nuextract3_gguf.ipynb`` yang sudah
terbukti menghasilkan ekstraksi akurat pada contoh dokumen finance.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# SCHEMA JSON per jenis dokumen
# ---------------------------------------------------------------------------
_SCHEMAS: dict[str, dict] = {
    "invoice": {
        "invoice_number": "verbatim-string",
        "invoice_date": "date",
        "due_date": "date",
        "po_number": "verbatim-string",
        "po_date": "date",
        "supplier_name": "verbatim-string",
        "customer_name": "verbatim-string",
        "total_amount": "number",
        "tax_amount": "number",
        "currency": "currency",
        "line_items": [
            {
                "description": "verbatim-string",
                "quantity": "number",
                "unit_price": "number",
                "amount": "number",
            }
        ],
    },
    "purchase_order": {
        "po_number": "verbatim-string",
        "po_date": "date",
        "cf_code": "verbatim-string",
        "supplier_name": "verbatim-string",
        "buyer_name": "verbatim-string",
        "total_amount": "number",
        "currency": "currency",
    },
    "delivery_order": {
        "do_number": "verbatim-string",
        "do_date": "date",
        "po_number": "verbatim-string",
        "po_date": "date",
        "supplier_name": "verbatim-string",
        "recipient_name": "verbatim-string",
        "delivery_address": "verbatim-string",
        "total_items": "number",
    },
    "kwitansi": {
        "kwitansi_number": "verbatim-string",
        "kwitansi_date": "date",
        "company_name": "verbatim-string",
        "invoice_reference": "verbatim-string",
        "total_value": "number",
        "materai_present": "boolean",
    },
    "tax_invoice": {
        "tax_invoice_number": "verbatim-string",
        "invoice_date": "date",
        "supplier_name": "verbatim-string",
        "supplier_tax_number": "verbatim-string",
        "total_amount": "number",
        "tax_amount": "number",
    },
}


def build_schema(doc_type: str) -> str:
    """Return the JSON schema template string for *doc_type* (default: invoice)."""
    return json.dumps(_SCHEMAS.get(doc_type, _SCHEMAS["invoice"]), indent=2)


# ---------------------------------------------------------------------------
# Petunjuk field & layout per jenis dokumen (Bahasa Indonesia)
# ---------------------------------------------------------------------------
_FIELD_HINTS_ID: dict[str, dict[str, str]] = {
    "invoice": {
        "invoice_number": "No. Invoice",
        "invoice_date": "Tanggal Invoice",
        "due_date": "Jatuh Tempo (isi HANYA jika ada tanggal eksplisit di samping label 'Tgl Jatuh Tempo'; jika tidak ada, kosongkan)",
        "po_number": "No. PO/SPK",
        "po_date": "Tanggal PO/SPK",
        "supplier_name": "Nama Penjual / Pemasok",
        "customer_name": "Nama Pembeli / Pelanggan",
        "total_amount": "Total / Jumlah",
        "tax_amount": "Pajak",
        "currency": "Mata Uang",
        "line_items": "Rincian Barang",
    },
    "purchase_order": {
        "po_number": "No. PO",
        "po_date": "Tanggal PO",
        "cf_code": "CF Code (kode di header PO, di samping PO.NO.; GABUNGKAN semua fragmen baris di bawah label CF Code menjadi satu string tanpa spasi)",
        "supplier_name": "Nama Pemasok / Supplier",
        "buyer_name": "Nama Pembeli",
        "total_amount": "Total",
        "currency": "Mata Uang",
    },
    "delivery_order": {
        "do_number": "No. Surat Jalan / DO",
        "do_date": "Tanggal",
        "po_number": "No. PO/SPK",
        "po_date": "Tanggal PO/SPK",
        "supplier_name": "Nama Pengirim",
        "recipient_name": "Nama Penerima",
        "delivery_address": "Alamat Pengiriman",
        "total_items": "Jumlah Barang",
    },
    "kwitansi": {
        "kwitansi_number": "No. Kwitansi (hanya jika dokumen punya nomor kwitansi tersendiri; kalau tidak ada, KOSONGKAN — jangan pakai nomor invoice)",
        "kwitansi_date": "Tanggal Kwitansi",
        "company_name": "Nama Perusahaan",
        "invoice_reference": "No. Referensi / No. Invoice",
        "total_value": "Nilai / Jumlah",
        "materai_present": "Ada Materai atau Tidak",
    },
    "tax_invoice": {
        "tax_invoice_number": "No. Faktur Pajak",
        "invoice_date": "Tanggal",
        "supplier_name": "Nama Penjual / PKP",
        "supplier_tax_number": "NPWP Penjual",
        "total_amount": "DPP / Total",
        "tax_amount": "PPN",
    },
}

_LAYOUT_HINTS_ID: dict[str, str] = {
    "invoice": "Penjual/pemasok (supplier_name) = nama perusahaan di kop/header atau bagian 'Dari'; "
    "pembeli/pelanggan (customer_name) = nama pada bagian 'Kepada'/'Billed To'. "
    "invoice_date = tanggal terbit invoice (label 'Print Dated'/'Tanggal'), BUKAN 'Due Date'. "
    "due_date = hanya jika ada tanggal jelas di samping label 'Tgl Jatuh Tempo'/'Due Date'; "
    "kalau tidak ada, KOSONGKAN — jangan menyalin Tanggal PO/SPK atau tanggal lain. "
    "No. PO/SPK = nomor referensi PO di header (di samping label 'No. PO/SPK'); "
    "Tanggal PO/SPK = tanggal PO/SPK, BUKAN tanggal invoice.",
    "purchase_order": "Pada PO: pihak yang dituju di bagian 'To'/'Kepada' adalah PEMASOK (supplier_name); "
    "pihak pada 'Ship To'/'Dikirim ke'/'Kepada Yth' adalah PEMBELI (buyer_name). "
    "CF Code = seluruh string pada baris berlabel 'CF Code' di header, ambil utuh tanpa dipotong.",
    "delivery_order": "Pada surat jalan: pengirim/penjual = supplier_name; penerima/pembeli = recipient_name. "
    "No. PO/SPK = nomor PO/SPK yang dirujuk; Tanggal PO/SPK = tanggal PO/SPK, "
    "BUKAN tanggal surat jalan (do_date adalah 'Tanggal').",
    "kwitansi": "kwitansi_number = No. Kwitansi, company_name = nama perusahaan penerbit kwitansi.",
    "tax_invoice": "supplier_name = nama penjual (PKP) di bagian 'Penjual'; "
    "tax_invoice_number = nomor seri faktur pajak; total_amount = DPP, tax_amount = PPN.",
}


def build_instructions(doc_type: str, lang: str = "id") -> str:
    """Petunjuk untuk model sesuai jenis dokumen (default: Bahasa Indonesia)."""
    if lang != "id":
        return (
            "Extract all values exactly as written on the document. "
            "Do not translate or reformat the values."
        )
    hints = _FIELD_HINTS_ID.get(doc_type, {})
    layout = _LAYOUT_HINTS_ID.get(doc_type, "")
    lines = [
        "Dokumen ini berbahasa Indonesia. "
        "Ekstrak seluruh nilai PERSIS seperti tertulis pada dokumen "
        "(jangan diterjemahkan atau mengubah format angka/tanggal)."
    ]
    if hints:
        pairs = ", ".join(f"{k} ({h})" for k, h in hints.items())
        lines.append("Kolom yang dicari: " + pairs + ".")
    if layout:
        lines.append(layout)
    lines.append(
        "Jika ada teks OCR pendukung yang disediakan, teks OCR adalah sumber "
        "paling akurat untuk nilai PERSIS (kode, angka, tanggal, nama) — "
        "gunakan nilai dari teks OCR bila tersedia; gambar dokumen hanya dipakai "
        "untuk memastikan layout bila teks OCR tidak jelas."
    )
    return " ".join(lines)
