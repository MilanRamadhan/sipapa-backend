import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def clean_text(raw) -> str:
    """
    Bersihin teks artikel dari scraped.csv jadi content_final.

    Fokus:
    - Hapus baris iklan / membership / download app (kalau ada)
    - Hapus 'Baca juga:' dan link terkait
    - Rapihin double quotes ""contoh"" -> "contoh"
    - Hilangkan bullet 'O ' di awal baris
    - Perbaiki kata nempel karena tag <a> hilang (wilayahPurworejo -> wilayah Purworejo)
    - Rapihin newline & spasi
    """
    if pd.isna(raw):
        raw = ""
    text = str(raw)

    # Normalisasi newline
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    clean_lines: list[str] = []

    for line in text.split("\n"):
        original = line
        line = line.strip()
        if not line:
            continue

        lower = line.lower()

        # Buang keterangan membership / aplikasi kalau nyangkut di konten
        if lower.startswith("membership:"):
            continue
        if lower.startswith("download aplikasi:"):
            continue

        # Buang baris yang berisi 'baca juga: ...'
        if "baca juga:" in lower:
            continue

        # Hapus bullet 'O ' di awal baris (hasil copy symbol list)
        line = re.sub(r"^O\s+", "", line)

        # Rapihin double double-quote -> single
        # ""teks"" -> "teks"
        line = re.sub(r'""([^"]+?)""', r'"\1"', line)

        # Rapihin spasi berlebihan
        line = re.sub(r"\s+", " ", line)

        # Perbaiki kata nempel karena <a> ketemu <p> (ex: wilayahPurworejo)
        line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)

        clean_lines.append(line)

    # Gabung lagi, pisah paragraf pakai newline
    text = "\n".join(clean_lines)

    # Rapihin newline ganda di akhir proses
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
    return text


def main():
    src_path = DATA_DIR / "scraped.csv"
    if not src_path.exists():
        raise FileNotFoundError(f"scraped.csv tidak ditemukan di: {src_path}")

    df = pd.read_csv(src_path)

    if "url" not in df.columns or "content" not in df.columns:
        raise ValueError("scraped.csv harus punya kolom 'url' dan 'content'")

    df["content_final"] = df["content"].apply(clean_text)

    output_cols = []
    for col in ["url", "title", "image_url"]:
        if col in df.columns:
            output_cols.append(col)
    output_cols.append("content_final")

    out_df = df[output_cols].copy()

    out_path = DATA_DIR / "corpus_clean_v2.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[OK] Saved cleaned corpus to {out_path} (rows: {len(out_df)})")


if __name__ == "__main__":
    main()
