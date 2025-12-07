"""
Script cepat untuk membuat corpus_clean.csv dari scraped_cleaned.csv
Langsung copy karena konten sudah dibersihkan, tinggal tambah image_url ke output
"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "scraped.csv"
OUTPUT_FILE = BASE_DIR / "data" / "corpus_clean.csv"

print(f"[INFO] Membaca: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"[INFO] Total baris: {len(df)}")

# Rename kolom untuk match dengan expected format
# content -> content_raw, word_count -> word_count_raw
df_clean = df[['url', 'title']].copy()
df_clean['image_url'] = df['image_url'] if 'image_url' in df.columns else ""
df_clean['word_count_raw'] = df['word_count']
df_clean['word_count_clean'] = df['word_count']  # sama aja karena sudah clean
df_clean['content_raw'] = df['content']
df_clean['content_clean'] = df['content']  # sama aja karena sudah clean

print(f"[INFO] Menyimpan ke: {OUTPUT_FILE}")
df_clean.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"\n[SUCCESS] Selesai!")
print(f"Total dokumen: {len(df_clean)}")
print(f"Kolom: {list(df_clean.columns)}")
print(f"\nPreview:")
print(df_clean.head(2))
