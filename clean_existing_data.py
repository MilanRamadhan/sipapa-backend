"""
Script untuk membersihkan konten di scraped.csv yang sudah ada.
- Hapus judul duplikat
- Hapus 'Baca juga:'
- Hapus caption Instagram
- Hapus copyright
- Tambah kolom image_url (kosong untuk data lama)
"""
import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "scraped.csv"
OUTPUT_FILE = BASE_DIR / "data" / "scraped_cleaned.csv"

print(f"[INFO] Membaca: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"[INFO] Total baris: {len(df)}")

def clean_content(content: str, title: str) -> str:
    """Bersihkan konten dari noise"""
    if pd.isna(content) or not str(content).strip():
        return ""
    
    content = str(content)
    title = str(title) if not pd.isna(title) else ""
    
    # Hapus judul duplikat di awal (bisa muncul 1-2x)
    if title:
        title_lower = title.strip().lower()
        # Split by line dan remove lines yang match dengan title
        lines = content.split('\n')
        cleaned_lines = []
        title_removed_count = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Skip judul yang sama persis atau sangat mirip (di 5 baris pertama)
            if i < 5 and title_removed_count < 2:
                if line_lower == title_lower:
                    title_removed_count += 1
                    continue
                # Juga skip jika line mengandung title dan panjangnya mirip (+/- 20 char)
                if title_lower in line_lower and abs(len(line_lower) - len(title_lower)) < 20:
                    title_removed_count += 1
                    continue
            
            # Skip "Editor", "Tim Redaksi", "Reporter", "Penulis" di baris awal
            if i < 3 and line_lower in ['editor', 'tim redaksi', 'reporter', 'penulis']:
                continue
                
            cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)
    
    # Hapus "Baca juga:" dan seluruh barisnya
    content = re.sub(r'^Baca juga:.*$', '', content, flags=re.MULTILINE|re.IGNORECASE)
    
    # Hapus caption Instagram
    content = re.sub(r'Sebuah kiriman dibagikan oleh.*?(?:\n|$)', '', content, flags=re.IGNORECASE)
    
    # Hapus copyright footer (berbagai format)
    content = re.sub(r'Copyright \d{4}.*?All Rights Reserved\.?', '', content, flags=re.IGNORECASE)
    content = re.sub(r'©.*?\d{4}.*?(?:\n|$)', '', content, flags=re.IGNORECASE)
    content = re.sub(r'Copyright.*?Kompas.*?Reserved\.?', '', content, flags=re.IGNORECASE)
    
    # Hapus caption gambar (KOMPAS.COM/NAMA atau DETIK.COM/NAMA)
    content = re.sub(r'[A-Z]{2,}\.[A-Z]{2,}/[A-Z]+[A-Z\s]*(?=\n|[A-Z][a-z])', '', content)
    
    # Rapikan whitespace berlebih
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # max 2 newlines
    content = re.sub(r' +', ' ', content)  # single space
    content = re.sub(r'\n ', '\n', content)  # remove leading space after newline
    content = content.strip()
    
    return content

print("[INFO] Membersihkan konten...")
df['content'] = df.apply(lambda row: clean_content(row['content'], row['title']), axis=1)

# Tambah kolom image_url (kosong untuk data lama)
if 'image_url' not in df.columns:
    df['image_url'] = ""
    print("[INFO] Menambahkan kolom image_url (kosong)")

# Update word_count setelah cleaning
df['word_count'] = df['content'].str.split().str.len()

print(f"[INFO] Menyimpan ke: {OUTPUT_FILE}")
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"\n[SUCCESS] Selesai!")
print(f"Total baris: {len(df)}")
print(f"Rata-rata kata per artikel: {df['word_count'].mean():.0f}")

# Preview beberapa hasil
print("\n[PREVIEW] 3 artikel pertama:")
for idx, row in df.head(3).iterrows():
    print(f"\n{idx+1}. {row['title'][:80]}")
    print(f"   Kata: {row['word_count']}")
    print(f"   Content: {row['content'][:200]}...")
