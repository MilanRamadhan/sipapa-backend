"""
Backfill SEMUA 4852 artikel dengan image URL.
Hanya scrape gambar, tidak perlu content parsing.
Lebih cepat dari full re-scrape.
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
import re

BASE_DIR = Path(__file__).parent
SCRAPED_FILE = BASE_DIR / "data" / "scraped.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def extract_images(soup):
    """Ekstrak URL gambar dari artikel"""
    # Prioritas 1: Open Graph image
    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content"):
        return og_image["content"].strip()
    
    # Prioritas 2: Twitter card
    twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
    if twitter_image and twitter_image.get("content"):
        return twitter_image["content"].strip()
    
    # Prioritas 3: First image in article
    article_containers = soup.find_all(["article", "div"], class_=lambda x: x and ('article' in x.lower() or 'content' in x.lower()))
    for container in article_containers:
        img = container.find("img", src=True)
        if img and img.get("src"):
            src = img["src"].strip()
            if not src.startswith('data:') and 'icon' not in src.lower() and 'logo' not in src.lower():
                return src
    
    # Fallback: first image
    img = soup.find("img", src=True)
    if img and img.get("src"):
        src = img["src"].strip()
        if not src.startswith('data:'):
            return src
    
    return ""

def scrape_image(url):
    """Scrape gambar dari satu URL"""
    try:
        # Remove query params untuk kompas.com
        clean_url = re.sub(r'\?page=.*', '', url)
        
        res = requests.get(clean_url, headers=HEADERS, timeout=15)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            image_url = extract_images(soup)
            return image_url
        return ""
    except Exception as e:
        return ""

print(f"[INFO] Membaca {SCRAPED_FILE}...")
df = pd.read_csv(SCRAPED_FILE)
total = len(df)
print(f"[INFO] Total artikel: {total}")

# Hitung berapa yang belum punya gambar
no_image = df['image_url'].isna() | (df['image_url'] == "")
need_scrape = no_image.sum()
print(f"[INFO] Artikel tanpa gambar: {need_scrape}/{total}")

if need_scrape == 0:
    print("[INFO] Semua artikel sudah punya gambar!")
    exit(0)

print(f"\n[START] Scraping {need_scrape} artikel...")
print(f"[ESTIMATE] Waktu: ~{need_scrape * 0.3 / 60:.1f} menit\n")

updated_count = 0
failed_count = 0
start_time = time.time()

for idx, row in df[no_image].iterrows():
    url = row['url']
    
    progress = updated_count + failed_count + 1
    pct = (progress / need_scrape) * 100
    
    print(f"[{progress}/{need_scrape}] ({pct:.1f}%) Scraping doc_id {idx}...")
    
    image_url = scrape_image(url)
    
    if image_url:
        df.at[idx, 'image_url'] = image_url
        updated_count += 1
        print(f"   ✓ {image_url[:70]}...")
    else:
        df.at[idx, 'image_url'] = ""
        failed_count += 1
        print(f"   ✗ No image")
    
    # Progress setiap 100 artikel
    if progress % 100 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / progress
        remaining = (need_scrape - progress) * avg_time
        print(f"\n   📊 Progress: {updated_count} success, {failed_count} failed")
        print(f"   ⏱️  Elapsed: {elapsed/60:.1f}m | Remaining: ~{remaining/60:.1f}m\n")
    
    time.sleep(0.25)  # Rate limit

# Final stats
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"[SUCCESS] Backfill selesai!")
print(f"[STATS]")
print(f"   ✓ Berhasil: {updated_count}/{need_scrape}")
print(f"   ✗ Gagal: {failed_count}/{need_scrape}")
print(f"   ⏱️  Waktu: {elapsed/60:.1f} menit")
print(f"   📈 Speed: {need_scrape/elapsed:.1f} artikel/detik")

# Save
print(f"\n[SAVE] Menyimpan ke {SCRAPED_FILE}...")
df.to_csv(SCRAPED_FILE, index=False, encoding='utf-8-sig')

# Sample preview
has_img = ~df['image_url'].isna() & (df['image_url'] != "")
pct_with_img = (has_img.sum() / total) * 100

print(f"\n[RESULT] {has_img.sum()}/{total} artikel ({pct_with_img:.1f}%) punya gambar")

print("\n[NEXT STEPS]")
print("1. python quick_corpus_clean.py")
print("2. python quick_indexing.py")
print("3. Restart backend API")
