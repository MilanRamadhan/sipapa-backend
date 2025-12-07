"""
Re-scrape artikel DENGAN gambar untuk update data existing.
Script ini akan:
1. Baca scraped.csv
2. Ambil 100 URL pertama
3. Scrape ulang dengan ekstraksi gambar
4. Update scraped.csv dengan image_url baru
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from pathlib import Path

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
        res = requests.get(url, headers=HEADERS, timeout=15)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            image_url = extract_images(soup)
            return image_url
        return ""
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return ""

print("[INFO] Membaca scraped.csv...")
df = pd.read_csv(SCRAPED_FILE)
print(f"[INFO] Total rows: {len(df)}")

# Ambil 100 URL pertama untuk test
test_urls = df.head(100)

print(f"\n[INFO] Akan scrape ulang {len(test_urls)} artikel untuk ambil gambar...")
print("[INFO] Estimasi waktu: ~30 detik\n")

updated_count = 0
for idx, row in test_urls.iterrows():
    url = row['url']
    current_image = row.get('image_url', '')
    
    # Skip jika sudah ada gambar
    if pd.notna(current_image) and str(current_image).strip() and str(current_image) != 'nan':
        print(f"[{idx+1}/{len(test_urls)}] ✓ Skip (sudah ada gambar)")
        continue
    
    print(f"[{idx+1}/{len(test_urls)}] Scraping: {url[:60]}...")
    
    image_url = scrape_image(url)
    
    if image_url:
        df.at[idx, 'image_url'] = image_url
        updated_count += 1
        print(f"   ✓ Gambar ditemukan: {image_url[:60]}...")
    else:
        df.at[idx, 'image_url'] = ""
        print(f"   ✗ Tidak ada gambar")
    
    time.sleep(0.3)  # Jeda sopan

print(f"\n[SUCCESS] Scraping selesai!")
print(f"[INFO] Updated: {updated_count}/{len(test_urls)} artikel dengan gambar")

# Save
print(f"\n[SAVE] Menyimpan ke {SCRAPED_FILE}...")
df.to_csv(SCRAPED_FILE, index=False, encoding='utf-8-sig')

print("\n[PREVIEW] Sample hasil:")
for idx, row in df.head(5).iterrows():
    has_img = "✓" if row.get('image_url') and str(row.get('image_url')) != 'nan' else "✗"
    print(f"{idx+1}. {has_img} {row['title'][:50]}")
    if row.get('image_url') and str(row.get('image_url')) != 'nan':
        print(f"   Image: {str(row['image_url'])[:70]}")

print("\n[NEXT STEPS]")
print("1. python quick_corpus_clean.py  # Regenerate corpus_clean.csv")
print("2. python quick_indexing.py      # Regenerate index")
print("3. python api.py                  # Restart backend")
