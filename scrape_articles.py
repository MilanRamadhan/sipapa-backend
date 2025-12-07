import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urlparse

# ============================
# PATH FILE
# ============================
BASE_DIR = Path(__file__).parent
URL_FILE = BASE_DIR / "data" / "urls.txt"
OUTPUT_FILE = BASE_DIR / "data" / "scraped.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ============================
# PARAMETER SCRAPING
# ============================
REQUEST_TIMEOUT = 15  # detik
SLEEP_BETWEEN_REQUESTS = 0.2  # jeda antar request biar sopan
MIN_WORDS = 40  # minimal jumlah kata supaya artikel dianggap valid

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ============================
# BACA URL DARI urls.txt
# ============================
def load_urls(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"File URL tidak ditemukan: {path}")

    with path.open(encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]

    seen = set()
    urls = []
    for u in raw:
        if u not in seen:
            seen.add(u)
            urls.append(u)

    print(f"[INFO] Total URL di {path.name}: {len(urls)}")
    return urls


# ============================
# FILTER: CUMA AMBIL URL ARTIKEL
# ============================
def is_article_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path or ""

    blocked_fragments = [
        "/copy/",
        "/komentar/",
        "/image/",
        "/search/",
    ]
    if any(b in path for b in blocked_fragments):
        return False

    if path in ["", "/"]:
        return False

    return True


# ============================
# CLEAN KONTEN LEVEL SCRAPER
# (sedikit aja, sisanya di clean_corpus_v2)
# ============================
def basic_clean(content: str) -> str:
    if not content:
        return ""

    # Hapus spasi berlebihan
    content = re.sub(r"[ \t]+", " ", content)
    # Normalisasi newline: max 2 newline berturut-turut
    content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
    return content.strip()


# ============================
# EXTRACT GAMBAR ARTIKEL
# ============================
def extract_images(soup: BeautifulSoup) -> str:
    og_image = soup.find("meta", attrs={"property": "og:image"})
    if og_image and og_image.get("content"):
        return og_image["content"].strip()

    twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
    if twitter_image and twitter_image.get("content"):
        return twitter_image["content"].strip()

    article_containers = soup.find_all(
        ["article", "div"],
        class_=lambda x: x and ("article" in x.lower() or "content" in x.lower()),
    )
    for container in article_containers:
        img = container.find("img", src=True)
        if img and img.get("src"):
            src = img["src"].strip()
            if (
                not src.startswith("data:")
                and "icon" not in src.lower()
                and "logo" not in src.lower()
            ):
                return src

    img = soup.find("img", src=True)
    if img and img.get("src"):
        src = img["src"].strip()
        if not src.startswith("data:"):
            return src

    return ""


# ============================
# EXTRACT TEKS DARI HTML
# ============================
def extract_from_kompas_read_content(soup: BeautifulSoup) -> str:
    """
    Khusus halaman Kompas (travel.kompas.com):
    ambil teks dari div.read__content → p + h2
    (urutan sesuai artikel, iframe/photo/ads diskip)
    """
    container = soup.find("div", class_=lambda x: x and "read__content" in x)
    if not container:
        return ""

    blocks: list[str] = []

    # Ambil semua <p> dan <h2> di dalam read__content
    for el in container.find_all(["p", "h2"], recursive=True):
        text = el.get_text(" ", strip=True)
        if not text:
            continue

        # Kadang ada p kosong hasil wrapper ads, skip
        if text.lower().startswith("membership:"):
            continue

        blocks.append(text)

    return "\n".join(blocks)


def extract_generic_paragraphs(soup: BeautifulSoup) -> str:
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if p]
    return "\n".join(paragraphs)


def extract_article(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # ===== Title =====
    title = None
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()

    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

    if not title:
        title = "No Title"

    domain = urlparse(url).netloc

    # ===== Content =====
    if "kompas.com" in domain:
        content = extract_from_kompas_read_content(soup)
        # fallback kalau tiba-tiba struktur berubah
        if not content:
            content = extract_generic_paragraphs(soup)
    else:
        content = extract_generic_paragraphs(soup)

    content = basic_clean(content)
    word_count = len(content.split())
    image_url = extract_images(soup)

    return {
        "url": url,
        "domain": domain,
        "title": title,
        "content": content,
        "image_url": image_url,
        "word_count": word_count,
        "timestamp": time.time(),
    }


# ============================
# HTTP REQUEST
# ============================
def fetch(url: str):
    try:
        res = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if res.status_code == 200:
            return res.text
        print(f"[STATUS {res.status_code}] {url}")
        return None
    except Exception as e:
        print(f"[ERROR] {url} -> {e}")
        return None


# ============================
# SCRAPE SATU URL
# ============================
def scrape_one(url: str):
    print(f"[SCRAPE] {url}")
    html = fetch(url)
    if not html:
        return None

    article = extract_article(url, html)

    if not article["content"] or article["word_count"] < MIN_WORDS:
        print(f"[SKIP - KONTEN PENDEK] {url}")
        return None

    print(f"[OK] {url} (≈ {article['word_count']} kata)")
    return article


# ============================
# MAIN LOOP
# ============================
def scrape_all():
    urls = load_urls(URL_FILE)

    article_urls = [u for u in urls if is_article_url(u)]
    print(f"[INFO] URL yang lolos filter pola artikel: {len(article_urls)}")

    results = []
    start_time = time.time()

    for idx, url in enumerate(article_urls, start=1):
        print(f"\n[{idx}/{len(article_urls)}]")
        data = scrape_one(url)
        if data:
            results.append(data)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Scraping selesai dalam {elapsed:.2f} detik.")
    print(f"[INFO] Total artikel valid: {len(results)}")
    return results


# ============================
# ENTRY POINT
# ============================
if __name__ == "__main__":
    print(f"[INFO] Working Directory : {BASE_DIR}")
    print(f"[INFO] URL source       : {URL_FILE}")
    print(f"[INFO] Output CSV       : {OUTPUT_FILE}\n")

    results = scrape_all()

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"\n[SUCCESS] Scraping selesai! File disimpan ke: {OUTPUT_FILE}")
        print(f"[INFO] Total data berhasil di-scrape: {len(results)}")
        print("\n[PREVIEW] 5 data pertama:")
        print(df.head())
    else:
        print("\n[WARNING] Tidak ada data yang berhasil di-scrape!")
