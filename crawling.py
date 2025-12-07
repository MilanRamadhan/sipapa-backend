# crawling.py
# Tahap 1: CRAWLING SAJA (kumpulkan link)
# Input : SEEDS & ALLOWED_DOMAINS dari config.py
# Output: data/urls.txt (1 URL per baris)
# Di akhir: print total URL + lama waktu crawling

import asyncio
import aiohttp
import os
import time
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup

from config import (
    SEEDS,
    ALLOWED_DOMAINS,
    MAX_URLS,
    MAX_CONCURRENT_TASKS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
)

OUTPUT_PATH = "data/urls.txt"
os.makedirs("data", exist_ok=True)

CRAWL_LIMIT = MAX_URLS  # sekarang = 1_000_000 dari config.py


def allowed_domain(url: str) -> bool:
    """Cek apakah domain URL termasuk yang diizinkan."""
    try:
        domain = urlparse(url).netloc
        return any(domain.endswith(allow) for allow in ALLOWED_DOMAINS)
    except Exception:
        return False


async def fetch(session: aiohttp.ClientSession, url: str, retry: int = 0):
    """Ambil HTML dari suatu URL dengan retry sederhana."""
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT) as res:
            if res.status == 200:
                return await res.text()
            else:
                print(f"[STATUS {res.status}] {url}")
                return None
    except Exception as e:
        if retry < MAX_RETRIES:
            print(f"[RETRY {retry+1}] {url} -> {e}")
            return await fetch(session, url, retry + 1)
        print(f"[FAILED] {url} -> {e}")
        return None


async def append_url(url: str, file_lock: asyncio.Lock):
    """Tulis satu URL ke data/urls.txt (append, aman untuk multi-task)."""
    async with file_lock:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(url.strip() + "\n")


async def worker(
    name: int,
    queue: asyncio.Queue,
    visited: set,
    file_lock: asyncio.Lock,
):
    async with aiohttp.ClientSession() as session:
        while True:
            url = await queue.get()

            # Sentinel stop
            if url is None:
                queue.task_done()
                print(f"[WORKER {name}] stop.")
                break

            # Kalau sudah kena limit, skip
            if len(visited) >= CRAWL_LIMIT:
                queue.task_done()
                continue

            # Skip jika sudah pernah dikunjungi
            if url in visited:
                queue.task_done()
                continue

            visited.add(url)

            # Pastikan domain diizinkan
            if not allowed_domain(url):
                queue.task_done()
                continue

            print(f"[WORKER {name}] CRAWL -> {url}")

            # Simpan URL ke file
            await append_url(url, file_lock)

            # Ambil HTML hanya untuk cari link baru
            html = await fetch(session, url)
            if not html:
                queue.task_done()
                continue

            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                new_url = urljoin(url, a["href"])

                if (
                    allowed_domain(new_url)
                    and new_url not in visited
                    and len(visited) + queue.qsize() < CRAWL_LIMIT
                ):
                    await queue.put(new_url)

            queue.task_done()


async def main():
    # Bersihkan urls.txt lama kalau ada
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    start_time = time.time()

    queue = asyncio.Queue()
    visited: set[str] = set()
    file_lock = asyncio.Lock()

    # Masukkan SEEDS ke antrian
    for seed in SEEDS:
        await queue.put(seed)

    # Buat worker
    workers = []
    n_workers = MAX_CONCURRENT_TASKS
    print(f"[INFO] Mulai crawling dengan {n_workers} worker, limit {CRAWL_LIMIT} URL")
    for i in range(n_workers):
        task = asyncio.create_task(worker(i + 1, queue, visited, file_lock))
        workers.append(task)

    # Tunggu sampai antrian kosong
    await queue.join()

    # Kirim sinyal stop ke semua worker
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers, return_exceptions=True)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n=== CRAWLING SELESAI ===")
    print(f"Total URL dikunjungi: {len(visited)}")
    print(f"File output         : {OUTPUT_PATH}")
    print(
        f"Total waktu         : {hours} jam {minutes} menit {seconds} detik "
        f"({elapsed:.2f} detik)"
    )


if __name__ == "__main__":
    asyncio.run(main())
