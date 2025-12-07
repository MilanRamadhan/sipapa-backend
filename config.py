# config.py

# ============ SEEDS: MULAI DARI HALAMAN TRAVEL AJA ============
SEEDS = [
    # ===== detikTravel =====
    "https://travel.detik.com/",
    "https://travel.detik.com/domestic-destination",
    "https://travel.detik.com/international-destination",
    "https://travel.detik.com/travel-news",
    "https://travel.detik.com/travel-tips",
    "https://travel.detik.com/travel-deals",
    "https://travel.detik.com/travel-food",

    # Tag yang jelas-jelas wisata
    "https://travel.detik.com/tag/wisata",
    "https://travel.detik.com/tag/destinasi",
    "https://travel.detik.com/tag/liburan",
    "https://travel.detik.com/tag/hotel",
    "https://travel.detik.com/tag/pantai",

    # ===== Kompas Travel =====
    "https://travel.kompas.com/",
    "https://travel.kompas.com/read",
    "https://travel.kompas.com/search?q=wisata",
    "https://travel.kompas.com/search?q=liburan",
    "https://travel.kompas.com/search?q=hotel",
    "https://travel.kompas.com/search?q=kuliner",
]

# ============ LIMIT DLL ============

MAX_URLS = 10_000     
MAX_CONCURRENT_TASKS = 50
PER_DOMAIN_DELAY = 1.0
REQUEST_TIMEOUT = 20
MAX_RETRIES = 2

# ============ KEYWORD FILTER KHUSUS PARIWISATA ============

KEYWORD_FILTER = [
    # Umum pariwisata
    "wisata", "pariwisata", "liburan", "jalan-jalan",
    "travelling", "tour", "tourism", "travel", "trip",
    "destinasi", "destinasi wisata", "tempat wisata",
    "objek wisata", "obyek wisata", "spot wisata",

    # Jenis destinasi
    "pantai", "laut", "pulau", "snorkeling", "diving",
    "gunung", "pendakian", "hiking", "camping", "campground",
    "hutan", "taman nasional", "cagar alam",
    "desa wisata", "kampung wisata",
    "air terjun", "curug", "danau", "telaga",

    # Akomodasi & fasilitas
    "hotel", "resort", "villa", "guest house", "homestay",
    "penginapan", "hostel", "akomodasi", "airbnb",
    "booking", "reservasi", "check-in", "check in",

    # Transportasi & akses
    "tiket pesawat", "bandara", "airport",
    "kereta api", "stasiun", "bus wisata", "kapal", "pelabuhan",
    "akses menuju lokasi", "rute perjalanan", "itinerary",

    # Aktivitas & pengalaman
    "city tour", "walking tour", "open trip", "paket wisata",
    "paket liburan", "one day trip",
    "kuliner", "kulineran", "makanan khas", "jajanan lokal",
    "festival budaya", "acara budaya", "upacara adat",

    # Info praktis
    "harga tiket", "jam operasional", "jam buka",
    "tips liburan", "tips wisata", "panduan wisata",
    "rekomendasi wisata", "tempat hits", "instagramable",
]

# ============ DOMAIN YANG BOLEH ============

ALLOWED_DOMAINS = [
    # BENER-BENER CUMA TRAVEL
    "travel.detik.com",
    "travel.kompas.com",
]
