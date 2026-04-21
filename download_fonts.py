"""
download_fonts.py
Downloads Playfair Display and Inter at Docker build time.
Falls back to apt-installed FreeFonts if download fails.
Runs as: python /app/download_fonts.py
"""
import os
import sys
import urllib.request
import urllib.error

PLAYFAIR_DIR = "/fonts/Playfair_Display"
INTER_DIR    = "/fonts/Inter"

os.makedirs(PLAYFAIR_DIR, exist_ok=True)
os.makedirs(INTER_DIR, exist_ok=True)

PLAYFAIR_OUT = f"{PLAYFAIR_DIR}/PlayfairDisplay-Regular.ttf"
INTER_OUT    = f"{INTER_DIR}/Inter-Regular.ttf"

# ── Google Fonts static CDN URLs ─────────────────────────────────────────────
# These are stable static URLs used by Google Fonts CSS API
# Fetched by requesting the CSS with a desktop User-Agent
FONTS = [
    {
        "name": "Playfair Display",
        "url": "https://fonts.gstatic.com/s/playfairdisplay/v37/nuFiD-vYSZviVYUb_rj3ij__anPXDTzYgEM86xRbAQ.ttf",
        "out": PLAYFAIR_OUT,
        "fallback": "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    },
    {
        "name": "Inter",
        "url": "https://fonts.gstatic.com/s/inter/v18/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfAZ9hiA.woff2",
        "out": INTER_OUT,
        "fallback": "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "is_woff2": True,
    },
]

def download(url, dest, name):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 1000:
            raise ValueError(f"Response too small ({len(data)} bytes) — likely an error page")
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  OK: {name} → {dest} ({len(data)//1024} KB)")
        return True
    except Exception as e:
        print(f"  FAIL: {name}: {e}")
        return False


def is_valid_font(path):
    """Check if file is a real font (TTF/OTF/WOFF2 magic bytes)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        # TTF: \x00\x01\x00\x00 or true / OTF: OTTO / WOFF2: wOF2
        return magic in (b'\x00\x01\x00\x00', b'true', b'OTTO', b'wOF2')
    except Exception:
        return False


for font in FONTS:
    print(f"\nDownloading {font['name']}...")
    
    # Try download
    ok = download(font["url"], font["out"], font["name"])
    
    # For woff2, we can't use it directly in PIL — PIL needs TTF/OTF
    # If Inter came as woff2, use fallback instead
    if ok and font.get("is_woff2"):
        if not is_valid_font(font["out"]):
            ok = False
        else:
            # Check magic bytes for woff2
            with open(font["out"], "rb") as f:
                magic = f.read(4)
            if magic == b'wOF2':
                print("  Downloaded as WOFF2 - PIL needs TTF, using fallback")
                ok = False

    if not ok or not is_valid_font(font["out"]):
        fallback = font["fallback"]
        if os.path.exists(fallback):
            import shutil
            shutil.copy(fallback, font["out"])
            print(f"  FALLBACK: copied {fallback} → {font['out']}")
        else:
            print(f"  ERROR: fallback {fallback} not found either!")
            sys.exit(1)

# Final verification
for font in FONTS:
    if is_valid_font(font["out"]):
        print(f"\n[OK] {font['name']}: {font['out']}")
    else:
        print(f"\n[FAIL] {font['name']}: {font['out']} is not a valid font!")
        sys.exit(1)

print("\nAll fonts ready.")
