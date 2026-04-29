"""
Download pre-trained age & gender Caffe model weights.
Run this ONCE before running dl.py:
    python download_models.py
"""
import os
import urllib.request
import ssl

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Try multiple sources for reliability
SOURCES = [
    {
        "age": "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel",
        "gender": "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel",
    },
    {
        "age": "https://github.com/Isfhan/age-gender-detection/raw/main/age_net.caffemodel",
        "gender": "https://github.com/Isfhan/age-gender-detection/raw/main/gender_net.caffemodel",
    },
]

# Allow HTTPS even with self-signed certs on some networks
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def download_file(name, url, dest):
    print(f"  Downloading {name} from {url[:60]}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as resp:
            data = resp.read()
        if len(data) < 500_000:  # caffemodel should be >1 MB
            print(f"  [SKIP] File too small ({len(data)} bytes), probably not the real model.")
            return False
        with open(dest, "wb") as f:
            f.write(data)
        size_mb = len(data) / (1024 * 1024)
        print(f"  OK - {name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


FILES = {
    "age_net.caffemodel": "age",
    "gender_net.caffemodel": "gender",
}

for fname, key in FILES.items():
    dest = os.path.join(MODELS_DIR, fname)

    if os.path.exists(dest) and os.path.getsize(dest) > 500_000:
        print(f"  [EXISTS] {fname} already downloaded, skipping.")
        continue

    downloaded = False
    for source in SOURCES:
        if download_file(fname, source[key], dest):
            downloaded = True
            break

    if not downloaded:
        print(f"\n  Could not download {fname} automatically.")
        print(f"  Please download it manually from:")
        print(f"    https://github.com/smahesh29/Gender-and-Age-Detection")
        print(f"  and place it in the '{MODELS_DIR}/' folder.\n")

print("\nDone! Run:  python dl.py")
