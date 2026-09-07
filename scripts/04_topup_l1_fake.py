"""
Top up images/L1DeepFake with StyleGAN2 faces from thispersondoesnotexist.com.

Each request to that endpoint returns a fresh, never-before-seen 1024x1024 face,
so it is the cheapest possible source of GAN-generated "ordinary person" images
to pair with the real FFHQ faces in L1Real.

    python scripts/04_topup_l1_fake.py [count]     # default 25
"""
import io
import os
import sys
import time

import requests
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_DIR, "images", "L1DeepFake")
URL = "https://thispersondoesnotexist.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset collection)"}
DELAY_S = 1.0  # the endpoint needs a moment to render a new face


def main(target):
    os.makedirs(OUT_DIR, exist_ok=True)
    have = len([f for f in os.listdir(OUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"L1DeepFake: {have} present, target {target}")
    seen = set()
    i = have
    fails = 0
    while i < target and fails < 20:
        try:
            raw = requests.get(URL, headers=HEADERS, timeout=45).content
            # The endpoint sometimes serves a cached frame; skip exact repeats.
            digest = hash(raw)
            if digest in seen:
                time.sleep(DELAY_S)
                continue
            seen.add(digest)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            i += 1
            name = f"tpdne_{i:03d}.jpg"
            img.save(os.path.join(OUT_DIR, name), "JPEG", quality=95)
            print(f"  [{i:03d}/{target}] {name}")
        except Exception as exc:
            fails += 1
            print(f"  retry ({type(exc).__name__}: {str(exc)[:60]})")
        time.sleep(DELAY_S)
    print(f"L1DeepFake: {len(os.listdir(OUT_DIR))} images on disk")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
