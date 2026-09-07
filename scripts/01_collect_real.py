"""
Collect the REAL (authentic) half of the dataset, one folder per sensitivity level.

    L0Real  Describable Textures Dataset (DTD)      -> abstract / texture
    L1Real  MS-COCO 2017 val                        -> everyday scenes
    L2Real  Amazon Berkeley Objects (ABO)           -> commercial products
    L3Real  Labeled Faces in the Wild (LFW)         -> identifiable people
    L4Real  OpenFake, political-keyword subset      -> national / international politics
    L5Real  OpenFake, military-keyword subset       -> armed conflict

Everything is pulled through the public HuggingFace datasets-server (no auth,
no multi-GB downloads) except ABO, which is served from its public S3 bucket.

    python scripts/01_collect_real.py            # all levels
    python scripts/01_collect_real.py L1 L3      # only some levels
"""
import io
import os
import sys
import time
import urllib.parse

import requests
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
TARGET_PER_CELL = 50
DELAY_S = 0.05
HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset collection)"}

ROWS_API = "https://datasets-server.huggingface.co/rows"
FILTER_API = "https://datasets-server.huggingface.co/filter"


# ── helpers ──────────────────────────────────────────────────────────────────
def out_dir(name):
    d = os.path.join(IMAGES_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def count(d):
    return len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def get_json(url, tries=6, wait=10):
    """datasets-server answers 500 'index is loading' on a cold dataset."""
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=180)
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:150]}"
        except Exception as exc:  # network hiccup
            last = str(exc)
        time.sleep(wait)
    raise RuntimeError(f"datasets-server failed after {tries} tries: {last}")


def save_image(src_url, dest, min_side=96, tries=4):
    """Download, verify it decodes, drop degenerate/too-small images, save as JPEG."""
    for attempt in range(tries):
        r = requests.get(src_url, headers=HEADERS, timeout=60)
        if r.status_code != 429:  # Commons throttles bursts; back off and retry
            break
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    img = img.convert("RGB")
    if min(img.size) < min_side:
        return False
    # A fully uniform image carries no signal (and is what a broken generation
    # looks like) -- reject it.
    if img.resize((32, 32)).getextrema() == ((0, 0), (0, 0), (0, 0)):
        return False
    img.save(dest, "JPEG", quality=95)
    return True


def fetch_rows(dataset, config, split, offset, length):
    url = (f"{ROWS_API}?dataset={urllib.parse.quote(dataset)}"
           f"&config={config}&split={split}&offset={offset}&length={length}")
    return get_json(url).get("rows", [])


def filter_rows(dataset, config, split, where, offset, length):
    url = (f"{FILTER_API}?dataset={urllib.parse.quote(dataset)}"
           f"&config={config}&split={split}&where={urllib.parse.quote(where)}"
           f"&offset={offset}&length={length}")
    return get_json(url).get("rows", [])


def harvest(folder, prefix, row_iter, target=TARGET_PER_CELL):
    """Pull rows from `row_iter` until `folder` holds `target` images."""
    d = out_dir(folder)
    have = count(d)
    if have >= target:
        print(f"  {folder}: already {have}/{target} - skipped")
        return
    print(f"  {folder}: {have}/{target}, downloading...")
    idx = have
    for src in row_iter:
        if idx >= target:
            break
        dest = os.path.join(d, f"{prefix}_{idx + 1:03d}.jpg")
        if os.path.exists(dest):
            idx += 1
            continue
        try:
            if save_image(src, dest):
                idx += 1
                print(f"    [{idx:03d}/{target}] {os.path.basename(dest)}")
            time.sleep(DELAY_S)
        except Exception as exc:
            print(f"    skip ({type(exc).__name__}: {str(exc)[:60]})")
    print(f"  {folder}: {count(d)}/{target} done")


def image_srcs(rows, field="image"):
    for row in rows:
        cell = row["row"].get(field)
        if isinstance(cell, dict) and cell.get("src"):
            yield cell["src"]


# ── L0: textures / abstract (DTD) ────────────────────────────────────────────
def collect_l0():
    def gen():
        # DTD train is 1880 rows across 47 texture classes; stride the offsets so
        # we do not end up with 50 near-identical images of the same class.
        for off in range(0, 1880, 37):
            for src in image_srcs(fetch_rows("tanganke/dtd", "default", "train", off, 2)):
                yield src

    harvest("L0Real", "dtd", gen())


# ── L1: everyday scenes + ordinary faces ─────────────────────────────────────
# The synthetic L1 half is 25 StyleGAN faces (thispersondoesnotexist) + 25
# generated scenes, so the real half mirrors that mix: 25 COCO scenes + 25 FFHQ
# faces.  Without this the classifier could separate the classes on "face vs
# scene" alone and never learn anything about synthesis artefacts.
def collect_l1():
    def coco():
        for off in range(0, 5000, 13):
            for src in image_srcs(fetch_rows("rafaelpadilla/coco2017", "default", "val", off, 2)):
                yield src

    def ffhq():
        for off in range(0, 60000, 211):
            for src in image_srcs(fetch_rows("bitmind/ffhq-256", "default", "train", off, 2)):
                yield src

    harvest("L1Real", "coco", coco(), target=25)
    harvest("L1Real", "ffhq", ffhq(), target=50)


# ── L2: commercial products (Amazon Berkeley Objects) ────────────────────────
def collect_l2():
    """ABO images are on public S3. The metadata CSV is ~60 MB gzipped, so it is
    only downloaded when the folder is actually short of images."""
    d = out_dir("L2Real")
    if count(d) >= TARGET_PER_CELL:
        print(f"  L2Real: already {count(d)}/{TARGET_PER_CELL} - skipped")
        return

    import csv
    import gzip
    import random

    meta_url = ("https://amazon-berkeley-objects.s3.amazonaws.com"
                "/images/metadata/images.csv.gz")
    base = "https://amazon-berkeley-objects.s3.amazonaws.com/images/small/"
    print("  L2Real: fetching ABO metadata (~60 MB)...")
    raw = requests.get(meta_url, headers=HEADERS, timeout=300).content
    with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as gz:
        rows = list(csv.DictReader(gz))
    random.Random(42).shuffle(rows)

    def gen():
        for row in rows:
            yield base + row["path"]

    harvest("L2Real", "abo", gen())


# ── L3: identifiable people (LFW + press photography) ────────────────────────
# LFW is 250x250, so after normalisation to 512 it is visibly soft while the
# synthetic L3 half is natively 512 and sharp.  A detector can separate those on
# blur alone.  Half of L3Real is therefore full-resolution press photography of
# public figures from Wikimedia Commons, which is also what the L3 brief allows
# ("LFW or targeted archives of press agencies").
L3_QUERIES = [
    "official portrait public figure photograph", "actor red carpet premiere photograph",
    "musician performing concert portrait", "author writer portrait photograph",
    "scientist researcher portrait photograph", "athlete press conference portrait",
    "business executive keynote portrait", "film director interview photograph",
    "television presenter studio photograph", "public figure speaking event portrait",
]


def collect_l3():
    def lfw():
        for off in range(0, 13000, 29):
            for src in image_srcs(fetch_rows("vilsonrodrigues/lfw", "default", "train", off, 2)):
                yield src

    harvest("L3Real", "lfw", lfw(), target=25)
    harvest("L3Real", "commons_person", commons_gen(L3_QUERIES), target=50)


# ── L4 / L5: OpenFake candidate pools ────────────────────────────────────────
# The datasets-server /filter endpoint takes minutes per page on a 2M-row
# dataset, and its SQL-ish LIKE matching is crude anyway ("tank" also matches
# "tank top", which is how fashion shots ended up tagged as conflict imagery).
# So: page the fast /rows endpoint in bulk, keyword-match captions client-side
# into a *candidate pool*, and let 02_curate_openfake.py do the real selection
# with CLIP.
CANDIDATE_DIR = os.path.join(IMAGES_DIR, "_candidates")
POOL_TARGET = 150

POLITICAL = [
    "president", "prime minister", "election", "parliament", "political rally",
    "campaign rally", "protest", "demonstration", "summit", "white house",
    "senator", "press conference", "politician", "government official",
    "state visit", "ballot", "voters", "chancellor", "diplomat", "embassy",
    "riot police", "political march", "head of state", "prime-minister",
]
MILITARY = [
    "soldier", "troops", "battlefield", "war zone", "war-torn", "wartime",
    "artillery", "airstrike", "air strike", "armored vehicle", "armoured vehicle",
    "battle tank", "machine gun", "missile launch", "military helicopter",
    "warship", "military parade", "armed conflict", "combat", "militants",
    "insurgents", "military base", "bombing", "air raid", "rubble", "air-raid",
    "refugee camp", "military uniform", "fighter jet", "army unit", "war ",
]


def openfake_rows(pages_per_split=40, page=100):
    """Stream (label, prompt, image_src) across the OpenFake splits."""
    for split in ("validation", "test"):
        for k in range(pages_per_split):
            try:
                rows = fetch_rows("ComplexDataLab/OpenFake", "core", split, k * page, page)
            except RuntimeError as exc:
                print(f"    OpenFake {split} page {k}: {str(exc)[:80]}")
                break
            if not rows:
                break
            for row in rows:
                rr = row["row"]
                cell = rr.get("image")
                if isinstance(cell, dict) and cell.get("src"):
                    yield rr.get("label"), str(rr.get("prompt") or ""), cell["src"]


def keyword_gen(want_label, keywords, exclude=()):
    kws = [w.lower() for w in keywords]
    exs = [w.lower() for w in exclude]
    for label, prompt, src in openfake_rows():
        if label != want_label:
            continue
        low = prompt.lower()
        if not any(w in low for w in kws):
            continue
        if exs and any(w in low for w in exs):
            continue
        yield src


# ── Wikimedia Commons ────────────────────────────────────────────────────────
# OpenFake's real half is LAION/ImageNet with free-text captions, which makes it
# a poor source of genuine conflict and political photography: keyword matching
# on those captions returns fashion shots ("tank top") and oil portraits
# ("military attire") as often as it returns war reporting.  Commons is searched
# by topic and returns actual press and public-domain photography, so it is the
# primary source for L4Real/L5Real and OpenFake supplements it.
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_HEADERS = {"User-Agent": "cv-research-dataset/1.0 (academic deepfake detection study)"}

MILITARY_QUERIES = [
    "soldiers combat operation photograph", "destroyed building shelling war",
    "artillery firing military exercise", "war zone civilians evacuation",
    "armoured vehicle convoy deployment", "military helicopter operation troops",
    "infantry patrol urban operation", "war damage city street rubble",
    "military parade troops marching", "warship naval operation deck",
    "refugees fleeing armed conflict", "field hospital wounded soldiers war",
    "trench position soldiers front line", "tank battlefield exercise",
    "air strike aftermath destroyed", "peacekeeping troops patrol",
    "military checkpoint soldiers road", "bomb damage building war photograph",
    "soldiers training live fire exercise", "military transport aircraft troops",
]
POLITICAL_QUERIES = [
    "president speech podium official", "prime minister press conference",
    "political rally crowd banners", "parliament session chamber debate",
    "street protest demonstration placards", "election polling station voters",
    "diplomatic meeting heads of state handshake", "state visit ceremony flags",
    "government minister official portrait event", "united nations assembly speech",
    "campaign rally supporters signs", "riot police demonstration street",
    "summit leaders group photograph", "press briefing government spokesperson",
    "ballot counting election officials", "inauguration ceremony official",
    "march protest banners crowd city", "embassy building officials",
    "cabinet meeting officials table", "signing ceremony agreement officials",
]


def commons_gen(queries, per_query=40):
    """Full-resolution-ish thumbnails for Commons search hits, query by query."""
    for q in queries:
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": q, "gsrnamespace": "6", "gsrlimit": per_query,
            "prop": "imageinfo", "iiprop": "url|size|mime", "iiurlwidth": "768",
        }
        try:
            r = requests.get(COMMONS_API, params=params, headers=COMMONS_HEADERS, timeout=60)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
        except Exception as exc:
            print(f"    commons '{q[:30]}': {type(exc).__name__}")
            continue
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            if not info.get("mime", "").startswith("image/"):
                continue
            url = info.get("thumburl") or info.get("url")
            if url:
                yield url
        time.sleep(0.5)


def collect_l4():
    harvest(os.path.join("_candidates", "L4Real"), "commons_pol",
            commons_gen(POLITICAL_QUERIES), target=POOL_TARGET)
    harvest(os.path.join("_candidates", "L4Real"), "openfake_pol_real",
            keyword_gen("real", POLITICAL, exclude=MILITARY), target=POOL_TARGET)
    harvest(os.path.join("_candidates", "L4DeepFake"), "openfake_pol_fake",
            keyword_gen("fake", POLITICAL, exclude=MILITARY), target=POOL_TARGET)


def collect_l5():
    harvest(os.path.join("_candidates", "L5Real"), "commons_mil",
            commons_gen(MILITARY_QUERIES), target=POOL_TARGET)
    harvest(os.path.join("_candidates", "L5Real"), "openfake_mil_real",
            keyword_gen("real", MILITARY), target=POOL_TARGET)
    # L5DeepFake comes from 03_generate_fakes_sd.py; OpenFake military fakes are
    # kept as a candidate pool in case that set needs topping up.
    harvest(os.path.join("_candidates", "L5DeepFake"), "openfake_mil_fake",
            keyword_gen("fake", MILITARY), target=POOL_TARGET)


LEVELS = {"L0": collect_l0, "L1": collect_l1, "L2": collect_l2,
          "L3": collect_l3, "L4": collect_l4, "L5": collect_l5}

if __name__ == "__main__":
    wanted = [a.upper() for a in sys.argv[1:]] or list(LEVELS)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    for lvl in wanted:
        print(f"\n=== {lvl} ===")
        LEVELS[lvl]()
    print("\nDone.")
