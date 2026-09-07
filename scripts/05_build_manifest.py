"""
Turn the raw images/ folders into a labelled, normalised dataset.

Two things happen here:

1. NORMALISATION.  Every image is centre-cropped to a square, resized to
   512x512 and re-encoded as JPEG q95 into dataset/images/.  This is not
   cosmetic: the raw pool mixes 256px ABO thumbnails, 1024px StyleGAN PNGs and
   512px diffusion output.  A detector trained on that pool can reach high
   accuracy purely by reading resolution and compression history instead of
   synthesis artefacts.  Forcing one encoding for every class removes that
   shortcut.

2. LABELLING.  Each image gets its labels written *both* into the image's own
   EXIF (UserComment holds a JSON blob, ImageDescription a human-readable
   summary) and into dataset_manifest.csv, which is what training reads.

Labels per image
    level             L0..L5 sensitivity level
    label             real | fake
    is_fake           0 | 1
    severity_weight   cost multiplier for an error at this level
    source            which corpus/generator it came from
    split             train | val | test  (stratified on level x label)

    python scripts/05_build_manifest.py
"""
import csv
import hashlib
import json
import os
import random

from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "images")
OUT_DIR = os.path.join(PROJECT_DIR, "dataset")
OUT_IMAGES = os.path.join(OUT_DIR, "images")
MANIFEST = os.path.join(PROJECT_DIR, "dataset_manifest.csv")
SIZE = 512
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test

# Cost of getting an image wrong, by level.  L3 (identifiable people) and
# L4/L5 (politics, armed conflict) are where an undetected deepfake does real
# damage, so errors there are weighted several times an L0 texture error.
SEVERITY = {"L0": 0.5, "L1": 1.0, "L2": 1.5, "L3": 2.5, "L4": 3.5, "L5": 4.0}

LEVEL_DESC = {
    "L0": "Procedural / abstract",
    "L1": "Everyday",
    "L2": "Commercial",
    "L3": "Identifiable people / national",
    "L4": "International / political",
    "L5": "Military / armed conflict",
}

# filename prefix -> provenance, used to tag `source`
SOURCE_BY_PREFIX = {
    "dtd": "dtd", "coco": "coco2017", "ffhq": "ffhq", "abo": "amazon-berkeley-objects",
    "lfw": "lfw", "openfake_pol_real": "openfake-real", "openfake_pol_fake": "openfake-fake",
    "openfake_mil_real": "openfake-real", "openfake_mil_fake": "openfake-fake",
    "sd_abstract": "stable-diffusion-1.5", "sd_scene": "stable-diffusion-1.5",
    "sd_product": "stable-diffusion-1.5-img2img", "sd_portrait": "stable-diffusion-1.5",
    "sd_conflict": "stable-diffusion-1.5", "sd_political": "stable-diffusion-1.5",
    "random-person": "thispersondoesnotexist", "tpdne": "thispersondoesnotexist",
    "commons_pol": "wikimedia-commons", "commons_mil": "wikimedia-commons",
    "commons_person": "wikimedia-commons",
    "3d_procedural": "bing-image-creator", "abstract_image": "bing-image-creator",
    "noise_frattale": "bing-image-creator", "random_colour_splash": "bing-image-creator",
    "fake": "openfake-fake", "real": "openfake-real",
}

# Fallback for files that predate the naming convention (the first batches were
# downloaded by hand and keep their original opaque filenames).
SOURCE_BY_FOLDER = {
    "L0DeepFake": "bing-image-creator",
    "L1DeepFake": "thispersondoesnotexist",
    "L2Real": "amazon-berkeley-objects",
}

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def guess_source(fname, is_fake, folder=None):
    stem = os.path.splitext(fname)[0].lower()
    for prefix in sorted(SOURCE_BY_PREFIX, key=len, reverse=True):
        if stem.startswith(prefix):
            return SOURCE_BY_PREFIX[prefix]
    if folder in SOURCE_BY_FOLDER:
        return SOURCE_BY_FOLDER[folder]
    return "synthetic-unknown" if is_fake else "real-unknown"


def normalise(src_path, dest_path, meta):
    """Square centre-crop -> SIZE -> JPEG q95, with the labels in EXIF."""
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))
    img = img.resize((SIZE, SIZE), Image.LANCZOS)

    exif = img.getexif()
    exif[0x010E] = (f"{meta['level']} ({LEVEL_DESC[meta['level']]}) | {meta['label']} | "
                    f"source={meta['source']} | severity={meta['severity_weight']}")  # ImageDescription
    exif[0x0131] = "deepfake-sensitivity-dataset"                                      # Software
    exif[0x9286] = json.dumps(meta, separators=(",", ":"))                             # UserComment
    img.save(dest_path, "JPEG", quality=95, exif=exif)
    return (w, h)


def collect():
    """Walk images/L<n>{Real,DeepFake} and yield one record per raw image."""
    records = []
    for folder in sorted(os.listdir(RAW_DIR)):
        path = os.path.join(RAW_DIR, folder)
        if not os.path.isdir(path) or folder.startswith("_"):
            continue
        level = folder[:2].upper()
        if level not in SEVERITY:
            print(f"  ignoring folder {folder} (no L0-L5 prefix)")
            continue
        rest = folder[2:].lower()
        if rest.startswith("real"):
            is_fake = 0
        elif "fake" in rest:
            is_fake = 1
        else:
            print(f"  ignoring folder {folder} (cannot tell real from fake)")
            continue
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith(EXTS):
                records.append((level, is_fake, os.path.join(path, fname), fname, folder))
    return records


def main():
    os.makedirs(OUT_IMAGES, exist_ok=True)
    raw = collect()
    if not raw:
        raise SystemExit("no images found under images/")

    # Split is assigned per (level, label) group so every cell keeps its
    # 70/15/15 proportions -- with 50 images per cell an unstratified split
    # would easily leave a level absent from the test set.
    rng = random.Random(SEED)
    groups = {}
    for rec in raw:
        groups.setdefault((rec[0], rec[1]), []).append(rec)

    rows = []
    counters = {}
    seen_hashes = {}
    for (level, is_fake), items in sorted(groups.items()):
        items = sorted(items, key=lambda r: r[3])
        rng.shuffle(items)
        n = len(items)
        n_tr = int(round(n * SPLIT_RATIOS[0]))
        n_va = int(round(n * SPLIT_RATIOS[1]))
        splits = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
        for split, (lvl, fake, src_path, fname, folder) in zip(splits, items):
            # Byte-identical files slip in when a collector is restarted and
            # re-fetches rows it already had.  Keeping both would put the same
            # image in train and test.
            digest = hashlib.sha1(open(src_path, "rb").read()).hexdigest()[:16]
            if digest in seen_hashes:
                print(f"  duplicate of {seen_hashes[digest]}: {fname}")
                continue
            seen_hashes[digest] = fname
            label = "fake" if fake else "real"
            key = f"{lvl}_{label}"
            counters[key] = counters.get(key, 0) + 1
            image_id = f"{lvl}_{label}_{counters[key]:03d}"
            meta = {
                "image_id": image_id,
                "level": lvl,
                "level_name": LEVEL_DESC[lvl],
                "label": label,
                "is_fake": fake,
                "severity_weight": SEVERITY[lvl],
                "source": guess_source(fname, fake, folder),
                "original_file": fname,
            }
            dest = os.path.join(OUT_IMAGES, image_id + ".jpg")
            try:
                ow, oh = normalise(src_path, dest, meta)
            except Exception as exc:
                print(f"  SKIP {src_path}: {type(exc).__name__}: {exc}")
                counters[key] -= 1
                continue
            rows.append({
                "image_id": image_id,
                "file_path": os.path.relpath(dest, PROJECT_DIR).replace("\\", "/"),
                "level": lvl,
                "level_idx": int(lvl[1]),
                "level_name": LEVEL_DESC[lvl],
                "label": label,
                "is_fake": fake,
                "severity_weight": SEVERITY[lvl],
                "source": meta["source"],
                "split": split,
                "orig_width": ow,
                "orig_height": oh,
                "orig_path": os.path.relpath(src_path, PROJECT_DIR).replace("\\", "/"),
                "sha1": digest,
            })

    rows.sort(key=lambda r: r["image_id"])
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ── summary ──
    print(f"\n{len(rows)} images -> {os.path.relpath(OUT_IMAGES, PROJECT_DIR)}")
    print(f"manifest        -> {os.path.relpath(MANIFEST, PROJECT_DIR)}\n")
    print(f"{'level':<6}{'severity':>9}{'real':>7}{'fake':>7}{'train':>7}{'val':>6}{'test':>6}")
    for lvl in sorted(SEVERITY):
        sub = [r for r in rows if r["level"] == lvl]
        if not sub:
            continue
        print(f"{lvl:<6}{SEVERITY[lvl]:>9}"
              f"{sum(1 for r in sub if not r['is_fake']):>7}"
              f"{sum(1 for r in sub if r['is_fake']):>7}"
              f"{sum(1 for r in sub if r['split'] == 'train'):>7}"
              f"{sum(1 for r in sub if r['split'] == 'val'):>6}"
              f"{sum(1 for r in sub if r['split'] == 'test'):>6}")
    dupes = len(rows) - len({r["sha1"] for r in rows})
    print(f"\nduplicate source files: {dupes}")


if __name__ == "__main__":
    main()
