"""
Pick the best 50 images per L4/L5 cell using CLIP zero-shot scoring.

Why this step exists: OpenFake has no topic labels, only free-text captions, so
L4/L5 candidates are gathered by keyword. Keyword matching on captions is
unreliable in both directions -- "tank" matches "tank top", "military attire"
matches an oil portrait, and plenty of genuine conflict photography is captioned
without any of the words we search for. Reviewing every candidate by hand does
not scale, so CLIP does it: each candidate is scored against a set of
level-appropriate descriptions versus a set of off-topic ones, and only the top
`TARGET` survive into the training set; the rest stay in images/_candidates.

    python scripts/02_curate_openfake.py                 # all OpenFake cells
    python scripts/02_curate_openfake.py L5Real          # just one
    python scripts/02_curate_openfake.py --dry-run       # score, change nothing
"""
import argparse
import os
import shutil

import torch
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
CAND_DIR = os.path.join(IMAGES_DIR, "_candidates")
CLIP_ID = "openai/clip-vit-base-patch32"
TARGET = 50
BATCH = 32
DEDUP_SIM = 0.93   # CLIP cosine above which two candidates count as the same shot

OFF_TOPIC = [
    "a fashion photograph of a model posing",
    "a studio product photograph on a white background",
    "a landscape or nature photograph",
    "an abstract artwork or painting",
    "a photograph of food on a plate",
    "a book cover or magazine page",
    "a screenshot of a page of text",
    "a cartoon or anime illustration",
    "a close-up portrait with a plain background",
    "a photograph of a sports event",
]

CELLS = {
    "L4Real": [
        "a news photograph of a politician giving a speech at a podium",
        "a political rally with a large crowd and banners",
        "a press conference with microphones and journalists",
        "a street protest or demonstration with placards",
        "government officials meeting around a table",
        "a parliament or legislative chamber in session",
        "people voting at a polling station",
        "heads of state shaking hands in front of national flags",
    ],
    "L4DeepFake": None,   # same descriptions as L4Real
    "L5Real": [
        "a photograph of soldiers in a war zone",
        "a destroyed city street after shelling, rubble and smoke",
        "military vehicles and tanks in a combat area",
        "war photojournalism from an armed conflict",
        "troops in uniform carrying weapons in the field",
        "an artillery piece or missile launcher being fired",
        "a bombed building with smoke rising",
        "refugees fleeing a conflict zone",
    ],
    "L5DeepFake": None,   # same descriptions as L5Real
}
CELLS["L4DeepFake"] = CELLS["L4Real"]
CELLS["L5DeepFake"] = CELLS["L5Real"]

EXTS = (".jpg", ".jpeg", ".png", ".webp")


def listdir(path):
    if not os.path.isdir(path):
        return []
    return sorted(f for f in os.listdir(path) if f.lower().endswith(EXTS))


def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {CLIP_ID} on {device}...")
    model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    return model, proc, device


def as_tensor(out):
    """transformers >=5 returns a model output object from get_*_features;
    older versions return the tensor directly."""
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "last_hidden_state", "text_embeds", "image_embeds"):
        val = getattr(out, attr, None)
        if torch.is_tensor(val):
            return val
    return out[0]


@torch.no_grad()
def score(model, proc, device, paths, positives):
    """Returns (on-topic score, embedding) per image.

    The score is the softmax mass CLIP puts on the level's own descriptions
    versus the off-topic ones; the embeddings are reused for de-duplication."""
    prompts = positives + OFF_TOPIC
    text = proc(text=prompts, return_tensors="pt", padding=True).to(device)
    tfeat = as_tensor(model.get_text_features(**text))
    tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    scores, embeds = [], []
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        px = proc(images=imgs, return_tensors="pt").to(device)
        ifeat = as_tensor(model.get_image_features(**px))
        ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
        probs = (model.logit_scale.exp() * ifeat @ tfeat.T).softmax(dim=-1)
        scores.extend(probs[:, :len(positives)].sum(dim=-1).cpu().tolist())
        embeds.append(ifeat.cpu())
    return scores, torch.cat(embeds)


def curate(cell, model, proc, device, dry_run=False, target=TARGET):
    final = os.path.join(IMAGES_DIR, cell)
    cands = os.path.join(CAND_DIR, cell)
    os.makedirs(final, exist_ok=True)
    os.makedirs(cands, exist_ok=True)

    # Everything currently in the cell folder goes back into the candidate pool
    # first, so selection always considers the full set rather than whatever
    # happened to be downloaded first, and so re-running is idempotent.
    # Filenames are preserved throughout, which is what the manifest reads
    # provenance from.
    for f in listdir(final):
        dest = os.path.join(cands, f)
        if os.path.exists(dest):
            os.remove(os.path.join(final, f))
        else:
            shutil.move(os.path.join(final, f), dest)

    paths = [os.path.join(cands, f) for f in listdir(cands)]
    if not paths:
        print(f"  {cell}: no candidates")
        return
    scores, embeds = score(model, proc, device, paths, CELLS[cell])
    order = sorted(range(len(paths)), key=lambda i: -scores[i])

    # Commons search returns bursts of near-identical frames from the same
    # event; keeping several would inflate the effective duplicate rate and leak
    # between train and test.  Reject anything too close to an image already
    # kept, walking the list from best score down.
    keep = []
    for i in order:
        if len(keep) >= target:
            break
        if keep:
            sim = float((embeds[keep] @ embeds[i]).max())
            if sim > DEDUP_SIM:
                continue
        keep.append(i)

    lo, hi = scores[keep[-1]], scores[keep[0]]
    print(f"  {cell}: {len(paths)} candidates -> keeping {len(keep)} "
          f"(on-topic {lo:.3f}..{hi:.3f}, dedup>{DEDUP_SIM})")
    if dry_run:
        for i in keep[:5]:
            print(f"      keep {scores[i]:.3f}  {os.path.basename(paths[i])}")
        for i in order[-3:]:
            print(f"      drop {scores[i]:.3f}  {os.path.basename(paths[i])}")
        return

    for i in keep:
        shutil.copy2(paths[i], os.path.join(final, os.path.basename(paths[i])))
    if len(keep) < target:
        print(f"    WARNING: only {len(keep)}/{target} images available for {cell}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args()

    # The deepfake halves of L4/L5 are generated locally and are on-topic by
    # construction, so only the real halves -- pooled from Commons search and
    # OpenFake captions -- need selecting.
    cells = args.cells or ["L4Real", "L5Real"]
    model, proc, device = load_clip()
    for cell in cells:
        curate(cell, model, proc, device, args.dry_run, args.target)
    print("\nDone.")


if __name__ == "__main__":
    main()
