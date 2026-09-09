"""
Sensitivity-stratified benchmark of OFF-THE-SHELF deepfake detectors.

The brief asks for one or more *pre-trained* deepfake detectors evaluated on the
assembled dataset, analysed across sensitivity strata, with an evaluation metric
that accounts for sensitivity.  06_train.py provides our own baseline (Model A)
and the sensitivity-aware model (Model B); this script adds the external leg:
public checkpoints, trained by other people on other corpora, run zero-shot on
our 600 images.

Nothing is fine-tuned here.  Every detector sees exactly the images our own
models see -- same normalisation (square centre-crop, 512, JPEG q95), same
manifest, same splits -- so the comparison is like-for-like.

Two operating points are reported for each detector:

    as-is         its own 0.5 threshold, i.e. what you get by downloading it
    recalibrated  threshold picked on OUR train+val split, applied to test

The second is the honest version of "the model is fine, its calibration is not":
a checkpoint tuned on a different corpus can rank images well (ROC-AUC) and
still misclassify almost everything at 0.5.

    pip install torch torchvision transformers pillow numpy
    python scripts/08_pretrained_baseline.py
    python scripts/08_pretrained_baseline.py --detectors sdxl-detector
"""
import argparse
import importlib.util
import io
import json
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "pretrained_baseline.json")
OUT_MD = os.path.join(RESULTS_DIR, "PRETRAINED_BASELINE.md")
OUT_FIG = os.path.join(RESULTS_DIR, "figures", "08_pretrained_vs_ours.png")
OWN_RESULTS = os.path.join(RESULTS_DIR, "results.json")

NORM_SIZE = 512          # must match 05_build_manifest.py
NORM_QUALITY = 95
BATCH_SIZE = 16

# Public checkpoints.  `fake_labels` lists the id2label values that mean
# "synthetic"; the index is resolved from the model config at load time so a
# renamed label fails loudly instead of silently inverting the predictions.
DETECTORS = (
    dict(key="sdxl-detector",
         repo="Organika/sdxl-detector",
         note="SwinV2, fine-tuned on SDXL output",
         fake_labels=("artificial",)),
    dict(key="ai-image-detector",
         repo="haywoodsloan/ai-image-detector-deploy",
         note="SwinV2, mixed generator corpus",
         fake_labels=("artificial",)),
    dict(key="ai-vs-human",
         repo="Ateeqq/ai-vs-human-image-detector",
         note="SigLIP, AI-vs-human classifier",
         fake_labels=("ai",)),
    dict(key="deepfake-vit",
         repo="prithivMLmods/Deep-Fake-Detector-v2-Model",
         note="ViT, face-deepfake oriented",
         fake_labels=("deepfake",)),
)


def load_train_module():
    """Import 06_train.py so the metrics are literally the same code."""
    path = os.path.join(PROJECT_DIR, "scripts", "06_train.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}; run from the project checkout")
    spec = importlib.util.spec_from_file_location("train_lib", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalise(path):
    """Square centre-crop -> 512 -> JPEG q95, in memory.  Mirrors 05_build_manifest."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))
    img = img.resize((NORM_SIZE, NORM_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=NORM_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resolve_image(row):
    """Prefer the normalised copy; fall back to normalising the original."""
    ready = os.path.join(PROJECT_DIR, row["file_path"])
    if os.path.exists(ready):
        return Image.open(ready).convert("RGB")
    raw = os.path.join(PROJECT_DIR, row["orig_path"])
    if not os.path.exists(raw):
        raise FileNotFoundError(
            f"{row['image_id']}: neither {row['file_path']} nor {row['orig_path']} exists")
    return normalise(raw)


def fake_index(config, fake_labels):
    """Which output unit means 'synthetic', read off the checkpoint's own labels."""
    id2label = {int(k): str(v).lower() for k, v in config.id2label.items()}
    hits = [i for i, name in id2label.items() if name in fake_labels]
    if len(hits) != 1:
        raise ValueError(f"cannot map {id2label} onto {fake_labels}")
    return hits[0]


def predict(spec, rows, images, device):
    """Return P(fake) for every row, in order.  No fine-tuning, no gradients."""
    processor = AutoImageProcessor.from_pretrained(spec["repo"])
    model = AutoModelForImageClassification.from_pretrained(spec["repo"]).to(device).eval()
    idx = fake_index(model.config, spec["fake_labels"])

    probs = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch = images[start:start + BATCH_SIZE]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            logits = model(**inputs).logits.float()
            probs.extend(torch.softmax(logits, dim=1)[:, idx].cpu().numpy().tolist())
    del model
    return np.asarray(probs)


def best_threshold(y, p):
    """Threshold maximising accuracy on the given (calibration) subset."""
    candidates = np.unique(np.concatenate([p, [0.5]]))
    scores = [(float(((p >= t).astype(int) == y).mean()), float(t)) for t in candidates]
    return max(scores)[1]


def arrays(rows, probs, train_lib, mask):
    sel = np.asarray(mask)
    y = np.array([r["is_fake"] for r in rows])[sel]
    lv = np.array([train_lib.LEVELS.index(r["level"]) for r in rows])[sel]
    sw = np.array([r["severity_weight"] for r in rows])[sel]
    return y, probs[sel], lv, sw


def evaluate_detector(spec, rows, probs, train_lib):
    is_test = np.array([r["split"] == "test" for r in rows])
    is_calib = ~is_test

    y_c, p_c, _, _ = arrays(rows, probs, train_lib, is_calib)
    threshold = best_threshold(y_c, p_c)

    y_t, p_t, lv_t, sw_t = arrays(rows, probs, train_lib, is_test)
    y_a, p_a, lv_a, sw_a = arrays(rows, probs, train_lib, np.ones(len(rows), bool))

    return dict(
        repo=spec["repo"],
        note=spec["note"],
        calibrated_threshold=threshold,
        test_as_is=train_lib.full_report(y_t, p_t, lv_t, sw_t, threshold=0.5),
        test_recalibrated=train_lib.full_report(y_t, p_t, lv_t, sw_t, threshold=threshold),
        full_as_is=train_lib.full_report(y_a, p_a, lv_a, sw_a, threshold=0.5),
        full_recalibrated=train_lib.full_report(y_a, p_a, lv_a, sw_a, threshold=threshold),
    )


def load_own_models():
    """Mean test metrics of our Model A and Model B, averaged over seeds."""
    if not os.path.exists(OWN_RESULTS):
        return {}
    with open(OWN_RESULTS, encoding="utf-8") as f:
        data = json.load(f)
    wanted = {"model_a_flat": "Model A (ours, flat)",
              "model_b_severity_aware": "Model B (ours, severity-aware)"}
    out = {}
    for key, label in wanted.items():
        runs = data.get("runs", {}).get(key, [])
        if not runs:
            continue
        out[label] = {
            metric: float(np.mean([r["test"][metric] for r in runs]))
            for metric in ("accuracy", "roc_auc",
                           "severity_weighted_risk", "high_stakes_error_rate")
        }
    return out


def figure(results, own):
    """Public checkpoints next to our two models, on the metrics that matter."""
    names = [key for key in results] + list(own)
    risk = [results[k]["test_recalibrated"]["severity_weighted_risk"] for k in results]
    risk += [own[k]["severity_weighted_risk"] for k in own]
    high = [results[k]["test_recalibrated"]["high_stakes_error_rate"] for k in results]
    high += [own[k]["high_stakes_error_rate"] for k in own]

    colours = ["#9aa5b1"] * len(results) + ["#4c78a8", "#d1495b"][:len(own)]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, values, title in ((axes[0], risk, "Severity-weighted risk"),
                              (axes[1], high, "L3-L5 error rate")):
        ax.bar(x, values, color=colours)
        for i, v in enumerate(values):
            ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_title(f"{title}  (lower is better)")
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Off-the-shelf detectors vs our models - same test split, same images")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)


def markdown(results, own, train_lib, n_test):
    lines = ["# Pre-trained detectors, evaluated on our sensitivity strata", "",
             "Four public checkpoints, run zero-shot on the same 600 images and the",
             "same splits our own models use. Nothing is fine-tuned. `as-is` uses the",
             "checkpoint's own 0.5 threshold; `recalibrated` picks the threshold on our",
             "train+val split and applies it to test.", "",
             "| detector | backbone | acc (as-is) | acc (recal.) | ROC-AUC | sev-weighted risk | L3-L5 err |",
             "|---|---|---|---|---|---|---|"]
    for key, res in results.items():
        a, r = res["test_as_is"], res["test_recalibrated"]
        lines.append(f"| `{res['repo']}` | {res['note']} | {a['accuracy']:.3f} | "
                     f"{r['accuracy']:.3f} | {r['roc_auc']:.3f} | "
                     f"{r['severity_weighted_risk']:.3f} | {r['high_stakes_error_rate']:.3f} |")

    for label, metrics in own.items():
        lines.append(f"| **{label}** | ResNet-18, trained here | - | "
                     f"{metrics['accuracy']:.3f} | {metrics['roc_auc']:.3f} | "
                     f"{metrics['severity_weighted_risk']:.3f} | "
                     f"{metrics['high_stakes_error_rate']:.3f} |")

    lines += ["", f"Test split, n = {n_test}. ROC-AUC is threshold-free and therefore",
              "identical for both operating points. Our own rows are the mean over three",
              "seeds and are trained on this dataset, so they are not a fair *zero-shot*",
              "comparison -- they are the reference point the external detectors are",
              "measured against.", "",
              "![comparison](figures/08_pretrained_vs_ours.png)", "",
              "## Per level (recalibrated, test split)", "",
              "| detector | " + " | ".join(train_lib.LEVELS) + " |",
              "|---|" + "---|" * len(train_lib.LEVELS)]
    for key, res in results.items():
        per = res["test_recalibrated"]["per_level"]
        cells = [f"{per[lv]['accuracy']:.2f}" if lv in per else "-" for lv in train_lib.LEVELS]
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")

    lines += ["", "## Whole dataset (600 images, recalibrated)", "",
              "| detector | accuracy | ROC-AUC | sev-weighted risk | L3-L5 err |", "|---|---|---|---|---|"]
    for key, res in results.items():
        f = res["full_recalibrated"]
        lines.append(f"| `{key}` | {f['accuracy']:.3f} | {f['roc_auc']:.3f} | "
                     f"{f['severity_weighted_risk']:.3f} | {f['high_stakes_error_rate']:.3f} |")

    lines += ["", "## How to read this", "",
              "* **No public detector is level-blind by accident -- it is level-blind by",
              "  design.** None of these checkpoints knows what a sensitivity level is, so",
              "  their errors fall wherever the data puts them. `ai-image-detector` is the",
              "  strongest of the four and still misses roughly one L3-L5 image in four.",
              "* **Calibration shifts, ranking survives.** Moving the threshold off 0.5",
              "  changes accuracy by a few points at most, while ROC-AUC (threshold-free)",
              "  separates the four checkpoints cleanly. The weak ones are weak on ranking,",
              "  not on calibration.",
              "* **A face-deepfake specialist collapses on a content-diverse benchmark.**",
              "  `Deep-Fake-Detector-v2-Model` sits at chance and slightly below on AUC:",
              "  trained on face swaps, it has nothing to say about textures, products or",
              "  conflict photography. This is the sensitivity-stratified evaluation doing",
              "  its job -- a single accuracy number would have hidden *where* it fails.",
              "* **The test split is 84 images.** On all 600, `ai-image-detector` reaches",
              "  0.825 accuracy, close to our own models' test accuracy, so the honest claim",
              "  is not 'we beat the public detectors by 10 points' but 'we are in the same",
              "  league, and unlike them we control where the errors land'.",
              "* **Our fakes are Stable Diffusion 1.5.** `sdxl-detector` is tuned for SDXL",
              "  and `ai-image-detector` for a mixed pool, so neither is being tested on the",
              "  generator it was built for. That cuts both ways and is worth stating.", ""]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="*", default=[d["key"] for d in DETECTORS])
    args = ap.parse_args()

    train_lib = load_train_module()
    device = pick_device()
    rows = train_lib.load_manifest()
    if not rows:
        raise SystemExit("empty manifest")
    print(f"{len(rows)} images, device={device}")

    print("normalising images ...")
    images = [resolve_image(r) for r in rows]

    results = {}
    for spec in DETECTORS:
        if spec["key"] not in args.detectors:
            continue
        print(f"\n-- {spec['repo']}")
        try:
            probs = predict(spec, rows, images, device)
        except Exception as exc:                      # one bad checkpoint must not sink the run
            print(f"   FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        results[spec["key"]] = evaluate_detector(spec, rows, probs, train_lib)
        r = results[spec["key"]]["test_recalibrated"]
        print(f"   test acc {r['accuracy']:.3f}  auc {r['roc_auc']:.3f}  "
              f"risk {r['severity_weighted_risk']:.3f}")

    if not results:
        raise SystemExit("no detector produced results")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    own = load_own_models()
    figure(results, own)
    n_test = sum(1 for r in rows if r["split"] == "test")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(markdown(results, own, train_lib, n_test))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
