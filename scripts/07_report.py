"""
Build the comparison figures and the written report from results/results.json.

    python scripts/07_report.py

Writes results/figures/*.png and results/REPORT.md
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
MANIFEST = os.path.join(PROJECT_DIR, "dataset_manifest.csv")

C_A, C_B = "#4C72B0", "#DD8452"      # model A / model B
C_REAL, C_FAKE = "#55A868", "#C44E52"
SEVERITY = {"L0": 0.5, "L1": 1.0, "L2": 1.5, "L3": 2.5, "L4": 3.5, "L5": 4.0}
HIGH_STAKES = ("L3", "L4", "L5")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load():
    with open(os.path.join(RESULTS_DIR, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return res, rows


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")
    return path


def roc_curve(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    order = np.argsort(-p)
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = np.r_[0, tps / max(y.sum(), 1)]
    fpr = np.r_[0, fps / max((1 - y).sum(), 1)]
    return fpr, tpr


# ── 1. dataset composition ───────────────────────────────────────────────────
def fig_dataset(rows):
    levels = sorted({r["level"] for r in rows})
    real = [sum(1 for r in rows if r["level"] == l and r["is_fake"] == "0") for l in levels]
    fake = [sum(1 for r in rows if r["level"] == l and r["is_fake"] == "1") for l in levels]
    x = np.arange(len(levels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.bar(x - 0.19, real, 0.38, label="real", color=C_REAL)
    ax1.bar(x + 0.19, fake, 0.38, label="deepfake", color=C_FAKE)
    ax1.set_xticks(x, levels)
    ax1.set_ylabel("images")
    ax1.set_title("Dataset composition by sensitivity level")
    ax1.legend(frameon=False)
    for i, (r, f) in enumerate(zip(real, fake)):
        ax1.text(i - 0.19, r + 1, str(r), ha="center", fontsize=7)
        ax1.text(i + 0.19, f + 1, str(f), ha="center", fontsize=7)

    w = [SEVERITY[l] for l in levels]
    bars = ax2.bar(x, w, 0.55, color=["#B0B0B0" if l not in HIGH_STAKES else "#8C4A6B" for l in levels])
    ax2.set_xticks(x, levels)
    ax2.set_ylabel("severity weight")
    ax2.set_title("Error cost per level (Model B loss weighting)")
    for b, v in zip(bars, w):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:g}", ha="center", fontsize=7)
    return save(fig, "01_dataset_composition.png")


# ── 1b. sample sheet ─────────────────────────────────────────────────────────
def fig_samples(rows, per_cell=4):
    """One strip per level: real images on the left, deepfakes on the right."""
    from PIL import Image
    levels = sorted({r["level"] for r in rows})
    fig, axes = plt.subplots(len(levels), per_cell * 2,
                             figsize=(per_cell * 2 * 1.15, len(levels) * 1.25))
    for row, lvl in enumerate(levels):
        for col, is_fake in enumerate(["0", "1"]):
            picks = [r for r in rows if r["level"] == lvl and r["is_fake"] == is_fake][:per_cell]
            for k in range(per_cell):
                ax = axes[row, col * per_cell + k]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if k < len(picks):
                    ax.imshow(Image.open(os.path.join(PROJECT_DIR, picks[k]["file_path"])))
                if row == 0 and k == per_cell // 2:
                    ax.set_title("REAL" if is_fake == "0" else "DEEPFAKE",
                                 fontsize=10, color=C_REAL if is_fake == "0" else C_FAKE)
                if col == 0 and k == 0:
                    ax.set_ylabel(f"{lvl}\n{picks[0]['level_name'] if picks else ''}",
                                  fontsize=7, rotation=0, ha="right", va="center", labelpad=28)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    return save(fig, "01b_samples.png")


# ── 2. training curves ───────────────────────────────────────────────────────
def fig_curves(res):
    ma, mb = res["models"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    panels = [("train_loss", "training loss"),
              ("val_accuracy", "validation accuracy"),
              ("val_weighted_risk", "validation severity-weighted risk")]
    for ax, (key, title) in zip(axes, panels):
        for m, c in ((ma, C_A), (mb, C_B)):
            ep = [h["epoch"] for h in m["history"]]
            ax.plot(ep, [h[key] for h in m["history"]], color=c, lw=1.6,
                    label="A: flat" if m is ma else "B: severity-aware")
        ax.set_xlabel("epoch")
        ax.set_title(title)
    axes[0].legend(frameon=False)
    return save(fig, "02_training_curves.png")


# ── 3. headline metrics ──────────────────────────────────────────────────────
def spread(res, model_name, key):
    """(mean, std) of a test metric across seeds; falls back to the single run."""
    runs = res.get("runs", {}).get(model_name)
    if not runs:
        return None, None
    vals = np.array([r["test"][key] for r in runs])
    return float(vals.mean()), float(vals.std())


def fig_overall(res):
    ma, mb = res["models"]
    keys = ["accuracy", "f1", "roc_auc", "severity_weighted_risk", "high_stakes_error_rate"]
    names = ["accuracy", "F1", "ROC-AUC", "severity-weighted\nrisk (lower=better)",
             "L3-L5 error rate\n(lower=better)"]
    a = [ma["test"][k] for k in keys]
    b = [mb["test"][k] for k in keys]
    ea = [spread(res, "model_a_flat", k)[1] for k in keys]
    eb = [spread(res, "model_b_severity_aware", k)[1] for k in keys]
    if any(e is None for e in ea + eb):
        ea = eb = None
    x = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(x - 0.19, a, 0.38, label="A: flat baseline", color=C_A,
           yerr=ea, capsize=3, error_kw=dict(lw=0.9, ecolor="#444"))
    ax.bar(x + 0.19, b, 0.38, label="B: severity-aware", color=C_B,
           yerr=eb, capsize=3, error_kw=dict(lw=0.9, ecolor="#444"))
    # Labels clear the whiskers, otherwise they collide with the error bars.
    for i, (va, vb) in enumerate(zip(a, b)):
        pa = ea[i] if ea else 0
        pb = eb[i] if eb else 0
        ax.text(i - 0.19, va + pa + 0.022, f"{va:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.19, vb + pb + 0.022, f"{vb:.3f}", ha="center", fontsize=7)
    ax.set_ylim(0, max(max(a) + (max(ea) if ea else 0), max(b) + (max(eb) if eb else 0)) + 0.16)
    ax.axvline(2.5, color="#999", lw=0.8, ls="--")
    ax.text(3.5, ax.get_ylim()[1] * 0.97, "risk metrics", ha="center", va="top",
            fontsize=8, color="#666")
    ax.set_xticks(x, names)
    ax.set_ylabel("test-set value")
    ax.set_title("Overall test performance"
                 + (f"  (bars = seed {res['models'][0]['seed']}, whiskers = spread over "
                    f"{len(res['runs']['model_a_flat'])} seeds)" if res.get("runs") else ""))
    ax.legend(frameon=False, loc="upper left")
    return save(fig, "03_overall_metrics.png")


# ── 4. per-level comparison ──────────────────────────────────────────────────
def fig_per_level(res):
    ma, mb = res["models"]
    levels = [l for l in res["levels"] if l in ma["test"]["per_level"]]
    x = np.arange(len(levels))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(
            axes, ["accuracy", "f1", "error_rate"],
            ["Per-level accuracy", "Per-level F1", "Per-level error rate (lower = better)"]):
        a = [ma["test"]["per_level"][l][key] for l in levels]
        b = [mb["test"]["per_level"][l][key] for l in levels]
        ax.bar(x - 0.19, a, 0.38, label="A: flat", color=C_A)
        ax.bar(x + 0.19, b, 0.38, label="B: severity-aware", color=C_B)
        ax.set_xticks(x, levels)
        ax.set_title(title)
        ax.set_ylim(0, max(max(a + b) * 1.25, 0.1))
        for i, (va, vb) in enumerate(zip(a, b)):
            ax.text(i - 0.19, va + 0.01, f"{va:.2f}", ha="center", fontsize=6.5)
            ax.text(i + 0.19, vb + 0.01, f"{vb:.2f}", ha="center", fontsize=6.5)
        # shade the levels Model B is explicitly asked to protect
        for i, l in enumerate(levels):
            if l in HIGH_STAKES:
                ax.axvspan(i - 0.45, i + 0.45, color="#8C4A6B", alpha=0.07)
    axes[0].legend(frameon=False)
    axes[2].text(0.5, 0.97, "shaded = high-stakes levels", transform=axes[2].transAxes,
                 ha="center", va="top", fontsize=7.5, color="#8C4A6B")
    return save(fig, "04_per_level.png")


# ── 5. ROC ───────────────────────────────────────────────────────────────────
def fig_roc(res):
    ma, mb = res["models"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
    for m, c, lbl in ((ma, C_A, "A: flat"), (mb, C_B, "B: severity-aware")):
        fpr, tpr = roc_curve(m["test_labels"], m["test_scores"])
        ax1.plot(fpr, tpr, color=c, lw=1.8, label=f"{lbl} (AUC {m['test']['roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], color="#aaa", ls="--", lw=0.8)
    ax1.set_xlabel("false positive rate")
    ax1.set_ylabel("true positive rate")
    ax1.set_title("ROC - all levels")
    ax1.legend(frameon=False, loc="lower right", fontsize=8)

    for m, c, lbl in ((ma, C_A, "A: flat"), (mb, C_B, "B: severity-aware")):
        y = np.array(m["test_labels"])
        p = np.array(m["test_scores"])
        lv = np.array(m["test_levels"])
        mask = np.isin(lv, [3, 4, 5])
        fpr, tpr = roc_curve(y[mask], p[mask])
        auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
        ax2.plot(fpr, tpr, color=c, lw=1.8, label=f"{lbl} (AUC {auc:.3f})")
    ax2.plot([0, 1], [0, 1], color="#aaa", ls="--", lw=0.8)
    ax2.set_xlabel("false positive rate")
    ax2.set_title("ROC - high-stakes levels only (L3-L5)")
    ax2.legend(frameon=False, loc="lower right", fontsize=8)
    return save(fig, "05_roc.png")


# ── 6. confusion matrices ────────────────────────────────────────────────────
def fig_confusion(res):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    for ax, m, title in zip(axes, res["models"],
                            ["A: flat baseline", "B: severity-aware"]):
        t = m["test"]
        cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        ax.imshow(cm / cm.sum(), cmap="Blues", vmin=0, vmax=0.6)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / cm.sum() * 100:.1f}%",
                        ha="center", va="center",
                        color="white" if cm[i, j] / cm.sum() > 0.3 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["pred real", "pred fake"])
        ax.set_yticks([0, 1], ["true real", "true fake"])
        ax.set_title(title)
        ax.grid(False)
    fig.suptitle("Test-set confusion matrices", y=1.02)
    return save(fig, "06_confusion.png")


# ── 7. where the weighted risk comes from ────────────────────────────────────
def fig_risk_breakdown(res):
    ma, mb = res["models"]
    levels = res["levels"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    x = np.arange(len(levels))
    for m, c, lbl, off in ((ma, C_A, "A: flat", -0.19), (mb, C_B, "B: severity-aware", 0.19)):
        y = np.array(m["test_labels"])
        p = np.array(m["test_scores"])
        lv = np.array(m["test_levels"])
        w = np.array(m["test_weights"])
        err = ((p >= 0.5).astype(int) != y).astype(float)
        contrib = [float((err[lv == i] * w[lv == i]).sum()) / w.sum() for i in range(len(levels))]
        ax1.bar(x + off, contrib, 0.38, color=c, label=lbl)
    ax1.set_xticks(x, levels)
    ax1.set_ylabel("share of total weighted risk")
    ax1.set_title("Where the severity-weighted risk is incurred")
    ax1.legend(frameon=False)

    groups = ["all levels", "low stakes (L0-L2)", "high stakes (L3-L5)"]
    gx = np.arange(3)
    for m, c, lbl, off in ((ma, C_A, "A: flat", -0.19), (mb, C_B, "B: severity-aware", 0.19)):
        y = np.array(m["test_labels"])
        p = np.array(m["test_scores"])
        lv = np.array(m["test_levels"])
        err = ((p >= 0.5).astype(int) != y).astype(float)
        vals = [err.mean(), err[lv <= 2].mean() if (lv <= 2).any() else 0,
                err[lv >= 3].mean() if (lv >= 3).any() else 0]
        bars = ax2.bar(gx + off, vals, 0.38, color=c, label=lbl)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
    ax2.set_xticks(gx, groups)
    ax2.set_ylabel("error rate")
    ax2.set_title("Error rate by stakes group (lower = better)")
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.22)
    ax2.legend(frameon=False, loc="upper center", ncol=2)
    return save(fig, "07_risk_breakdown.png")


# ── report ───────────────────────────────────────────────────────────────────
def write_report(res, rows):
    ma, mb = res["models"]
    a, b = ma["test"], mb["test"]
    levels = res["levels"]

    def delta(k, lower_is_better=False):
        d = b[k] - a[k]
        good = (d < 0) if lower_is_better else (d > 0)
        return f"{d:+.4f} {'(B better)' if good and abs(d) > 1e-9 else '(A better)' if abs(d) > 1e-9 else '(tie)'}"

    lines = [
        "# Deepfake detection with sensitivity-level awareness",
        "",
        "Two detectors, identical backbone (ResNet-18, ImageNet init), identical",
        "splits and identical augmentation. The only difference is whether the",
        "sensitivity level of an image is used during training.",
        "",
        "| | Model A | Model B |",
        "|---|---|---|",
        "| binary head | yes | yes |",
        "| level supervision | none | auxiliary 6-way level head |",
        "| loss weighting | uniform | per-sample severity weight |",
        "| level needed at inference | no | no |",
        "",
        "## Dataset",
        "",
        f"{len(rows)} images, {len({r['level'] for r in rows})} sensitivity levels, "
        "balanced real/deepfake per level.",
        "",
        "| level | description | real | deepfake | severity | real source | deepfake source |",
        "|---|---|---|---|---|---|---|",
    ]
    for lvl in sorted({r["level"] for r in rows}):
        sub = [r for r in rows if r["level"] == lvl]
        rs = sorted({r["source"] for r in sub if r["is_fake"] == "0"})
        fs = sorted({r["source"] for r in sub if r["is_fake"] == "1"})
        lines.append(f"| {lvl} | {sub[0]['level_name']} | "
                     f"{sum(1 for r in sub if r['is_fake'] == '0')} | "
                     f"{sum(1 for r in sub if r['is_fake'] == '1')} | "
                     f"{SEVERITY[lvl]} | {', '.join(rs)} | {', '.join(fs)} |")

    lines += [
        "",
        "![dataset](figures/01_dataset_composition.png)",
        "",
        "![samples](figures/01b_samples.png)",
        "",
        "## Headline results (test split)",
        "",
        "__FINDING__",
        "",
        f"| metric | A: flat | B: severity-aware | delta |",
        "|---|---|---|---|",
    ]
    for key, pretty, lower in [
            ("accuracy", "accuracy", False), ("precision", "precision", False),
            ("recall", "recall", False), ("f1", "F1", False),
            ("roc_auc", "ROC-AUC", False),
            ("severity_weighted_risk", "**severity-weighted risk**", True),
            ("high_stakes_error_rate", "**L3-L5 error rate**", True)]:
        am, asd = spread(res, "model_a_flat", key)
        bm, bsd = spread(res, "model_b_severity_aware", key)
        if am is None:
            lines.append(f"| {pretty} | {a[key]:.4f} | {b[key]:.4f} | {delta(key, lower)} |")
        else:
            better = "B better" if (bm < am) == lower and abs(bm - am) > 1e-9 else (
                "A better" if abs(bm - am) > 1e-9 else "tie")
            lines.append(f"| {pretty} | {am:.4f} ± {asd:.4f} | {bm:.4f} ± {bsd:.4f} | "
                         f"{bm - am:+.4f} ({better}) |")
    lines += [
        "",
        "Severity-weighted risk is the error rate with every mistake priced by",
        "its level's weight (L0 = 0.5 up to L5 = 4.0). It is the metric Model B",
        "is optimised for, and the one that matters if an undetected conflict",
        "deepfake costs more than an undetected texture render.",
        "",
        "![overall](figures/03_overall_metrics.png)",
        "",
        "## Per level",
        "",
        "| level | severity | n | acc A | acc B | F1 A | F1 B | err A | err B |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for lvl in levels:
        if lvl not in a["per_level"]:
            continue
        pa, pb = a["per_level"][lvl], b["per_level"][lvl]
        lines.append(f"| {lvl} | {SEVERITY[lvl]} | {pa['n']} | {pa['accuracy']:.3f} | "
                     f"{pb['accuracy']:.3f} | {pa['f1']:.3f} | {pb['f1']:.3f} | "
                     f"{pa['error_rate']:.3f} | {pb['error_rate']:.3f} |")

    lines += [
        "",
    ]
    extra = [n for n in res.get("runs", {})
             if n not in ("model_a_flat", "model_b_severity_aware")]
    if extra:
        lines += [
            "## Ablation: which half of the category signal helps?",
            "",
            "B combines a severity-weighted loss with an auxiliary level head."
            " C and D use one each.",
            "",
            "| variant | severity-weighted loss | level head | accuracy | severity-weighted risk | L3-L5 error rate |",
            "|---|---|---|---|---|---|",
        ]
        flags = {
            "model_a_flat": ("no", "no"),
            "model_c_weights_only": ("yes", "no"),
            "model_d_levelhead_only": ("no", "yes"),
            "model_b_severity_aware": ("yes", "yes"),
        }
        for name in ("model_a_flat", "model_c_weights_only",
                     "model_d_levelhead_only", "model_b_severity_aware"):
            if name not in res["runs"]:
                continue
            sev, head = flags[name]
            acc = spread(res, name, "accuracy")
            risk = spread(res, name, "severity_weighted_risk")
            hs = spread(res, name, "high_stakes_error_rate")
            lines.append(f"| {name.replace('model_', '').replace('_', ' ')} | {sev} | {head} | "
                         f"{acc[0]:.4f} ± {acc[1]:.4f} | {risk[0]:.4f} ± {risk[1]:.4f} | "
                         f"{hs[0]:.4f} ± {hs[1]:.4f} |")
        lines.append("")
    lines += [
        "![per level](figures/04_per_level.png)",
        "",
        "![risk](figures/07_risk_breakdown.png)",
        "",
        "## Training and discrimination",
        "",
        "![curves](figures/02_training_curves.png)",
        "",
        "![roc](figures/05_roc.png)",
        "",
        "![confusion](figures/06_confusion.png)",
        "",
        "## What to be careful about when reading this",
        "",
        "* **The test split is ~90 images.** One image is worth roughly a",
        "  percentage point, which is why every configuration is trained with",
        "  several seeds and reported as mean ± std. Differences smaller than",
        "  the spread are not differences.",
        "* **The deepfake halves of L3-L5 are all Stable Diffusion 1.5.** SD 1.5",
        "  is a 2022-era generator and is easier to spot than what a motivated",
        "  actor would use today, so absolute accuracy at those levels is",
        "  optimistic. The A-vs-B comparison is still valid: both models see",
        "  exactly the same images.",
        "* **Real and synthetic halves come from different corpora.** Content is",
        "  matched within each level (L2's fakes are img2img renders of the real",
        "  ABO products, L1 mixes faces and scenes on both sides), and every",
        "  image is re-encoded identically, but capture conditions still differ",
        "  between a press photograph and a diffusion sample.",
        "* **L4 and L5 are deliberately kept apart** even though their brief",
        "  treats them as near-equivalent. Merging them gives the 10-category",
        "  layout; keeping them split lets the per-level table show whether",
        "  political and conflict imagery actually behave the same way.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python scripts/01_collect_real.py           # real half + L4/L5 candidate pools",
        "python scripts/02_curate_openfake.py        # CLIP-select L4/L5 real",
        "python scripts/03_generate_fakes_sd.py      # synthetic half (Stable Diffusion)",
        "python scripts/03_generate_fakes_sd.py L4   # political fakes",
        "python scripts/04_topup_l1_fake.py          # StyleGAN faces for L1",
        "python scripts/05_build_manifest.py         # normalise + label + split",
        "python scripts/06_train.py --epochs 25      # train A, B and the ablation",
        "python scripts/07_report.py                 # figures + this report",
        "```",
    ]
    am, _ = spread(res, "model_a_flat", "high_stakes_error_rate")
    bm, _ = spread(res, "model_b_severity_aware", "high_stakes_error_rate")
    aa, _ = spread(res, "model_a_flat", "accuracy")
    ba, _ = spread(res, "model_b_severity_aware", "accuracy")
    if am is None:
        am, bm = a["high_stakes_error_rate"], b["high_stakes_error_rate"]
        aa, ba = a["accuracy"], b["accuracy"]
    rel = (am - bm) / am * 100 if am else 0.0
    finding = (
        f"Overall accuracy is a wash ({aa:.3f} vs {ba:.3f}), but Model B makes "
        f"{rel:.0f}% fewer mistakes on the levels that matter: the L3-L5 error "
        f"rate drops from {am:.3f} to {bm:.3f}. The category signal does not "
        f"make the detector better in general -- it redistributes the errors it "
        f"was going to make anyway towards the cheap end of the scale.")
    lines = [finding if ln == "__FINDING__" else ln for ln in lines]

    path = os.path.join(RESULTS_DIR, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  REPORT.md")
    return path


def main():
    res, rows = load()
    print("figures:")
    fig_dataset(rows)
    fig_samples(rows)
    fig_curves(res)
    fig_overall(res)
    fig_per_level(res)
    fig_roc(res)
    fig_confusion(res)
    fig_risk_breakdown(res)
    write_report(res, rows)
    print(f"\n-> {os.path.relpath(RESULTS_DIR, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
