"""
Train and compare two deepfake detectors on the same data and the same splits.

  MODEL A -- "flat" baseline
      ResNet-18 (ImageNet weights) -> 1 logit.  Plain BCE, every image counts
      the same.  It never sees the sensitivity level.

  MODEL B -- "severity-aware"
      Same backbone and same splits, but the category is used in two ways:
        1. the binary loss of each sample is scaled by its level's severity
           weight, so an L5 (armed conflict) mistake costs 8x an L0 mistake;
        2. an auxiliary 6-way head predicts the level itself, forcing the
           shared trunk to keep category information instead of collapsing to
           whatever separates real from fake on average.
      The level is a *training-time* signal only -- at inference Model B takes
      an image and nothing else, exactly like Model A, so the comparison is
      fair and the model stays deployable.

  MODEL C / MODEL D -- ablation
      B's two ingredients on their own: C uses the severity-weighted loss with
      no level head, D uses the level head with a uniform loss.  They answer
      "which half of the category signal is doing the work".

Reported for each: accuracy / precision / recall / F1 / ROC-AUC overall and per
level, plus severity-weighted risk -- the error rate with each mistake weighted
by its level's cost, which is the metric the whole exercise is about.  Every
config is trained with `--runs` different seeds, because a ~90-image test split
moves by a couple of points between seeds on its own.

    python scripts/06_train.py                 # A, B and the ablation, 3 seeds
    python scripts/06_train.py --epochs 25 --no-ablation --runs 1
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(PROJECT_DIR, "dataset_manifest.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
SEED = 42
LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── data ─────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.15, 0.15, 0.15, 0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class DeepfakeDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows = rows
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(PROJECT_DIR, r["file_path"])).convert("RGB")
        return (self.tf(img),
                torch.tensor(float(r["is_fake"])),
                torch.tensor(LEVELS.index(r["level"])),
                torch.tensor(float(r["severity_weight"])))


def load_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["is_fake"] = int(r["is_fake"])
        r["severity_weight"] = float(r["severity_weight"])
    return rows


# ── models ───────────────────────────────────────────────────────────────────
class Detector(nn.Module):
    """ResNet-18 trunk + binary head, plus an optional 6-way level head."""

    def __init__(self, with_level_head=False):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.trunk = net
        self.dropout = nn.Dropout(0.3)
        self.binary_head = nn.Linear(feat_dim, 1)
        self.level_head = nn.Linear(feat_dim, len(LEVELS)) if with_level_head else None

    def forward(self, x):
        h = self.dropout(self.trunk(x))
        logit = self.binary_head(h).squeeze(1)
        level_logits = self.level_head(h) if self.level_head is not None else None
        return logit, level_logits


# ── metrics ──────────────────────────────────────────────────────────────────
def roc_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks over ties, otherwise AUC is biased when scores saturate
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def prf(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1,
                accuracy=(tp + tn) / max(len(y), 1))


def evaluate(model, loader):
    model.eval()
    ys, ps, lv, sw = [], [], [], []
    with torch.no_grad():
        for x, y, level, w in loader:
            logit, _ = model(x.to(device, non_blocking=True))
            ps.extend(torch.sigmoid(logit).cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
            lv.extend(level.numpy().tolist())
            sw.extend(w.numpy().tolist())
    return np.array(ys), np.array(ps), np.array(lv), np.array(sw)


def full_report(y, p, lv, sw, threshold=0.5):
    yhat = (p >= threshold).astype(int)
    rep = prf(y, yhat)
    rep["roc_auc"] = roc_auc(y, p)
    errs = (yhat != y).astype(float)
    # The headline metric: error rate with every mistake priced by its level.
    rep["severity_weighted_risk"] = float((errs * sw).sum() / sw.sum())
    rep["high_stakes_error_rate"] = float(
        errs[np.isin(lv, [3, 4, 5])].mean()) if np.isin(lv, [3, 4, 5]).any() else float("nan")
    rep["per_level"] = {}
    for i, name in enumerate(LEVELS):
        m = lv == i
        if not m.any():
            continue
        sub = prf(y[m], yhat[m])
        sub["roc_auc"] = roc_auc(y[m], p[m])
        sub["n"] = int(m.sum())
        sub["error_rate"] = float(errs[m].mean())
        rep["per_level"][name] = sub
    return rep


# ── training ─────────────────────────────────────────────────────────────────
def train_one(name, with_level_head, use_severity, rows, args, seed=SEED):
    set_seed(seed)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    te = [r for r in rows if r["split"] == "test"]

    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(DeepfakeDataset(tr, train_tf), batch_size=args.batch_size,
                       shuffle=True, num_workers=args.workers, generator=g, drop_last=False)
    dl_va = DataLoader(DeepfakeDataset(va, eval_tf), batch_size=args.batch_size,
                       num_workers=args.workers)
    dl_te = DataLoader(DeepfakeDataset(te, eval_tf), batch_size=args.batch_size,
                       num_workers=args.workers)

    model = Detector(with_level_head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"\n{'=' * 62}\n{name}\n"
          f"  level head: {with_level_head}   severity-weighted loss: {use_severity}\n"
          f"  train/val/test = {len(tr)}/{len(va)}/{len(te)}\n{'=' * 62}")

    history = []
    best_val, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for x, y, level, w in dl_tr:
            x, y = x.to(device), y.to(device)
            level, w = level.to(device), w.to(device)
            logit, level_logits = model(x)

            per_sample = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            if use_severity:
                # Normalising by mean weight keeps the gradient scale (and so the
                # effective learning rate) comparable to the unweighted model --
                # otherwise Model B would just be training "harder".
                loss = (per_sample * w).sum() / w.sum()
            else:
                loss = per_sample.mean()
            if level_logits is not None:
                loss = loss + args.level_loss_weight * F.cross_entropy(level_logits, level)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()

        y, p, lv, sw = evaluate(model, dl_va)
        val = full_report(y, p, lv, sw)
        # Model selection uses the same criterion the model was trained for:
        # plain accuracy for A, severity-weighted risk for B.
        score = -val["severity_weighted_risk"] if use_severity else val["accuracy"]
        history.append(dict(epoch=epoch, train_loss=tot / seen,
                            val_accuracy=val["accuracy"], val_f1=val["f1"],
                            val_auc=val["roc_auc"],
                            val_weighted_risk=val["severity_weighted_risk"]))
        flag = ""
        if score > best_val:
            best_val, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  *"
        print(f"  epoch {epoch:2d}/{args.epochs}  loss {tot / seen:.4f}  "
              f"val_acc {val['accuracy']:.3f}  val_f1 {val['f1']:.3f}  "
              f"val_auc {val['roc_auc']:.3f}  val_risk {val['severity_weighted_risk']:.3f}{flag}")

    model.load_state_dict(best_state)
    y, p, lv, sw = evaluate(model, dl_te)
    test = full_report(y, p, lv, sw)
    print(f"  best epoch {best_epoch} | TEST acc {test['accuracy']:.3f}  f1 {test['f1']:.3f}  "
          f"auc {test['roc_auc']:.3f}  weighted-risk {test['severity_weighted_risk']:.3f}  "
          f"({time.time() - t0:.0f}s)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if seed == SEED:  # keep the weights of the reference run only
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{name}.pt"))
    return dict(name=name, seed=seed, with_level_head=with_level_head,
                use_severity=use_severity,
                best_epoch=best_epoch, history=history, test=test,
                test_scores=p.tolist(), test_labels=y.tolist(), test_levels=lv.tolist(),
                test_weights=sw.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--level-loss-weight", type=float, default=0.3)
    ap.add_argument("--ablation", action="store_true", default=True,
                    help="also train weights-only and level-head-only variants")
    ap.add_argument("--no-ablation", dest="ablation", action="store_false")
    ap.add_argument("--runs", type=int, default=3,
                    help="repeat both models with different seeds; the test set "
                         "is small, so a single run is noisy")
    args = ap.parse_args()

    rows = load_manifest()
    print(f"{len(rows)} images from the manifest, device={device}, runs={args.runs}")

    # A and B are the headline comparison; C and D split B's two ingredients
    # apart so the report can say which one actually does the work.
    configs = [
        ("model_a_flat", False, False),
        ("model_b_severity_aware", True, True),
    ]
    if args.ablation:
        configs += [
            ("model_c_weights_only", False, True),
            ("model_d_levelhead_only", True, False),
        ]

    seeds = [SEED + 1000 * k for k in range(args.runs)]
    runs = {name: [] for name, _, _ in configs}
    for seed in seeds:
        for name, head, sev in configs:
            runs[name].append(train_one(name, with_level_head=head, use_severity=sev,
                                        rows=rows, args=args, seed=seed))

    # The reference run (seed == SEED) is what the figures use; the other seeds
    # give the spread reported next to it.
    models = [next(r for r in runs[n] if r["seed"] == SEED)
              for n in ("model_a_flat", "model_b_severity_aware")]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(config=vars(args), levels=LEVELS, seeds=seeds,
                       models=models, runs=runs), f, indent=2)
    print(f"\nresults -> {os.path.relpath(out, PROJECT_DIR)}")

    print(f"\n{'metric':<26}{'A: flat':>20}{'B: severity-aware':>22}{'delta':>10}")
    for k in ("accuracy", "f1", "roc_auc", "severity_weighted_risk", "high_stakes_error_rate"):
        va = np.array([r["test"][k] for r in runs["model_a_flat"]])
        vb = np.array([r["test"][k] for r in runs["model_b_severity_aware"]])
        print(f"{k:<26}{va.mean():>13.4f} +-{va.std():.3f}"
              f"{vb.mean():>15.4f} +-{vb.std():.3f}{vb.mean() - va.mean():>+10.4f}")


if __name__ == "__main__":
    main()
