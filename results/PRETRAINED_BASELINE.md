# Pre-trained detectors, evaluated on our sensitivity strata

Four public checkpoints, run zero-shot on the same 600 images and the
same splits our own models use. Nothing is fine-tuned. `as-is` uses the
checkpoint's own 0.5 threshold; `recalibrated` picks the threshold on our
train+val split and applies it to test.

| detector | backbone | acc (as-is) | acc (recal.) | ROC-AUC | sev-weighted risk | L3-L5 err |
|---|---|---|---|---|---|---|
| `Organika/sdxl-detector` | SwinV2, fine-tuned on SDXL output | 0.655 | 0.702 | 0.753 | 0.319 | 0.333 |
| `haywoodsloan/ai-image-detector-deploy` | SwinV2, mixed generator corpus | 0.738 | 0.726 | 0.821 | 0.261 | 0.238 |
| `Ateeqq/ai-vs-human-image-detector` | SigLIP, AI-vs-human classifier | 0.607 | 0.619 | 0.672 | 0.398 | 0.381 |
| `prithivMLmods/Deep-Fake-Detector-v2-Model` | ViT, face-deepfake oriented | 0.476 | 0.476 | 0.422 | 0.522 | 0.524 |
| **Model A (ours, flat)** | ResNet-18, trained here | - | 0.829 | 0.908 | 0.181 | 0.167 |
| **Model B (ours, severity-aware)** | ResNet-18, trained here | - | 0.829 | 0.896 | 0.145 | 0.103 |

Test split, n = 84. ROC-AUC is threshold-free and therefore
identical for both operating points. Our own rows are the mean over three
seeds and are trained on this dataset, so they are not a fair *zero-shot*
comparison -- they are the reference point the external detectors are
measured against.

![comparison](figures/08_pretrained_vs_ours.png)

## Per level (recalibrated, test split)

| detector | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| `sdxl-detector` | 0.79 | 0.57 | 0.86 | 0.79 | 0.43 | 0.79 |
| `ai-image-detector` | 0.64 | 0.64 | 0.79 | 0.86 | 0.79 | 0.64 |
| `ai-vs-human` | 0.71 | 0.71 | 0.43 | 0.71 | 0.50 | 0.64 |
| `deepfake-vit` | 0.43 | 0.50 | 0.50 | 0.50 | 0.43 | 0.50 |

## Whole dataset (600 images, recalibrated)

| detector | accuracy | ROC-AUC | sev-weighted risk | L3-L5 err |
|---|---|---|---|---|
| `sdxl-detector` | 0.663 | 0.716 | 0.358 | 0.363 |
| `ai-image-detector` | 0.825 | 0.892 | 0.147 | 0.120 |
| `ai-vs-human` | 0.713 | 0.748 | 0.290 | 0.283 |
| `deepfake-vit` | 0.527 | 0.456 | 0.488 | 0.500 |

## How to read this

* **No public detector is level-blind by accident -- it is level-blind by
  design.** None of these checkpoints knows what a sensitivity level is, so
  their errors fall wherever the data puts them. `ai-image-detector` is the
  strongest of the four and still misses roughly one L3-L5 image in four.
* **Calibration shifts, ranking survives.** Moving the threshold off 0.5
  changes accuracy by a few points at most, while ROC-AUC (threshold-free)
  separates the four checkpoints cleanly. The weak ones are weak on ranking,
  not on calibration.
* **A face-deepfake specialist collapses on a content-diverse benchmark.**
  `Deep-Fake-Detector-v2-Model` sits at chance and slightly below on AUC:
  trained on face swaps, it has nothing to say about textures, products or
  conflict photography. This is the sensitivity-stratified evaluation doing
  its job -- a single accuracy number would have hidden *where* it fails.
* **The test split is 84 images.** On all 600, `ai-image-detector` reaches
  0.825 accuracy, close to our own models' test accuracy, so the honest claim
  is not 'we beat the public detectors by 10 points' but 'we are in the same
  league, and unlike them we control where the errors land'.
* **Our fakes are Stable Diffusion 1.5.** `sdxl-detector` is tuned for SDXL
  and `ai-image-detector` for a mixed pool, so neither is being tested on the
  generator it was built for. That cuts both ways and is worth stating.

