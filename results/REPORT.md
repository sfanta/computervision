# Deepfake detection with sensitivity-level awareness

Two detectors, identical backbone (ResNet-18, ImageNet init), identical
splits and identical augmentation. The only difference is whether the
sensitivity level of an image is used during training.

| | Model A | Model B |
|---|---|---|
| binary head | yes | yes |
| level supervision | none | auxiliary 6-way level head |
| loss weighting | uniform | per-sample severity weight |
| level needed at inference | no | no |

## Dataset

600 images, 6 sensitivity levels, balanced real/deepfake per level.

| level | description | real | deepfake | severity | real source | deepfake source |
|---|---|---|---|---|---|---|
| L0 | Procedural / abstract | 50 | 50 | 0.5 | dtd | bing-image-creator, stable-diffusion-1.5 |
| L1 | Everyday | 50 | 50 | 1.0 | coco2017, ffhq | stable-diffusion-1.5, thispersondoesnotexist |
| L2 | Commercial | 50 | 50 | 1.5 | amazon-berkeley-objects | stable-diffusion-1.5-img2img |
| L3 | Identifiable people / national | 50 | 50 | 2.5 | lfw, wikimedia-commons | stable-diffusion-1.5 |
| L4 | International / political | 50 | 50 | 3.5 | openfake-real, wikimedia-commons | stable-diffusion-1.5 |
| L5 | Military / armed conflict | 50 | 50 | 4.0 | openfake-real, wikimedia-commons | stable-diffusion-1.5 |

![dataset](figures/01_dataset_composition.png)

![samples](figures/01b_samples.png)

## Headline results (test split)

Overall accuracy is a wash (0.829 vs 0.829), but Model B makes 38% fewer mistakes on the levels that matter: the L3-L5 error rate drops from 0.167 to 0.103. The category signal does not make the detector better in general -- it redistributes the errors it was going to make anyway towards the cheap end of the scale.

| metric | A: flat | B: severity-aware | delta |
|---|---|---|---|
| accuracy | 0.8294 ± 0.0148 | 0.8294 ± 0.0570 | +0.0000 (tie) |
| precision | 0.8434 ± 0.0204 | 0.8397 ± 0.0683 | -0.0036 (A better) |
| recall | 0.8095 ± 0.0194 | 0.8175 ± 0.0405 | +0.0079 (B better) |
| F1 | 0.8259 ± 0.0151 | 0.8282 ± 0.0542 | +0.0023 (B better) |
| ROC-AUC | 0.9076 ± 0.0163 | 0.8964 ± 0.0414 | -0.0111 (A better) |
| **severity-weighted risk** | 0.1813 ± 0.0175 | 0.1447 ± 0.0337 | -0.0366 (B better) |
| **L3-L5 error rate** | 0.1667 ± 0.0194 | 0.1032 ± 0.0224 | -0.0635 (B better) |

Severity-weighted risk is the error rate with every mistake priced by
its level's weight (L0 = 0.5 up to L5 = 4.0). It is the metric Model B
is optimised for, and the one that matters if an undetected conflict
deepfake costs more than an undetected texture render.

![overall](figures/03_overall_metrics.png)

## Per level

| level | severity | n | acc A | acc B | F1 A | F1 B | err A | err B |
|---|---|---|---|---|---|---|---|---|
| L0 | 0.5 | 14 | 0.714 | 0.786 | 0.600 | 0.727 | 0.286 | 0.214 |
| L1 | 1.0 | 14 | 1.000 | 0.857 | 1.000 | 0.875 | 0.000 | 0.143 |
| L2 | 1.5 | 14 | 0.857 | 0.857 | 0.857 | 0.857 | 0.143 | 0.143 |
| L3 | 2.5 | 14 | 0.929 | 1.000 | 0.923 | 1.000 | 0.071 | 0.000 |
| L4 | 3.5 | 14 | 0.857 | 1.000 | 0.875 | 1.000 | 0.143 | 0.000 |
| L5 | 4.0 | 14 | 0.714 | 0.786 | 0.714 | 0.769 | 0.286 | 0.214 |

## Ablation: which half of the category signal helps?

B combines a severity-weighted loss with an auxiliary level head. C and D use one each.

| variant | severity-weighted loss | level head | accuracy | severity-weighted risk | L3-L5 error rate |
|---|---|---|---|---|---|
| a flat | no | no | 0.8294 ± 0.0148 | 0.1813 ± 0.0175 | 0.1667 ± 0.0194 |
| c weights only | yes | no | 0.8333 ± 0.0350 | 0.1630 ± 0.0402 | 0.1270 ± 0.0449 |
| d levelhead only | no | yes | 0.8135 ± 0.0736 | 0.1639 ± 0.0791 | 0.1190 ± 0.0673 |
| b severity aware | yes | yes | 0.8294 ± 0.0570 | 0.1447 ± 0.0337 | 0.1032 ± 0.0224 |

![per level](figures/04_per_level.png)

![risk](figures/07_risk_breakdown.png)

## Training and discrimination

![curves](figures/02_training_curves.png)

![roc](figures/05_roc.png)

![confusion](figures/06_confusion.png)

## What to be careful about when reading this

* **The test split is ~90 images.** One image is worth roughly a
  percentage point, which is why every configuration is trained with
  several seeds and reported as mean ± std. Differences smaller than
  the spread are not differences.
* **The deepfake halves of L3-L5 are all Stable Diffusion 1.5.** SD 1.5
  is a 2022-era generator and is easier to spot than what a motivated
  actor would use today, so absolute accuracy at those levels is
  optimistic. The A-vs-B comparison is still valid: both models see
  exactly the same images.
* **Real and synthetic halves come from different corpora.** Content is
  matched within each level (L2's fakes are img2img renders of the real
  ABO products, L1 mixes faces and scenes on both sides), and every
  image is re-encoded identically, but capture conditions still differ
  between a press photograph and a diffusion sample.
* **L4 and L5 are deliberately kept apart** even though their brief
  treats them as near-equivalent. Merging them gives the 10-category
  layout; keeping them split lets the per-level table show whether
  political and conflict imagery actually behave the same way.

## Reproducing

```bash
python scripts/01_collect_real.py           # real half + L4/L5 candidate pools
python scripts/02_curate_openfake.py        # CLIP-select L4/L5 real
python scripts/03_generate_fakes_sd.py      # synthetic half (Stable Diffusion)
python scripts/03_generate_fakes_sd.py L4   # political fakes
python scripts/04_topup_l1_fake.py          # StyleGAN faces for L1
python scripts/05_build_manifest.py         # normalise + label + split
python scripts/06_train.py --epochs 25      # train A, B and the ablation
python scripts/07_report.py                 # figures + this report
```
