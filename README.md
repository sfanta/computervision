# Deepfake detection with sensitivity-level awareness

A deepfake image detector trained twice on the same data: once as a plain
real-vs-fake classifier, and once with the *sensitivity level* of each image
used as a training signal, so that mistakes on politically or militarily
sensitive imagery are penalised harder than mistakes on abstract textures.

## Sensitivity levels

| level | meaning | real source | deepfake source | severity |
|---|---|---|---|---|
| L0 | procedural / abstract | Describable Textures Dataset | Bing Image Creator + Stable Diffusion 1.5 | 0.5 |
| L1 | everyday scenes and faces | COCO 2017 val (25) + FFHQ (25) | thispersondoesnotexist / StyleGAN2 (25) + SD 1.5 scenes (25) | 1.0 |
| L2 | commercial products | Amazon Berkeley Objects | SD 1.5 img2img seeded from the ABO photos | 1.5 |
| L3 | identifiable people | LFW (25) + Wikimedia Commons press photos (25) | SD 1.5 press-photo portraits | 2.5 |
| L4 | politics / international | Wikimedia Commons, CLIP-curated | SD 1.5 political scenes | 3.5 |
| L5 | armed conflict | Wikimedia Commons, CLIP-curated | SD 1.5 conflict scenes | 4.0 |

50 real + 50 deepfake per level. The severity column is the cost multiplier
applied to a classification error at that level.

## Pipeline

```bash
python scripts/01_collect_real.py           # real half + L4/L5 candidate pools
python scripts/02_curate_openfake.py        # CLIP-select the best 50 for L4/L5 real
python scripts/03_generate_fakes_sd.py      # synthetic half (Stable Diffusion 1.5)
python scripts/03_generate_fakes_sd.py L4   # political fakes (top-up set)
python scripts/04_topup_l1_fake.py          # StyleGAN2 faces for L1
python scripts/05_build_manifest.py         # normalise, label, split
python scripts/06_train.py --epochs 25      # train A, B and the ablation
python scripts/07_report.py                 # figures + results/REPORT.md
```

`images/` holds the raw downloads and generations, one folder per level and
class (`L5Real`, `L5DeepFake`, ...). `dataset/images/` holds the normalised
copies that training actually reads, and `dataset_manifest.csv` the labels.

## Two design decisions worth knowing about

**Normalisation is not cosmetic.** The raw pool mixes 256px ABO thumbnails,
1024px StyleGAN PNGs and 512px diffusion output. A detector trained on that
pool can hit high accuracy by reading resolution and compression history rather
than synthesis artefacts. `05_build_manifest.py` centre-crops everything to a
square, resizes to 512 and re-encodes as JPEG q95, so every class shares one
encoding history.

**Content is matched within a level.** If L1's real half were only COCO scenes
and its fake half only GAN faces, the classifier would separate the classes on
"face vs scene". L1 is therefore 25 scenes + 25 faces on *both* sides, L2's
fakes are img2img renders seeded from the real ABO photos, and L5's fakes are
conflict scenes paired against real conflict photography. Two consequences of
that rule are worth spelling out:

* **L4/L5 real halves come from Wikimedia Commons, not OpenFake.** OpenFake's
  real half is LAION/ImageNet with free-text captions and no topic labels;
  keyword-matching those captions returned fashion shots (`tank` also matches
  "tank top") and oil portraits (`military attire`) as often as war reporting.
  `02_curate_openfake.py` pools candidates from Commons topic search *and* the
  OpenFake keyword matches, then keeps the 50 that CLIP scores highest against
  level-appropriate descriptions, rejecting near-duplicate frames of the same
  event. Commons wins almost every slot; the OpenFake political/military
  *fakes* turned out to be off-topic too, which is why L4's deepfakes are
  generated locally.
* **L3's real half is half LFW, half press photography.** LFW is 250x250, so
  after normalisation to 512 it is visibly soft while the synthetic half is
  natively sharp — a detector could separate them on blur alone.

## Labels

Every image carries its labels twice:

* **In the file.** EXIF `ImageDescription` holds a human-readable summary and
  `UserComment` a JSON blob with `level`, `label`, `is_fake`,
  `severity_weight`, `source` and `image_id`.
* **In `dataset_manifest.csv`.** Same fields plus the train/val/test split
  (stratified per level × class), original resolution, provenance path and a
  SHA-1 of the source file for duplicate detection.

Read them back with:

```python
from PIL import Image
import json
json.loads(Image.open("dataset/images/L5_fake_001.jpg").getexif()[0x9286])
```

## The two models

Identical backbone (ResNet-18, ImageNet init), identical splits, identical
augmentation. The only difference is whether the level is used while training.

|  | Model A — flat | Model B — severity-aware |
|---|---|---|
| binary head | yes | yes |
| level supervision | none | auxiliary 6-way level head |
| loss weighting | uniform | per-sample severity weight |
| needs the level at inference | no | no |

Model B uses the category in two ways: each sample's binary loss is scaled by
its level's severity weight, and an auxiliary head has to predict the level
itself, which keeps category structure in the shared trunk. Both are used at
training time only — at inference Model B takes an image and nothing else, so
the comparison is like-for-like and the model stays deployable on unlabelled
input.

Two further variants isolate B's ingredients: **C** uses the weighted loss
alone, **D** the level head alone, so the report can say which half of the
category signal actually does the work.

The headline comparison metric is **severity-weighted risk**: the error rate
with every mistake priced by its level's weight. Every configuration is trained
with three seeds, because a ~90-image test split moves by a couple of points
between seeds on its own. Results, per-level breakdowns and all figures land in
`results/REPORT.md`.

## The 10 vs 12 category question

The brief lists ten categories — five levels, each split real/deepfake — with
L4 and L5 treated as roughly equivalent. Here they are kept apart, giving
twelve cells, because that is strictly more information: merging L4 and L5
afterwards is one line of code, whereas splitting them is not, and the
per-level table can then show whether political and conflict imagery actually
behave the same way or not.

## Note on the generated imagery

The synthetic images exist to train and evaluate a detector. They depict no
real person, no real event and no real brand; product prompts are generic
categories rather than trademarks, and portrait prompts specify fictional
subjects. Each generated file is marked as synthetic in its EXIF.
