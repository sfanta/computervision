# Project 9 — Presentation Script

13 slides. Part 1: slides 1–7, Speaker 1 (~5:10). Part 2: slides 8–13, Speaker 2
(~5:50). Full text runs about 11 minutes at a normal pace.

**If you must land on exactly 10:00**, drop any of these — they are the only parts
nothing else depends on:

* slide 3, the sentence about art counterfeiting (~10 s)
* slide 8, the normalisation paragraph (~35 s) — say instead: *"one detail: every
  image is re-encoded identically, so the model can't cheat on resolution"*
* slide 13, the second limitation about SD 1.5 (~20 s)

Do **not** cut the ethics slide or the numbers on slides 9–11: both are explicit
requirements in the brief.

---

## PART 1 — Speaker 1 (slides 1 to 7)

### Slide 1 — Title (~20 s)
Good morning everyone. We're Andrea and Nicolò, and this is our Computer Vision
project, number 9: content-sensitivity-aware deepfake detection.

The starting idea is simple. Almost every deepfake detector treats all images as
equal: a fake texture and a fake war photograph are just "one mistake" each — but
in the real world they clearly aren't. So: what happens if we tell the model,
during training, *how much a given mistake would actually cost?*

### Slide 2 — Index and preliminary information (~40 s)
Here's how we'll go through it. First the sensitivity scale we built, then how we
collected and labelled the images, then the model, and finally the results.

Two things to know up front. Every level has exactly 100 images, 50 real and 50
fake, so the dataset is perfectly balanced — 600 images in total.

And a disclaimer: the scale is ours. We built it from a journalistic
perspective, reading NYT, WSJ and Semafor coverage, because a deepfake really
does its damage the moment media picks it up. It's not neutral, and the results
are tied to it.

We also considered satire as a category and dropped it: no large dataset exists,
the phenomenon is extremely regional, and half the images we found we couldn't
interpret ourselves.

### Slide 3 — L0, abstract images (~35 s)
Level zero is abstract and procedural imagery — textures, generative art.

Impact is basically nil and it's the easiest level to spot. The only real risk is
art counterfeiting — a market that already has experts and has been dealing with
fakes forever.

Real images come from the Describable Textures Dataset; the fakes from Bing Image
Creator and Stable Diffusion 1.5 — about a third from Bing, because one of us got
rate-banned for excessive use.

*(point at the images)* Real or fake? Even at this level it's not obvious.

### Slide 4 — L1, everyday images (~35 s)
Level one is everyday life: ordinary scenes and ordinary faces.

Low impact, but not zero. Stock photography was already being misrepresented long
before AI; the danger here is limited political manipulation — a generic crowd or
face placed in the wrong context.

Real side: COCO 2017 for scenes and FFHQ for faces. Fake side:
thispersondoesnotexist and StyleGAN2 for faces, Stable Diffusion for the scenes.

Important detail: it's 25 scenes and 25 faces on *both* sides. If the real half
were only scenes and the fake half only faces, the model would just learn "face
versus scene" and we'd have measured nothing.

### Slide 5 — L2, commercial images (~35 s)
Level two: products and companies.

Medium impact — this is where fraud lives. Products that don't exist, or real
products with altered features: brand damage, and in Italy a crime under article
473.

Real images from Amazon Berkeley Objects; the fakes are Stable Diffusion img2img
renders seeded from those exact same photos, so both halves show the same objects
and the model can't cheat on content.

*(point at image)* This looks like an Apple Watch, and it costs a tenth of it.

### Slide 6 — L3, national and internal risks (~40 s)
Level three: identifiable people, and events that can create internal unrest.

High impact: a fake image of a prominent figure can move public opinion and a
decision-making process. And it's strongly culture-specific — what's explosive in
one country is nothing in another, which makes it hard to regionalise.

Real side: half LFW, half press photography from Wikimedia Commons. Fake side:
Stable Diffusion portraits in press-photo style. One honest caveat — LFW is 250
by 250, so after normalisation it looks soft next to natively sharp diffusion
output.

*(point)* Only one of these is the original — which one?

### Slide 7 — L4 and L5, international and military risk (~45 s)
The top of the scale: international and military — images built to provoke a
reaction from another country, escalate tension, potentially trigger a military
response.

Real images from Wikimedia Commons. We started from OpenFake, which is the
politically-grounded benchmark cited in the brief, but its real half is LAION with
free-text captions and no topic labels — keyword-matching gave us fashion shots,
because "tank" also matches "tank top", and oil paintings for "military attire".
So we pooled candidates from Commons and from OpenFake together, and used CLIP to
keep only the 50 that best matched a level-appropriate description, rejecting
near-duplicate frames of the same event. Commons won almost every slot: out of
100 real images at L4 and L5, only five came from OpenFake. And the political
fakes in OpenFake were off-topic too, so those we generated ourselves.

Conceptually we merged L4 and L5: the more images we looked at, the more
political and conflict imagery converged — countries have very different
thresholds for the use of force — and there just aren't many fake war photos out
there. In the dataset we kept them as separate cells, twelve instead of ten,
because merging afterwards is one line of code and splitting isn't.

Nicolò will now take you through labelling, the benchmark and the results.

---

## PART 2 — Speaker 2 (slides 8 to 13)

### Slide 8 — Labelling and training (~65 s)
Thanks. So: 600 images, and every single one was labelled twice.

Once automatically at generation time, once manually by us with a script in the
notebook — every image individually seen, none assigned in bulk. The labels live
in the file's EXIF and in a CSV manifest: level, class, severity weight, source,
and a hash for duplicate detection.

One decision cost us accuracy on purpose. The raw pool mixed 256-pixel
thumbnails, 1024-pixel PNGs and 512-pixel diffusion output, and a detector
trained on *that* gets a great score by reading resolution and compression
history instead of synthesis artefacts. So everything is centre-cropped, resized
to 512, re-encoded as JPEG q95 — one encoding history for every class. It
probably lowered our numbers; not doing it would have made them meaningless.

The architecture is identical before and after the sensitivity layer: ResNet-18
in PyTorch, same splits, same augmentation. Model A is the sensitivity-agnostic
baseline. Model B uses the level twice — each sample's loss scaled by its
severity weight, from 0.5 at L0 to 4.0 at L5, plus an auxiliary head that has to
predict the level. Both only at training time: at inference B takes an image and
nothing else.

### Slide 9 — Baseline: pre-trained detectors (~60 s)
Before our own models, we used the dataset the way a benchmark is meant to be
used: we took four public, off-the-shelf AI-image detectors and ran them on it
zero-shot. Nothing fine-tuned, same images, same splits, stratified by level.

Two findings. First, the best of them — a SwinV2 on a mixed generator pool —
reaches 0.82 ROC-AUC, which is respectable, but it still misses about one
L3-to-L5 image in four. It has no notion of where its errors land, because
nothing ever told it that some images cost more than others.

Second, and this is the one we'd point at: the face-deepfake specialist sits at
chance, essentially 0.5 AUC. Trained on face swaps, it has nothing to say about
textures, products or conflict photography. A single accuracy number would have
hidden that — the stratified evaluation is what makes it visible. That's the
argument for building the benchmark this way in the first place.

### Slide 10 — Results #1 (~65 s)
Now our two models, and the headline is an interesting one.

Overall accuracy: 0.829 for A, 0.829 for B — an exact tie, with everything else
moving by fractions of a point. If we'd stopped at accuracy, our conclusion would
have been "sensitivity does nothing".

But look at the metric we introduced: severity-weighted risk — the error rate
with every mistake priced by its level's weight. That drops from 0.181 to 0.145.
And the error rate on levels 3 to 5 drops from 0.167 to 0.103: 38% fewer mistakes
exactly where mistakes are expensive. Against the best public detector on the
same split, that's 0.238 down to 0.103.

Our ablation splits the signal in two — weighted loss alone, level head alone:
each helps, together they help most. Everything is three seeds, reported as mean
± standard deviation, because the test split is 84 images — one image is worth
about a point, so differences smaller than the spread are not differences.

### Slide 11 — Results #2 (~40 s)
Same story, broken down per level — and here the mechanism is visible.

At L3 and L4, model B goes to zero errors. At L5 it improves too. And at L1 it
gets *worse* than A. That's not a bug, that's the trade: the category signal
doesn't make the detector better in general, it redistributes the errors it was
going to make anyway towards the cheap end of the scale.

One-line summary: not better overall, better where it matters.

### Slide 12 — Ethics (~45 s)
Which brings us to something we think matters here more than usual.

A sensitivity-aware detector is, by construction, a system deciding that some
content deserves more scrutiny than other content. Three consequences.

The taxonomy becomes policy: ours is a Western, journalistic reading of harm, so
the model is most careful about the things *we* found alarming — another scale
would move that attention elsewhere.

The error cost is asymmetric by design: what you just saw at L1 is the price of
what you saw at L4. Ordinary people's images get the less careful model. That's a
defensible trade for a newsroom queue, and indefensible as a silent default.

And flagging is one step from ranking — a score for "politically sensitive" is
trivially repurposed into one that suppresses political content. Which is why
we'd put this behind a human reviewer, never in front of an automatic takedown.

### Slide 13 — Conclusion and future development (~45 s)
To close. Sensitivity awareness is not a solution to deepfakes: it doesn't detect
anything a flat model can't. What it does is let you choose where your errors
land — and that's a genuinely useful thing to be able to choose.

Two limitations. 600 images is small, and hard to grow: at the high-risk levels
large-scale deepfake datasets don't exist yet. And our L3-to-L5 fakes are all
Stable Diffusion 1.5, easier to spot than what a motivated actor would use today
— so absolute accuracy up there is optimistic, though the comparisons hold, since
every model saw the same images.

Next step is video, where the impact is much higher; audio would be the real
challenge, a different signal our scale would need rethinking for.

Thank you — we're happy to take questions.

---

## Anticipated questions (not part of the 10 minutes)

**"Which pre-trained detectors, exactly, and why those?"**
Four, chosen for architectural and training-corpus diversity:
`haywoodsloan/ai-image-detector-deploy` and `Organika/sdxl-detector` (both
SwinV2, one on a mixed generator pool, one tuned on SDXL),
`Ateeqq/ai-vs-human-image-detector` (SigLIP), and
`prithivMLmods/Deep-Fake-Detector-v2-Model` (ViT, face-oriented). All run
zero-shot, at their own 0.5 threshold and at a threshold picked on our train+val
split — the second because a checkpoint tuned on another corpus can rank well and
still be badly calibrated. Everything is in `scripts/08_pretrained_baseline.py`
and `results/PRETRAINED_BASELINE.md`.

**"Do you really beat the public detectors by ten points?"**
No, and we don't claim that. On our 84-image test split, yes — 0.829 against
0.726 for the best of them. But on all 600 images that same checkpoint reaches
0.825, so the honest statement is that we're in the same league on accuracy, and
the difference that actually holds up is *where* the errors land: 0.103 L3–L5
error for Model B against 0.238 for the best public detector.

**"Why no public-health level, since the brief mentions health crises?"**
The brief asks us to define and justify our own taxonomy. Ours is built on a
journalistic reading of harm, where the driver is the reaction an image provokes:
health misinformation lands between L3 and L4 on that axis rather than forming a
level of its own. We also couldn't source 50 credible real plus 50 fake health
images without the level collapsing into generic stock photography.

**"Why is L1 worse under Model B?"**
That's the trade working as designed: the weighted loss makes L1 mistakes cheap,
so the optimiser spends its capacity on L3–L5. Also, with 14 test images per
level, one image is about seven points, so that gap is roughly one image.

**"Isn't the accuracy tie a negative result?"**
On accuracy, yes — and we report it as a tie rather than hiding it. The claim we
make is about *where* the errors sit, which is what severity-weighted risk
measures: 0.181 down to 0.145, and 38% fewer errors on L3–L5.

**"How do we know the model isn't reading resolution instead of artefacts?"**
That's exactly why every image is centre-cropped, resized to 512 and re-encoded
as JPEG q95, so all classes share one encoding history — and why content is
matched within each level (L2's fakes are img2img renders of the real ABO
products, L1 is 25 faces + 25 scenes on both sides).
