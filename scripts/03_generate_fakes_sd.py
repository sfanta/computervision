"""
Generate the synthetic (deepfake) half of the dataset with Stable Diffusion 1.5.

    python scripts/03_generate_fakes_sd.py            # every set
    python scripts/03_generate_fakes_sd.py L5 L2      # only some

Sets
    L0  50  abstract / procedural renders            (text2img)
    L2  50  commercial product / counterfeit renders (img2img, seeded from ABO)
    L3  50  press-photo style portraits              (text2img)
    L5  50  armed-conflict scenes                    (text2img)

Notes
  * On CUDA the pipeline runs in float16: ~1.5 s per 512x512 image on an RTX
    5070, so a full 200-image run takes a few minutes.  The original script was
    slow because MPS forces float32 -- and the all-black outputs were the known
    MPS float16 NaN bug, which does not occur on CUDA.  A blank-output guard is
    kept anyway (see `is_blank`): any near-uniform image is retried with a
    different seed instead of being silently written to disk.
  * Purpose is deepfake-*detection* research: these images are training data for
    a classifier, they carry a `synthetic` marker in their EXIF (added by
    05_build_manifest.py) and depict no real person or real brand.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
MODEL_ID = "runwayml/stable-diffusion-v1-5"
TARGET = 50
STEPS = 30
GUIDANCE = 8.0
SIZE = 512
MAX_RETRIES = 4

NEGATIVE = ("cartoon, anime, painting, drawing, illustration, low quality, "
            "blurry, watermark, text, logo, deformed, unrealistic")

device = "cuda" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
# float16 is only safe on CUDA; MPS float16 yields NaN latents -> black images.
dtype = torch.float16 if device == "cuda" else torch.float32


# ── prompt banks ─────────────────────────────────────────────────────────────
L5_PROMPTS = [
    "photorealistic war zone, destroyed city buildings, smoke and fire, rubble, soldiers in urban combat, gritty documentary photography, 4k",
    "soldiers advancing through destroyed urban streets, war photography, dust and smoke, realistic, high detail",
    "burning city block during military conflict, civilians fleeing, documentary war photo, realistic",
    "snipers on rooftop overlooking destroyed city, urban warfare, photorealistic, dramatic lighting",
    "military convoy moving through bombed-out city, war documentary style, realistic photograph",
    "soldiers in trench warfare, battlefield explosions, muddy ground, dark sky, photorealistic, war photography",
    "military helicopters over conflict zone, explosions below, realistic war photograph, dramatic",
    "soldiers under artillery fire, battlefield smoke, photorealistic documentary style",
    "tank advancing on battlefield, destroyed vehicles, smoke, realistic war photograph",
    "soldiers treating wounded comrade under fire, battlefield, photorealistic war journalism",
    "war propaganda poster style, realistic military imagery, soldiers and explosions, dramatic composition",
    "photorealistic military leader addressing troops before battle, dramatic war imagery",
    "realistic image of military airstrike, explosion, conflict zone, propaganda style",
    "news-style war photograph, soldiers and civilians in conflict zone, realistic",
    "viral social media war image, soldiers in urban combat, photorealistic, disinformation style",
    "photorealistic airstrike explosion, buildings collapsing, smoke and fire, conflict zone, documentary",
    "military jets dropping bombs on urban area, realistic war photography, dramatic sky",
    "missile impact in city center, large explosion, realistic conflict photography, smoke",
    "aerial bombing of military target, explosion and destruction, photorealistic war image",
    "drone strike aftermath, destroyed compound, smoke, realistic conflict photograph",
    "refugees fleeing war zone, crowded border crossing, photorealistic documentary photography",
    "civilians sheltering in bombed building, war zone, realistic photojournalism style",
    "family fleeing burning city, war destruction background, emotional documentary photograph",
    "displaced people in conflict zone camp, tents and barbed wire, realistic documentary",
    "medical workers evacuating civilians in war zone, dramatic photorealistic war photography",
    "warship firing missiles at night, realistic military photograph, dramatic sea battle",
    "submarine surfacing in conflict zone, military operation, photorealistic photograph",
    "aircraft carrier strike group in formation, war footing, realistic military photography",
    "naval bombardment of coastal city, explosions and smoke, photorealistic war image",
    "military amphibious assault landing, soldiers on beach under fire, photorealistic",
    "night vision style military raid, soldiers breaching building, green tint, realistic",
    "soldiers in night combat, tracer rounds, explosions in darkness, photorealistic",
    "military special forces night operation, urban environment, photorealistic photograph",
    "artillery firing at night, dramatic muzzle flash, battlefield darkness, realistic",
    "military base under rocket attack at night, explosions and alarm lights, realistic",
    "soldiers in hazmat suits in conflict zone, devastated landscape, photorealistic documentary",
    "military convoy with protective vehicles, conflict zone, realistic photograph",
    "abandoned town after conflict, eerie atmosphere, photorealistic documentary style",
    "military checkpoint with armed soldiers, tension, photorealistic war zone atmosphere",
    "UN peacekeepers in active conflict zone, blue helmets, smoke, realistic documentary",
    "propaganda broadcast truck in war zone, soldiers and camera crews, photorealistic",
    "war correspondent reporting from active conflict zone, explosions nearby, realistic",
    "screenshot of a viral war video, soldiers in combat, social media aesthetic",
    "satellite image showing military buildup, realistic overhead view",
    "news anchor reporting a war story, realistic studio setting",
    "photorealistic image of modern soldiers styled as historical war photograph, black and white",
    "colorized war photograph style, soldiers in mid-20th century conflict, realistic render",
    "vintage news wire photograph aesthetic, soldiers in combat, photorealistic",
    "old combat photograph look, grainy, soldiers under fire, photorealistic",
    "staged combat scene in the style of a historical military photograph, photorealistic",
]

# L2: product renders.  Brands are fictional on purpose -- the research question
# is "can a detector spot a synthetic product photo", not "can we copy a brand".
_PRODUCT_SUBJECTS = [
    "luxury wristwatch with chronograph dials", "leather handbag on a plain backdrop",
    "wireless over-ear headphones", "running sneaker, side view",
    "smartphone with a glass back", "ceramic coffee mug",
    "stainless steel cookware set", "designer sunglasses",
    "mechanical keyboard with backlit keys", "electric kettle",
    "cordless power drill", "fountain pen with a metal cap",
    "modern desk lamp", "bluetooth speaker cylinder",
    "backpack with multiple compartments", "wooden dining chair",
    "cosmetic serum bottle with a dropper", "wristband fitness tracker",
    "espresso machine, stainless finish", "office swivel chair",
    "insulated water bottle", "gaming mouse with RGB lighting",
    "digital camera body with a lens", "table fan",
    "toaster with two slots",
]
L2_PROMPTS = []
for _s in _PRODUCT_SUBJECTS:
    L2_PROMPTS.append(f"professional e-commerce product photo of a {_s}, "
                      f"pure white background, studio lighting, sharp focus, catalog listing, 4k")
    L2_PROMPTS.append(f"photorealistic 3d product render of a counterfeit-luxury style {_s}, "
                      f"white seamless background, marketplace listing photo, glossy, high detail")

L3_PROMPTS = [
    "press photograph of a fictional head of state at a podium, national flags behind, photorealistic",
    "official portrait of a fictional government minister in a suit, studio lighting, photorealistic",
    "photorealistic press conference photo of a fictional politician answering questions, microphones",
    "candid photograph of a fictional diplomat shaking hands at a summit, photorealistic",
    "news photograph of a fictional senator walking through a corridor, photojournalism style",
    "photorealistic portrait of a fictional public figure at a red carpet event, flash photography",
    "photorealistic headshot of a fictional news anchor in a broadcast studio",
    "fictional celebrity photographed by paparazzi outside a restaurant, photorealistic",
    "official identity-style photograph of a fictional person, neutral grey background, photorealistic",
    "photorealistic photo of a fictional business executive giving a keynote on stage",
]
L3_PROMPTS = [f"{p}, {v}" for p in L3_PROMPTS
              for v in ("middle aged man", "middle aged woman", "young man",
                        "young woman", "elderly person")]

L1_PROMPTS = [
    "photograph of a cluttered kitchen counter with fruit and a kettle, natural light, snapshot",
    "amateur photo of a dog lying on a living room carpet, phone camera",
    "candid photo of people eating at a small restaurant table, warm indoor light",
    "photograph of a bicycle leaning against a brick wall on a city street",
    "snapshot of a park bench under trees on an overcast day",
    "photo of a home office desk with a laptop, notebook and coffee cup",
    "photograph of a supermarket aisle with shelves of groceries",
    "street photograph of a bus stop with people waiting, daytime",
    "photo of a cat sitting on a windowsill, curtains, daylight",
    "photograph of a messy child bedroom with toys on the floor",
    "snapshot of a plate of pasta on a wooden table, casual food photo",
    "photograph of a suburban street with parked cars, cloudy sky",
    "photo of a laundry room with a washing machine and baskets",
    "photograph of a train platform with commuters, motion blur",
    "photo of a bathroom sink with toiletries, ordinary interior",
    "photograph of a beach with a few people and umbrellas, summer day",
    "snapshot of a birthday cake with candles on a dining table",
    "photograph of a construction site with a crane, daytime",
    "photo of a bookshelf filled with books and small ornaments",
    "photograph of a rainy sidewalk with reflections and umbrellas",
    "photo of a garden with potted plants and a watering can",
    "photograph of an office meeting room with chairs and a whiteboard",
    "snapshot of a farmers market stall with vegetables",
    "photograph of a gym interior with exercise machines",
    "photo of a parking lot at dusk with a few cars",
]

# L4: political scenes.  Used as a top-up when the OpenFake political subset
# does not yield 50 usable fakes (its filter API pages slowly at high offsets).
L4_PROMPTS = [
    "photorealistic press photo of a large political rally, crowd waving generic flags, stage lighting",
    "photorealistic news photo of an international summit, delegates around a round table, flags",
    "documentary photograph of a street protest with banners, daytime, crowd, realistic",
    "photorealistic photo of a parliament chamber during a session, wide angle",
    "news photograph of a campaign rally at night, crowd with signs, stage, realistic",
    "photorealistic image of a diplomatic handshake in front of national flags, press pool",
    "photograph of a polling station with voters queueing, realistic documentary style",
    "photorealistic photo of riot police facing demonstrators in a city square",
    "news photo of a state ceremony with honour guard and flags, photorealistic",
    "photorealistic photograph of a televised political debate on a studio stage",
    "documentary photo of a border checkpoint with officials and vehicles, realistic",
    "photorealistic press photo of a government building surrounded by journalists",
    "photograph of a peace negotiation table with delegations, realistic news style",
    "photorealistic photo of a general strike march through a capital city",
    "news photograph of an emergency government press briefing, podium and microphones",
    "photorealistic image of an embassy entrance with security and flags",
    "documentary photo of election night results on large screens, crowd watching",
    "photorealistic photo of a UN-style general assembly hall, delegates seated",
    "press photo of a motorcade of black cars with security escort, realistic",
    "photorealistic photograph of a political banner drop from a bridge, activists",
    "news photo of a refugee reception centre with officials, documentary realism",
    "photorealistic photo of a cabinet meeting around a long table, wide shot",
    "documentary photograph of a national day military parade on a boulevard",
    "photorealistic press photo of a signing ceremony for an international treaty",
    "photograph of a crowd celebrating an election result in a public square, realistic",
]
L4_PROMPTS = [f"{p}, {v}" for p in L4_PROMPTS for v in ("wide shot", "telephoto news crop")]

L0_PROMPTS = [
    "3d procedural geometric render, abstract shapes, soft studio lighting, high detail",
    "fractal geometry render, recursive patterns, vivid colors, abstract art",
    "abstract generative art, flowing gradients, smooth surfaces, 4k render",
    "procedural noise texture, organic pattern, macro abstract",
    "isometric abstract 3d composition, pastel colors, clean render",
    "voronoi cellular pattern render, abstract, high resolution",
    "abstract metallic liquid surface, iridescent reflections, 3d render",
    "kaleidoscopic fractal pattern, symmetric, deep colors, abstract",
    "abstract crystalline structure render, translucent, studio light",
    "generative wave interference pattern, monochrome, abstract art",
    "procedural terrain heightmap visualization, abstract topography",
    "abstract 3d tessellation of polygons, ambient occlusion, render",
    "spiral fractal art, logarithmic curves, colorful abstract",
    "abstract fluid simulation render, swirling colors, high detail",
    "geometric minimal 3d scene, spheres and cubes, soft shadows",
    "abstract fibrous texture render, macro, monochrome",
    "mandelbrot set zoom render, intricate detail, vivid palette",
    "abstract woven lattice structure, 3d render, neutral colors",
    "procedural marble veining pattern, abstract macro render",
    "abstract particle field render, depth of field, dark background",
    "cellular automata pattern visualization, abstract, high contrast",
    "abstract layered paper cut geometry, gradient colors, 3d render",
    "reaction diffusion pattern render, organic abstract texture",
    "abstract chrome tubes composition, studio reflections, 3d render",
    "procedural sand ripple pattern, abstract overhead render",
]
L0_PROMPTS = [f"{p} [{v}]" for p in L0_PROMPTS for v in ("variation a", "variation b")]


# ── helpers ──────────────────────────────────────────────────────────────────
def is_blank(img, std_threshold=3.0):
    """Detect the failure mode where the sampler returns NaN latents and the
    decoder emits a flat (usually pure black) frame."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    return bool(np.isnan(a).any() or a.std() < std_threshold)


def out_dir(name):
    d = os.path.join(IMAGES_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def existing(d):
    return {f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))}


def load_pipes(need_img2img):
    from diffusers import StableDiffusionPipeline
    print(f"Loading {MODEL_ID} on {device} ({dtype})...")
    txt = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False, low_cpu_mem_usage=True).to(device)
    txt.set_progress_bar_config(disable=True)
    txt.enable_attention_slicing()
    img = None
    if need_img2img:
        from diffusers import StableDiffusionImg2ImgPipeline
        img = StableDiffusionImg2ImgPipeline(**txt.components,
                                             requires_safety_checker=False)
        img.set_progress_bar_config(disable=True)
        img.enable_attention_slicing()
    return txt, img


def generate(pipe, folder, prefix, prompts, seed_base, init_images=None,
             strength=0.72, target=TARGET):
    """Bring images/<folder> up to `target` images, resuming if some exist."""
    d = out_dir(folder)
    have = existing(d)
    # `target` is the size of the whole folder, not this generator's share:
    # L0DeepFake and L1DeepFake already hold images from other sources, and
    # over-generating here would unbalance them against their real counterpart.
    need = max(0, target - len(have))
    print(f"\n== {folder}: {len(have)} present, target {target} -> generating {need}")
    made, i = 0, 0
    while made < need and i < target * 3:
        name = f"{prefix}_{i + 1:03d}.jpg"
        dest = os.path.join(d, name)
        idx = i
        i += 1
        if os.path.exists(dest):
            continue
        prompt = prompts[idx % len(prompts)]
        img = None
        for attempt in range(MAX_RETRIES):
            g = torch.Generator(device="cpu").manual_seed(seed_base + idx * 17 + attempt * 9973)
            kw = dict(prompt=prompt, negative_prompt=NEGATIVE,
                      num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=g)
            if init_images is None:
                cand = pipe(width=SIZE, height=SIZE, **kw).images[0]
            else:
                init = init_images[idx % len(init_images)].resize((SIZE, SIZE), Image.LANCZOS)
                cand = pipe(image=init, strength=strength, **kw).images[0]
            if not is_blank(cand):
                img = cand
                break
            print(f"   blank output, retrying ({attempt + 1}/{MAX_RETRIES})")
        if img is None:
            print(f"   {name} FAILED (blank after {MAX_RETRIES} tries) - skipped")
            continue
        img.save(dest, "JPEG", quality=95)
        made += 1
        print(f"   [{made:03d}/{need}] {name}  <- {prompt[:60]}...")
    print(f"== {folder}: {len(existing(d))}/{target} on disk (+{made} this run)")


def abo_seed_images():
    """50 real ABO product photos used as img2img seeds for L2DeepFake."""
    src = os.path.join(IMAGES_DIR, "L2Real")
    files = sorted(f for f in os.listdir(src)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit("images/L2Real is empty - run 01_collect_real.py L2 first")
    return [Image.open(os.path.join(src, f)).convert("RGB") for f in files[:TARGET]]


SETS = {
    "L0": dict(folder="L0DeepFake", prefix="sd_abstract", prompts=L0_PROMPTS, seed=100, img2img=False),
    # L1DeepFake already holds 25 StyleGAN faces; add 25 generated scenes so the
    # synthetic half matches the face/scene mix of the real half.
    "L1": dict(folder="L1DeepFake", prefix="sd_scene", prompts=L1_PROMPTS, seed=300,
               img2img=False, target=50),
    "L2": dict(folder="L2DeepFake", prefix="sd_product", prompts=L2_PROMPTS, seed=500, img2img=True),
    "L3": dict(folder="L3DeepFake", prefix="sd_portrait", prompts=L3_PROMPTS, seed=700, img2img=False),
    # Top-up only: L4DeepFake is primarily OpenFake's own political fakes.
    "L4": dict(folder="L4DeepFake", prefix="sd_political", prompts=L4_PROMPTS, seed=850, img2img=False),
    "L5": dict(folder="L5DeepFake", prefix="sd_conflict", prompts=L5_PROMPTS, seed=1000, img2img=False),
}

if __name__ == "__main__":
    # L4 is a top-up set, so it is not part of the default "everything" run.
    wanted = [a.upper() for a in sys.argv[1:]] or [k for k in SETS if k != "L4"]
    need_i2i = any(SETS[w]["img2img"] for w in wanted)
    txt_pipe, img_pipe = load_pipes(need_i2i)
    for key in wanted:
        cfg = SETS[key]
        tgt = cfg.get("target", TARGET)
        if cfg["img2img"]:
            generate(img_pipe, cfg["folder"], cfg["prefix"], cfg["prompts"],
                     cfg["seed"], init_images=abo_seed_images(), target=tgt)
        else:
            generate(txt_pipe, cfg["folder"], cfg["prefix"], cfg["prompts"],
                     cfg["seed"], target=tgt)
    print("\nDone.")
