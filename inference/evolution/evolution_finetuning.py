from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
from PIL import Image, ImageDraw, ImageFont
import os
import gc

# ── General Configuration ─────────────────────────────────────────────────────
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
OUTPUT_DIR = "results/evolution_finetuning"

# Dictionary of runs: label -> directory path
TRAININGS = {
    "Full Fine-Tuning Run 1": "outputs/finetuning/experiment_ft_run_1",
}

# Intermediate checkpoints to evaluate
CHECKPOINTS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

# Prompts for visual progression tracking
# CHANGE THIS. THERE ARE THE REFERENCE EXAMPLES
PROMPTS = [
    "a drawing of a building",
    "a drawing of a fireplace",
    "a temple in ruins",
    "a plan of a house",
]

SEED = 1337
NUM_INFERENCE_STEPS = 50
UNET_WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helper Functions ──────────────────────────────────────────────────────────
def generate(pipe, prompt, seed, steps=50):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]


def make_grid(images, labels, title, img_size=256):
    n = len(images)
    cols = n
    rows = 1
    pad = 4
    label_h = 24
    title_h = 32

    w = cols * (img_size + pad) + pad
    h = title_h + rows * (img_size + label_h + pad) + pad

    grid = Image.new("RGB", (w, h), color=(245, 245, 245))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    draw.text((pad, pad), title, fill=(50, 50, 50), font=title_font)

    for idx, (img, label) in enumerate(zip(images, labels)):
        col = idx % cols
        x = pad + col * (img_size + pad)
        y = title_h + pad

        img_resized = img.resize((img_size, img_size))
        grid.paste(img_resized, (x, y))
        draw.text((x + 2, y + img_size + 2), label, fill=(80, 80, 80), font=font)

    return grid


# ── Main Loop ─────────────────────────────────────────────────────────────────
for prompt in PROMPTS:
    safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:40].strip()
    prompt_dir = os.path.join(OUTPUT_DIR, safe_prompt)
    os.makedirs(prompt_dir, exist_ok=True)

    print(f"\n{'=' * 60}\nPrompt: {prompt}\n{'=' * 60}")

    for train_label, train_dir in TRAININGS.items():
        print(f"\n--- Evaluating: {train_label} ---")

        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=DTYPE,
            safety_checker=None,
        ).to(DEVICE)

        images = []
        labels = []

        # 1. Base model image (reference point at step 0)
        print("Generating base reference image (unadapted SD)...")
        img_base = generate(pipe, prompt, SEED, NUM_INFERENCE_STEPS)
        images.append(img_base)
        labels.append("SD base")
        img_base.save(os.path.join(prompt_dir, f"{train_label.replace(' ', '_')}_base.png"))

        # 2. Checkpoint progression (swapping UNet)
        for ckpt in CHECKPOINTS:
            checkpoint_dir = os.path.join(train_dir, f"checkpoint-{ckpt}")
            unet_dir = os.path.join(checkpoint_dir, "unet")

            if not os.path.isdir(unet_dir) or not os.path.isfile(os.path.join(unet_dir, UNET_WEIGHTS_FILE)):
                print(f"  [SKIP] UNet weights not found: {unet_dir}")
                continue

            print(f"  Generating checkpoint-{ckpt}...")
            unet = UNet2DConditionModel.from_pretrained(
                checkpoint_dir,
                subfolder="unet",
                torch_dtype=DTYPE,
            ).to(DEVICE)
            pipe.unet = unet

            img = generate(pipe, prompt, SEED, NUM_INFERENCE_STEPS)
            images.append(img)
            labels.append(f"step {ckpt}")
            img.save(os.path.join(prompt_dir, f"{train_label.replace(' ', '_')}_step{ckpt}.png"))

            del unet
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # 3. Create comparison grid
        grid_title = f"{train_label} - {prompt}"
        grid = make_grid(images, labels, grid_title, img_size=256)
        grid_path = os.path.join(prompt_dir, f"grid_{train_label.replace(' ', '_')}.png")
        grid.save(grid_path)
        print(f"  Saved progression grid: {grid_path}")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

print("\nFinished evaluation.")
