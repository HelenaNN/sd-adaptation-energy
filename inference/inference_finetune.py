from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
import pandas as pd
import os

# ── General Configuration ─────────────────────────────────────────────────────
BASE_DIR = "outputs/finetuning"
OUTPUT_ROOT = "generated_images/finetuning"
PROMPTS_FILE = "prompts.csv"

# List of experiment folder names to evaluate
TRAIN_NAMES = [
    "experiment_finetuning_run_1",
]

# Checkpoint step intervals to generate images for
CHECKPOINTS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

# Generation parameters
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
UNET_WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"

# ── Load Prompts ──────────────────────────────────────────────────────────────
prompts_df = pd.read_csv(PROMPTS_FILE, header=None)
prompt_texts = prompts_df[0]

# ── Main Loop ─────────────────────────────────────────────────────────────────
for train_name in TRAIN_NAMES:
    train_dir = os.path.join(BASE_DIR, train_name)

    print(f"\n--- Loading base pipeline components for: {train_name} ---")

    # Load base pipeline once per experiment; UNet is swapped per checkpoint
    pipe = StableDiffusionPipeline.from_pretrained(
        train_dir,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    for ckpt in CHECKPOINTS:
        unet_path = os.path.join(train_dir, f"checkpoint-{ckpt}", "unet")

        if not os.path.isfile(os.path.join(unet_path, UNET_WEIGHTS_FILE)):
            print(f"[SKIP] UNet weights not found: {os.path.join(unet_path, UNET_WEIGHTS_FILE)}")
            continue

        print(f"\n=== Evaluation: {train_name} | Checkpoint: {ckpt} ===")

        # Swap in the UNet corresponding to the current checkpoint
        pipe.unet = UNet2DConditionModel.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
        ).to("cuda")

        img_output_dir = os.path.join(OUTPUT_ROOT, train_name, f"checkpoint-{ckpt}")
        os.makedirs(img_output_dir, exist_ok=True)

        for idx, prompt in enumerate(prompt_texts):
            print(f"Generating image {idx + 1}/{len(prompt_texts)} for checkpoint-{ckpt}...")
            image = pipe(
                prompt=prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            ).images[0]

            safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:50].strip()
            image.save(os.path.join(img_output_dir, f"{safe_prompt}-{idx}.png"))

    del pipe
    torch.cuda.empty_cache()
