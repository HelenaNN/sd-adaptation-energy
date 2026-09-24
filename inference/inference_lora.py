from diffusers import StableDiffusionPipeline
import torch
import pandas as pd
import os

# ── General Configuration ─────────────────────────────────────────────────────
MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_DIR = "outputs/lora"
OUTPUT_ROOT = "generated_images/lora"
PROMPTS_FILE = "prompts.csv"

# List of experiment folder names to evaluate
TRAIN_NAMES = [
    "experiment_lora_run_1",
]

# Checkpoint step intervals to generate images for
CHECKPOINTS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

# Generation parameters
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
WEIGHTS_FILE = "pytorch_lora_weights.safetensors"

# ── Load Prompts ──────────────────────────────────────────────────────────────
prompts_df = pd.read_csv(PROMPTS_FILE, header=None)
prompt_texts = prompts_df[0]

# ── Main Loop ─────────────────────────────────────────────────────────────────
for train_name in TRAIN_NAMES:
    for ckpt in CHECKPOINTS:
        ckpt_path = os.path.join(BASE_DIR, train_name, f"checkpoint-{ckpt}")

        if not os.path.isfile(os.path.join(ckpt_path, WEIGHTS_FILE)):
            print(f"[SKIP] Weights file not found: {os.path.join(ckpt_path, WEIGHTS_FILE)}")
            continue

        print(f"\n=== Evaluation: {train_name} | Checkpoint: {ckpt} ===")

        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        pipe.load_lora_weights(ckpt_path, weight_name=WEIGHTS_FILE)
        pipe.to("cuda")

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

        # Release GPU VRAM between checkpoints
        del pipe
        torch.cuda.empty_cache()
