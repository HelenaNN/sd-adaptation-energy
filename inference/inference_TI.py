from diffusers import AutoPipelineForText2Image
import torch
import pandas as pd
import os

# ── General Configuration ─────────────────────────────────────────────────────
MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_DIR = "outputs/textual_inversion"
OUTPUT_ROOT = "generated_images/textual_inversion"
PROMPTS_FILE = "prompts.csv"

# List of experiment folder names to evaluate
TRAIN_NAMES = [
    "experiment_ti_run_1",
]

# Saved steps to generate images for
STEPS_LIST = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

# Learned placeholder token used during training
PLACEHOLDER_TOKEN = "<custom-style>"

# Generation parameters
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

# ── Load Prompts ──────────────────────────────────────────────────────────────
prompts_df = pd.read_csv(PROMPTS_FILE, header=None)
prompt_texts = prompts_df[0]


def build_prompt(raw_prompt: str, token: str) -> str:
    """Appends the placeholder token if not already in the prompt."""
    return raw_prompt if token in raw_prompt else f"{raw_prompt} {token}"


# ── Main Loop ─────────────────────────────────────────────────────────────────
for train_name in TRAIN_NAMES:
    train_dir = os.path.join(BASE_DIR, train_name)

    print(f"\n--- Loading base pipeline for: {train_name} ---")

    base_pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    for step in STEPS_LIST:
        embeds_path = os.path.join(train_dir, f"learned_embeds-steps-{step}.safetensors")

        if not os.path.isfile(embeds_path):
            print(f"[SKIP] Embedding file not found: {embeds_path}")
            continue

        print(f"\n=== Evaluation: {train_name} | Step: {step} ===")

        base_pipe.load_textual_inversion(
            embeds_path,
            token=PLACEHOLDER_TOKEN,
        )

        img_output_dir = os.path.join(OUTPUT_ROOT, train_name, f"step-{step}")
        os.makedirs(img_output_dir, exist_ok=True)

        for idx, raw_prompt in enumerate(prompt_texts):
            prompt = build_prompt(raw_prompt, PLACEHOLDER_TOKEN)
            print(f"Generating image {idx + 1}/{len(prompt_texts)} for step-{step}...")

            image = base_pipe(
                prompt=prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
            ).images[0]

            safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:50].strip()
            image.save(os.path.join(img_output_dir, f"{safe_prompt}-{idx}.png"))

        # Unload the embedding before loading the next step
        base_pipe.unload_textual_inversion()

    del base_pipe
    torch.cuda.empty_cache()
