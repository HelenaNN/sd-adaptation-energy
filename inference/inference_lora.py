from diffusers import StableDiffusionPipeline
import torch
import pandas as pd
import os

# CONFIGURACIÓN

base_dir = "/home/helena/Desktop/tfm/experiments/sdv15/lora/outputs_a40_with_cp"

train_names = [
    "0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
    "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
    "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
]

# Checkpoints a generar 
checkpoints = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

prompts_file = "piranesi_prompts.csv"

#output_root = "images_random"

output_root = "images/outputs_a40_outputs_with_cp"
os.makedirs(output_root, exist_ok=True)

# CARGA DE PROMPTS 
prompts = pd.read_csv(prompts_file, header=None)
prompt_texts = prompts[0]

# BUCLE PRINCIPAL: entrenamientos -> checkpoints -> prompts

for train_name in train_names:
    for ckpt in checkpoints:

        ckpt_path = os.path.join(base_dir, train_name, f"checkpoint-{ckpt}")
        weights_file = "pytorch_lora_weights.safetensors"

        if not os.path.isfile(os.path.join(ckpt_path, weights_file)):
            print(f"[WARNING] It does not exist {os.path.join(ckpt_path, weights_file)}, it is passed.")
            continue

        print(f"\n=== Training: {train_name} | Checkpoint: {ckpt} ===")

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        pipe.load_lora_weights(ckpt_path, weight_name=weights_file)
        pipe.to("cuda")

        img_path = os.path.join(output_root, train_name, f"checkpoint-{ckpt}")
        os.makedirs(img_path, exist_ok=True)

        for i, prompt in enumerate(prompt_texts):
            print(f"Generating image {i} for {train_name} / checkpoint-{ckpt}...")
            image = pipe(prompt=prompt).images[0]

            safe_prompt = "".join(
                c if c.isalnum() or c in " _-" else "_" for c in prompt
            )[:50]
            image.save(os.path.join(img_path, f"{safe_prompt}-{i}.png"))

        # Liberar memoria GPU antes de cargar el siguiente checkpoint
        del pipe
        torch.cuda.empty_cache()
