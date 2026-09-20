from diffusers import AutoPipelineForText2Image
import torch
import pandas as pd
import os

# CONFIGURACIÓN

base_dir = "/home/helena/Desktop/tfm/experiments/sdv15/textual_inversion/outputs_a40_isolated_gpu"

train_names = [
    "0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
    "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
    "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
]

# Steps a generar (usan learned_embeds-steps-{N}.safetensors)
steps_list = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

placeholder_token = "<piranesi-style>"

prompts_file = "piranesi_prompts.csv"

output_root = "images/outputs_a40_isolated_gpu"
os.makedirs(output_root, exist_ok=True)

# CARGA DE PROMPTS
prompts = pd.read_csv(prompts_file, header=None)
prompt_texts = prompts[0]


def build_prompt(raw_prompt: str) -> str:
    """Inserta el placeholder token en el prompt si no está ya presente."""
    if placeholder_token in raw_prompt:
        return raw_prompt
    return f"{raw_prompt} {placeholder_token}"


# BUCLE PRINCIPAL: entrenamientos -> steps -> prompts

for train_name in train_names:
    train_dir = os.path.join(base_dir, train_name)

    print(f"\n--- Cargando pipeline base para {train_name} ---")

    # Pipeline base (sin textual inversion cargada todavía)
    base_pipe = AutoPipelineForText2Image.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    for step in steps_list:

        embeds_path = os.path.join(train_dir, f"learned_embeds-steps-{step}.safetensors")

        if not os.path.isfile(embeds_path):
            print(f"[WARNING] It does not exist {embeds_path}, it is passed.")
            continue

        print(f"\n=== Training: {train_name} | Step: {step} ===")

        # Cargar el embedding de textual inversion correspondiente a este step.
        # token=placeholder_token fuerza a que se registre con el token original,
        # evitando que diffusers le asigne un nombre distinto si ya existe.
        base_pipe.load_textual_inversion(
            embeds_path,
            token=placeholder_token,
        )

        img_path = os.path.join(output_root, train_name, f"step-{step}")
        os.makedirs(img_path, exist_ok=True)

        for i, raw_prompt in enumerate(prompt_texts):
            prompt = build_prompt(raw_prompt)
            print(f"Generating image {i} for {train_name} / step-{step}...")

            image = base_pipe(
                prompt,
                num_inference_steps=50,
            ).images[0]

            safe_prompt = "".join(
                c if c.isalnum() or c in " _-" else "_" for c in prompt
            )[:50]
            image.save(os.path.join(img_path, f"{safe_prompt}-{i}.png"))

        # Quitar el embedding actual antes de cargar el del siguiente step,
        # para no acumular tokens distintos en el mismo pipeline.
        base_pipe.unload_textual_inversion()

    # Liberar memoria GPU antes de pasar al siguiente entrenamiento
    del base_pipe
    torch.cuda.empty_cache()
