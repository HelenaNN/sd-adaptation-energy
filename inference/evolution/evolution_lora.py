from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageDraw, ImageFont
import os

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------

base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
placeholder_token = "<piranesi-style>"

# Entrenamientos: (label, base_dir, usa learned_embeds-steps-N.safetensors)
trainings = {
    "Titan batch12": "/home/helena/Desktop/tfm/experiments/sdv15/lora/outputs_titan_with_cp/0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
    "Titan batch6": "/home/helena/Desktop/tfm/experiments/sdv15/lora/outputs_titan_with_cp/0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
    "A40 batch24":   "/home/helena/Desktop/tfm/experiments/sdv15/lora/outputs_a40_with_cp/0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
    "A40 batch6":    "/home/helena/Desktop/tfm/experiments/sdv15/lora/outputs_a40_with_cp/0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
    "A40 batch18":   "/home/helena/Desktop/tfm/experiments/sdv15/lora/outputs_a40_with_cp/0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18",
}


checkpoints = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

prompts = [
    "a drawing of a building",
    "A drawing of a fireplace",
    "a temple in ruins",
    "a plan of a house",
]

seed = 1337
num_inference_steps = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

output_dir = "results/evolution_LoRA"
os.makedirs(output_dir, exist_ok=True)


# ----------------------------------------------------------------------
# FUNCIÓN: generar imagen con semilla fija
# ----------------------------------------------------------------------

def generate(pipe, prompt, seed, steps=50):
    generator = torch.Generator(device=device).manual_seed(seed)

    return pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]


# ----------------------------------------------------------------------
# FUNCIÓN: crear grid con etiquetas
# ----------------------------------------------------------------------

def make_grid(images, labels, title, img_size=256):
    n = len(images)

    cols = n
    rows = 1

    pad = 4
    label_h = 24
    title_h = 32

    w = cols * (img_size + pad) + pad
    h = title_h + rows * (img_size + label_h + pad) + pad

    grid = Image.new(
        "RGB",
        (w, h),
        color=(245, 245, 245)
    )

    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            13
        )

        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            15
        )

    except:
        font = ImageFont.load_default()
        title_font = font

    # Título
    draw.text(
        (pad, pad),
        title,
        fill=(50, 50, 50),
        font=title_font
    )

    for idx, (img, label) in enumerate(zip(images, labels)):

        col = idx % cols

        x = pad + col * (img_size + pad)
        y = title_h + pad

        img_resized = img.resize((img_size, img_size))

        grid.paste(img_resized, (x, y))

        draw.text(
            (x + 2, y + img_size + 2),
            label,
            fill=(80, 80, 80),
            font=font
        )

    return grid


# ----------------------------------------------------------------------
# FUNCIÓN: localizar pesos LoRA
# ----------------------------------------------------------------------

def get_lora_path(train_dir, ckpt):
    """
    Busca los pesos LoRA correspondientes al checkpoint.

    Estructura típica de Diffusers:

    train_dir/
        checkpoint-500/
            pytorch_lora_weights.safetensors

        checkpoint-1000/
            pytorch_lora_weights.safetensors
    """

    checkpoint_dir = os.path.join(
        train_dir,
        f"checkpoint-{ckpt}"
    )

    lora_path = os.path.join(
        checkpoint_dir,
        "pytorch_lora_weights.safetensors"
    )

    return lora_path


# ----------------------------------------------------------------------
# BUCLE PRINCIPAL
# ----------------------------------------------------------------------

for prompt in prompts:

    safe_prompt = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in prompt
    )[:40]

    prompt_dir = os.path.join(
        output_dir,
        safe_prompt
    )

    os.makedirs(
        prompt_dir,
        exist_ok=True
    )

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    for train_label, train_dir in trainings.items():

        print(f"\n--- {train_label} ---")

        # --------------------------------------------------------------
        # Cargar SD base una sola vez
        # --------------------------------------------------------------

        pipe = AutoPipelineForText2Image.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)

        images = []
        labels = []

        # --------------------------------------------------------------
        # 1. Imagen base
        # --------------------------------------------------------------

        print("Generando imagen base (SD original)...")

        img_base = generate(
            pipe,
            prompt,
            seed,
            num_inference_steps
        )

        images.append(img_base)
        labels.append("SD base")

        img_base.save(
            os.path.join(
                prompt_dir,
                f"{train_label.replace(' ', '_')}_base.png"
            )
        )

        # --------------------------------------------------------------
        # 2. Evolución LoRA
        # --------------------------------------------------------------

        for ckpt in checkpoints:

            lora_path = get_lora_path(
                train_dir,
                ckpt
            )

            if not os.path.isfile(lora_path):

                print(
                    f"  [WARNING] No existe {lora_path}, se omite."
                )

                continue

            print(f"  Checkpoint {ckpt}...")

            # ----------------------------------------------------------
            # Cargar LoRA
            # ----------------------------------------------------------

            pipe.load_lora_weights(
                os.path.dirname(lora_path),
                weight_name=os.path.basename(lora_path)
            )

            # ----------------------------------------------------------
            # Generar
            # ----------------------------------------------------------

            img = generate(
                pipe,
                prompt,
                seed,
                num_inference_steps
            )

            images.append(img)
            labels.append(f"step {ckpt}")

            img.save(
                os.path.join(
                    prompt_dir,
                    f"{train_label.replace(' ', '_')}_step{ckpt}.png"
                )
            )

            # ----------------------------------------------------------
            # Descargar LoRA para cargar el siguiente checkpoint
            # ----------------------------------------------------------

            pipe.unload_lora_weights()

        # --------------------------------------------------------------
        # 3. Grid de evolución
        # --------------------------------------------------------------

        grid_title = f"{train_label} — {prompt}"

        grid = make_grid(
            images,
            labels,
            grid_title,
            img_size=256
        )

        grid_path = os.path.join(
            prompt_dir,
            f"grid_{train_label.replace(' ', '_')}.png"
        )

        grid.save(grid_path)

        print(
            f"  Grid guardado en {grid_path}"
        )

        del pipe

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


print("\nFinalizado.")