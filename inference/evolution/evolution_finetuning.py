from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
from PIL import Image, ImageDraw, ImageFont
import os
import gc

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------

base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"

trainings = {
    "Titan batch6": "/home/helena/Desktop/tfm/experiments/sdv15/finetuning/outputs_titan_outputs_with_cp/0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
    "A40 batch18":  "/home/helena/Desktop/tfm/experiments/sdv15/finetuning/outputs_a40_outputs_with_cp/0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
}

checkpoints = [
    500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
]

prompts = [
    "a drawing of a building",
    "A drawing of a fireplace",
    "a temple in ruins",
    "a plan of a house",
]

seed = 1337
num_inference_steps = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

output_dir = "results/evolution_Finetuning"
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
# FUNCIÓN: localizar checkpoint de Fine-Tuning
# ----------------------------------------------------------------------

def get_checkpoint_dir(train_dir, ckpt):

    return os.path.join(
        train_dir,
        f"checkpoint-{ckpt}"
    )


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

    print(f"\n{'=' * 60}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 60}")

    for train_label, train_dir in trainings.items():

        print(f"\n--- {train_label} ---")

        # --------------------------------------------------------------
        # 1. Cargar pipeline base
        # --------------------------------------------------------------

        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)

        images = []
        labels = []

        # --------------------------------------------------------------
        # 2. Imagen SD base
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
        # 3. Evolución Fine-Tuning
        # --------------------------------------------------------------

        for ckpt in checkpoints:

            checkpoint_dir = get_checkpoint_dir(
                train_dir,
                ckpt
            )

            unet_dir = os.path.join(
                checkpoint_dir,
                "unet"
            )

            if not os.path.isdir(unet_dir):

                print(
                    f"  [WARNING] No existe {unet_dir}, se omite."
                )

                continue

            print(f"  Checkpoint {ckpt}...")

            # ----------------------------------------------------------
            # Cargar UNet entrenada en este checkpoint
            # ----------------------------------------------------------

            unet = UNet2DConditionModel.from_pretrained(
                checkpoint_dir,
                subfolder="unet",
                torch_dtype=dtype,
            )

            unet = unet.to(device)

            # Sustituir UNet base por la entrenada
            pipe.unet = unet

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
            # Liberar memoria
            # ----------------------------------------------------------

            del unet

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

        # --------------------------------------------------------------
        # 4. Grid de evolución
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

        # --------------------------------------------------------------
        # Liberar pipeline
        # --------------------------------------------------------------

        del pipe

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()


print("\nFinalizado.")