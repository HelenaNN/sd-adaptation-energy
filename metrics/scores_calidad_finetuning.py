import os
import csv
import clip
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

from scores_calidad import func_clip_T_score

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------

images_root = "/home/helena/Desktop/tfm/images_and_scores/sdv15/finetuning/images/outputs_a40_outputs_with_cp"

train_names = [
    "0_sdv15_finetuning_res512_Battista_steps5000_seed1337_batch6",
    "1_sdv15_finetuning_res512_Battista_steps5000_seed1337_batch6",
    "2_sdv15_finetuning_res512_Battista_steps5000_seed1337_batch6",
]

checkpoints = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

prompts_file = "piranesi_prompts.csv"

real_images_dir = "/home/helena/backup/datasets/wikiart/512_Battista_dataset"

output_csv = "scores_finetuning_a40_batch6.csv"

# Parámetros KID (mismos que en el script viejo de LoRA)
kid_image_size = (299, 299)
kid_batch_size = 32
kid_subset_size = 50  # nº de imágenes generadas por checkpoint determina el máximo viable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# CARGA DE PROMPTS (una sola vez, se reutiliza en todos los checkpoints)
# ----------------------------------------------------------------------

prompts_df = pd.read_csv(prompts_file, header=None)
prompt_texts = prompts_df[0].tolist()

# ----------------------------------------------------------------------
# MODELOS (se cargan una sola vez)
# ----------------------------------------------------------------------

print("Cargando CLIP...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# ----------------------------------------------------------------------
# DATASET PARA KID
# ----------------------------------------------------------------------

kid_transform = transforms.Compose([
    transforms.Resize(kid_image_size),
    transforms.CenterCrop(kid_image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).clamp(0, 255).byte()),
])


class ImageDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        return kid_transform(img)


def find_image_path(folder, prompt, idx):
    """
    Replica la lógica del script viejo: el nombre de archivo se generó como
    f"{safe_prompt}-{i}.png", probando con y sin espacio antes del guion
    por si el prompt original lo llevaba.
    """
    candidate_1 = os.path.join(folder, f"{prompt}-{idx}.png")
    candidate_2 = os.path.join(folder, f"{prompt} -{idx}.png")

    if os.path.isfile(candidate_1):
        return candidate_1
    if os.path.isfile(candidate_2):
        return candidate_2
    return None


# ----------------------------------------------------------------------
# CARGAR EL DATASET REAL UNA SOLA VEZ (no cambia entre checkpoints)
# ----------------------------------------------------------------------

print(f"Cargando dataset real de: {real_images_dir}")
real_loader = DataLoader(
    ImageDataset(real_images_dir),
    batch_size=kid_batch_size,
    shuffle=False,
)

# ----------------------------------------------------------------------
# BUCLE PRINCIPAL: entrenamientos -> checkpoints
# ----------------------------------------------------------------------

results = []

for train_name in train_names:
    for ckpt in checkpoints:

        img_dir = os.path.join(images_root, train_name, f"checkpoint-{ckpt}")

        if not os.path.isdir(img_dir):
            print(f"[WARNING] No existe {img_dir}, se omite.")
            continue

        print(f"\n=== Entrenamiento: {train_name} | Checkpoint: {ckpt} ===")

        # ---------------- CLIP-T ----------------
        clip_t_scores = []
        missing = 0

        for i, prompt in enumerate(prompt_texts):
            img_path = find_image_path(img_dir, prompt, i)

            if img_path is None:
                missing += 1
                continue

            score = func_clip_T_score(img_path, prompt, clip_model, clip_preprocess, device)
            clip_t_scores.append(score)

        if missing:
            print(f"[WARNING] {missing} imagen(es) no encontradas para {train_name}/checkpoint-{ckpt}")

        clip_t_mean = sum(clip_t_scores) / len(clip_t_scores) if clip_t_scores else float("nan")

        # ---------------- KID ----------------
        fake_dataset = ImageDataset(img_dir)
        n_fake = len(fake_dataset)

        # subset_size no puede superar el nº de imágenes disponibles
        effective_subset_size = min(kid_subset_size, n_fake)

        kid = KernelInceptionDistance(
            subsets=50,
            subset_size=effective_subset_size,
            normalize=False,
            reset_real_features=True,
        ).to(device)

        for batch in tqdm(real_loader, desc="  Procesando reales", leave=False):
            kid.update(batch.to(device), real=True)

        fake_loader = DataLoader(fake_dataset, batch_size=kid_batch_size, shuffle=False)
        for batch in tqdm(fake_loader, desc="  Procesando fakes", leave=False):
            kid.update(batch.to(device), real=False)

        kid_mean, kid_std = kid.compute()
        kid_mean = kid_mean.item()
        kid_std = kid_std.item()

        print(f"  CLIP-T: {clip_t_mean:.4f} | KID: {kid_mean:.6f} ± {kid_std:.6f}")

        results.append({
            "train_name": train_name,
            "checkpoint": ckpt,
            "n_images": n_fake,
            "CLIP_T_score": clip_t_mean,
            "KID_mean": kid_mean,
            "KID_std": kid_std,
        })

        # Liberar memoria GPU entre checkpoints
        del kid
        torch.cuda.empty_cache()

# ----------------------------------------------------------------------
# GUARDAR CSV FINAL
# ----------------------------------------------------------------------

fieldnames = ["train_name", "checkpoint", "n_images", "CLIP_T_score", "KID_mean", "KID_std"]

with open(output_csv, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResultados guardados en {output_csv}")
