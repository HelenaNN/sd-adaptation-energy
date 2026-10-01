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

# ── General Configuration ─────────────────────────────────────────────────────
IMAGES_ROOT = "generated_images/finetuning"
REAL_IMAGES_DIR = "datasets/reference_images"
PROMPTS_FILE = "piranesi_prompts.csv"
OUTPUT_CSV = "scores_finetuning.csv"

TRAIN_NAMES = [
    "experiment_finetuning_run_1",
]

CHECKPOINTS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

KID_IMAGE_SIZE = (299, 299)
KID_BATCH_SIZE = 32
KID_SUBSET_SIZE = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load Prompts & CLIP ───────────────────────────────────────────────────────
prompts_df = pd.read_csv(PROMPTS_FILE, header=None)
prompt_texts = prompts_df[0].tolist()

print("Loading CLIP (ViT-B/32)...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

# ── Dataset Definition ────────────────────────────────────────────────────────
kid_transform = transforms.Compose([
    transforms.Resize(KID_IMAGE_SIZE),
    transforms.CenterCrop(KID_IMAGE_SIZE),
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
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:50].strip()
    candidate_1 = os.path.join(folder, f"{safe}-{idx}.png")
    candidate_2 = os.path.join(folder, f"{safe} -{idx}.png")

    if os.path.isfile(candidate_1):
        return candidate_1
    if os.path.isfile(candidate_2):
        return candidate_2
    return None


# ── Load Real Dataset Once ────────────────────────────────────────────────────
print(f"Loading reference dataset from: {REAL_IMAGES_DIR}")
real_loader = DataLoader(
    ImageDataset(REAL_IMAGES_DIR),
    batch_size=KID_BATCH_SIZE,
    shuffle=False,
)

# ── Main Metric Computation Loop ──────────────────────────────────────────────
results = []

for train_name in TRAIN_NAMES:
    for ckpt in CHECKPOINTS:
        img_dir = os.path.join(IMAGES_ROOT, train_name, f"checkpoint-{ckpt}")

        if not os.path.isdir(img_dir):
            print(f"[SKIP] Directory not found: {img_dir}")
            continue

        print(f"\n=== Evaluating: {train_name} | Checkpoint: {ckpt} ===")

        # 1. CLIP-T Score
        clip_t_scores = []
        missing = 0

        for i, prompt in enumerate(prompt_texts):
            img_path = find_image_path(img_dir, prompt, i)
            if img_path is None:
                missing += 1
                continue

            score = func_clip_T_score(img_path, prompt, clip_model, clip_preprocess, DEVICE)
            clip_t_scores.append(score)

        if missing:
            print(f"  [WARNING] {missing} image(s) not matched for checkpoint-{ckpt}")

        clip_t_mean = sum(clip_t_scores) / len(clip_t_scores) if clip_t_scores else float("nan")

        # 2. Kernel Inception Distance (KID)
        fake_dataset = ImageDataset(img_dir)
        n_fake = len(fake_dataset)

        if n_fake == 0:
            print(f"  [WARNING] No generated images found in {img_dir}. Skipping KID.")
            continue

        effective_subset_size = min(KID_SUBSET_SIZE, n_fake)

        kid = KernelInceptionDistance(
            subsets=50,
            subset_size=effective_subset_size,
            normalize=False,
            reset_real_features=True,
        ).to(DEVICE)

        for batch in tqdm(real_loader, desc="  Feeding real distribution", leave=False):
            kid.update(batch.to(DEVICE), real=True)

        fake_loader = DataLoader(fake_dataset, batch_size=KID_BATCH_SIZE, shuffle=False)
        for batch in tqdm(fake_loader, desc="  Feeding generated distribution", leave=False):
            kid.update(batch.to(DEVICE), real=False)

        kid_mean, kid_std = kid.compute()
        kid_mean = kid_mean.item()
        kid_std = kid_std.item()

        print(f"  CLIP-T: {clip_t_mean:.4f} | KID: {kid_mean:.6f} +/- {kid_std:.6f}")

        results.append({
            "train_name": train_name,
            "checkpoint": ckpt,
            "n_images": n_fake,
            "CLIP_T_score": clip_t_mean,
            "KID_mean": kid_mean,
            "KID_std": kid_std,
        })

        del kid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ── Export CSV Summary ────────────────────────────────────────────────────────
fieldnames = ["train_name", "checkpoint", "n_images", "CLIP_T_score", "KID_mean", "KID_std"]

with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults successfully exported to {OUTPUT_CSV}")
