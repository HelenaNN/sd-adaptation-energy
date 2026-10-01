# Image Quality and Alignment Metrics

This directory contains scripts and utility functions to evaluate the visual quality and text-prompt alignment of images generated across training checkpoints for **LoRA**, **Full Fine-Tuning**, and **Textual Inversion**.

The primary metrics evaluated are **CLIP-T Score** (text-to-image semantic alignment) and **Kernel Inception Distance (KID)** (distributional similarity against a reference dataset of real images).

---

## Directory Overview

```text
metrics/
├── scores_calidad.py               # Shared metric utility library (CLIP-T, CLIP-I, DINO, FID, KID)
├── scores_calidad_lora.py          # Batch metric evaluator for LoRA checkpoint outputs
├── scores_calidad_finetuning.py    # Batch metric evaluator for Full Fine-Tuning outputs
└── scores_calidad_TI.py            # Batch metric evaluator for Textual Inversion step outputs

```

---

## Evaluated Metrics

### 1. CLIP-T Score (Text Alignment)

* Measures the cosine similarity between the visual embedding of a generated image and the textual embedding of the prompt used to produce it.
* Evaluated individually for every prompt-image pair and averaged per checkpoint.
* Implemented using OpenAI's `ViT-B/32` CLIP backbone.



### 2. Kernel Inception Distance (KID)

* Compares the feature distribution of generated images against a reference dataset of ground-truth style images.


* Unlike Fréchet Inception Distance (FID), KID is unbiased and remains statistically reliable with smaller sample sizes (e.g., 50 images per checkpoint).


* Computes polynomial kernel distances across multiple randomly sampled subsets (`subsets=50`), outputting both the mean KID and standard deviation (`KID_mean` and `KID_std`).



### Additional Utilities in `scores_calidad.py`

The utility file also provides standalone implementations for:

* **CLIP-I Score:** Cosine similarity between generated images and reference images.
* **DINO Score:** Self-supervised feature similarity using Vision Transformers.
* **FID (Fréchet Inception Distance):** Standard metric for large distribution comparison.



---

## Output CSV Format

Each script aggregates measurements across all evaluated runs and step intervals into a structured CSV file (`scores_<technique>.csv`):

| Column | Description |
| --- | --- |
| `train_name` | Experiment directory identifier |
| `checkpoint` | Evaluated step or checkpoint index |
| `n_images` | Number of generated images evaluated for this step |
| `CLIP_T_score` | Mean text-image alignment score across evaluated prompts |
| `KID_mean` | Mean Kernel Inception Distance relative to the real dataset |
| `KID_std` | Standard deviation of the KID calculation across subsets |

---

## Configuration & Usage

Configure your paths and targets directly at the top of each script:

```python
IMAGES_ROOT = "generated_images/lora"         # Generated samples from inference scripts[cite: 22]
REAL_IMAGES_DIR = "datasets/reference_images" # Ground-truth reference images[cite: 22]
PROMPTS_FILE = "piranesi_prompts.csv"         # Target prompt list[cite: 22]
OUTPUT_CSV = "scores_lora.csv"                # Output summary table[cite: 22]
TRAIN_NAMES = ["experiment_lora_run_1"]       # Folders to process[cite: 22]
CHECKPOINTS = [500, 1000, 1500, ..., 5000]    # Steps to evaluate[cite: 22]

```

### Running the Evaluations

```bash
python scores_calidad_lora.py
python scores_calidad_finetuning.py
python scores_calidad_TI.py

```
