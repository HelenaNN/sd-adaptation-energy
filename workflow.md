# Workflow — Step-by-Step Pipeline

This document describes the full pipeline from training to final plots.
Follow the steps in order. Each step depends on the outputs of the previous one.

---

## Step 0 — Environment Setup

Follow [`setup.md`](setup.md) to create and activate the conda environment.

```bash
conda activate sd-diffusers
```

---

## Step 1 — Training

Launch the training experiments using the launcher scripts in `training/launchers/`.
Each launcher runs 3 repetitions (seeds 0, 1, 2) of the same configuration.

```bash
# Example: LoRA on Titan RTX
CUDA_VISIBLE_DEVICES=0 python training/launchers/lora_titan.py

# Example: Fine-tuning on A40
CUDA_VISIBLE_DEVICES=0 python training/launchers/finetuning_a40.py

# Example: Textual Inversion on Titan RTX
CUDA_VISIBLE_DEVICES=0 python training/launchers/ti_titan.py
```

**Outputs per experiment folder:**
- `energy_per_checkpoint.csv` — energy consumed per checkpoint save
- `emissions.csv` — CodeCarbon full energy log
- `checkpoint-N/` — model weights at each checkpoint
- `train.log` — training log

---

## Step 2 — Checkpoint Duration Calibration

Before cleaning energy and duration, you need to calibrate the checkpoint-saving duration.
This runs short adaptation trainings (300 steps) to measure how long each checkpoint save takes.

```bash
# Run calibration for each technique and GPU
CUDA_VISIBLE_DEVICES=0 python energy/calibration/calibration_lora_titan.py
CUDA_VISIBLE_DEVICES=0 python energy/calibration/calibration_lora_a40.py
CUDA_VISIBLE_DEVICES=0 python energy/calibration/calibration_finetuning_titan.py
CUDA_VISIBLE_DEVICES=0 python energy/calibration/calibration_finetuning_a40.py
CUDA_VISIBLE_DEVICES=0 python energy/calibration/calibration_ti_titan.py
CUDA_VISIBLE_DEVICES=0 python energy/calibration/calibration_ti_a40.py

# Compute mean calibration duration
python energy/calibration/compute_calibration_duration.py
```

**Output:** `checkpoint_duration_calibration.csv`

---

## Step 3 — Clean Energy and Duration

Remove checkpoint-saving overhead from the raw energy and duration measurements.

```bash
python energy/clean_energy_data.py
python energy/clean_duration_data.py
```

**Output per experiment:** 
- `energy_clean_per_checkpoint.csv`
- `duration_clean_per_checkpoint.csv`

---

## Step 4 — Copy Results to Analysis Folder

Organise the cleaned files into the analysis directory.

```bash
python energy/copy_to_results.py
```

---

## Step 5 — Image Generation

Generate images for quality evaluation. Run separately for each technique.

```bash
# Style personalisation
python inference/inference_lora.py
python inference/inference_ti.py

# Visual evolution grids
python inference/evolution/evolution_lora.py
python inference/evolution/evolution_ti.py

# Subject personalisation (Textual Inversion)
python inference/inference_subject_reconstruction.py
python inference/inference_subject_editability.py
```

---

## Step 6 — Quality Metrics

Compute CLIP-T, KID, and subject-specific metrics on the generated images.

```bash
# Style metrics (CLIP-T + KID)
python metrics/scores_lora.py
python metrics/scores_finetuning.py
python metrics/scores_ti.py

# Subject metrics (reconstruction + editability)
python metrics/subject/scores_reconstruction.py
python metrics/subject/scores_editability.py
```

**Output per technique:** `scores_*.csv`

---

## Step 7 — Visualisation

Generate all plots. Switch to the scores/analysis environment if needed.

```bash
conda activate scores
```

### Energy plots

```bash
python plots/energy/plot_lineplot_gpu.py
python plots/energy/plot_lineplot_batch_color.py
python plots/energy/plot_barplots_comparison.py
python plots/energy/plot_barplots_scenarios.py
python plots/energy/plot_barplot_optimal.py
```

### Quality plots

```bash
python plots/quality/plot_scores_energy.py
python plots/quality/plot_bubble_scores.py
python plots/quality/plot_errorbar_scores.py
python plots/quality/plot_scatter_quality.py
```

---

## Summary of Outputs

| Step | Output |
|---|---|
| Training | Checkpoints, `energy_per_checkpoint.csv`, `emissions.csv` |
| Calibration | `checkpoint_duration_calibration.csv` |
| Cleaning | `energy_clean_per_checkpoint.csv`, `duration_clean_per_checkpoint.csv` |
| Inference | Generated images per checkpoint |
| Metrics | `scores_*.csv` |
| Plots | `.png` and `.svg` figures |

---

## Notes

- All training scripts use `CUDA_VISIBLE_DEVICES=0` to run on a single GPU.
- The two conda environments serve different purposes:
  - `sd-diffusers` — training and inference
  - `scores` — metric computation and plotting
- Checkpoint-saving energy overhead is subtracted automatically in Step 3.
- See each folder's `README.md` for configuration details (paths, batch sizes, etc.).
