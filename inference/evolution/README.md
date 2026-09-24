# Training Evolution and Progression Analysis

This directory contains evaluation tools specifically built to inspect how style, structure, and image quality evolve across optimization steps during training for **LoRA**, **Full Fine-Tuning**, and **Textual Inversion**.

Unlike general inference sampling, these scripts lock the generation seed and generate horizontal side-by-side progression grids (stripes) starting with the unadapted base model as step 0, followed by every evaluated checkpoint.

---

## Directory Overview

```text
inference/evolution/
├── evolution_lora.py         # Progression strips for LoRA checkpoints
├── evolution_finetuning.py   # Progression strips for Full Fine-Tuning checkpoints
└── evolution_TI.py           # Progression strips for Textual Inversion embeddings

```

---

## How It Works

Each script uses a fixed random seed (`SEED = 1337`) to keep the initial latent noise identical across all checkpoints. This ensures that changes observed from left to right reflect the model weights learning the style rather than random diffusion variations.

### Execution Flow

1. **Prompt Isolation:** Loops through selected target prompts defined in `PROMPTS`.


2. **SD Base Baseline:** Generates a reference image using the original pretrained Stable Diffusion v1.5 pipeline before applying any checkpoint weights (labeled `SD base`).


3. **Sequential Loading:** Iterates over step numbers in `CHECKPOINTS`:


* **LoRA:** Loads adapter layers via `pipe.load_lora_weights()`, generates the sample, and immediately unloads them with `pipe.unload_lora_weights()`.


* **Full Fine-Tuning:** Swaps only the trained `UNet2DConditionModel` from each checkpoint folder into the pipeline.


* **Textual Inversion:** Loads the step embedding via `pipe.load_textual_inversion()`, appends the placeholder token to the prompt, and unloads it with `pipe.unload_textual_inversion()`.




4. **Grid Composition:** Uses PIL to assemble all step outputs into a single labeled horizontal image (`grid_<experiment_label>.png`).



---

## Output Structure

Results are written to an isolated directory per prompt, containing both individual checkpoint images and composite comparison grids:

```text
results/evolution_<technique>/
└── <prompt_name>/
    ├── <experiment_label>_base.png
    ├── <experiment_label>_step500.png
    ├── <experiment_label>_step1000.png
    ├── ...
    ├── <experiment_label>_step5000.png
    └── grid_<experiment_label>.png

```

---

## Configuration & Usage

Configure your experiment targets and checkpoints at the top of each file:

| Setting | Description |
| --- | --- |
| `TRAININGS` | Dictionary mapping custom display labels to output directories |
| `CHECKPOINTS` | List of step numbers to plot along the horizontal axis |
| `PROMPTS` | Curated prompts to test stylistic acquisition across domains |
| `SEED` | Fixed integer seed ensuring identical initial latent noise across steps |
| `PLACEHOLDER_TOKEN` *(TI only)* | Special learned token added to prompts during generation |

### Running the Scripts

```bash
python evolution_lora.py
python evolution_finetuning.py
python evolution_TI.py

```
