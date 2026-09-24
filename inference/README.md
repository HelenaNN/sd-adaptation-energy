# Inference and Evaluation Scripts

This directory contains evaluation scripts used to generate image samples across multiple intermediate checkpoints and trained runs for **LoRA**, **Full Fine-Tuning**, and **Textual Inversion**.

The scripts iterate through experiment directories, load the corresponding model state at each checkpoint step, and generate outputs using a shared list of prompts.

---

## Directory Overview

```text
inference/
├── prompts.csv             # Target prompts (one per line, headerless CSV)
├── inference_lora.py       # Checkpoint sampling for LoRA adapters
├── inference_finetune.py   # Checkpoint sampling for full fine-tuned models
└── inference_TI.py         # Step sampling for Textual Inversion embeddings

```

---

## How It Works

Each script follows a 3-tier nested loop structure:

1. **Experiment run:** Iterates over the folder names provided in `TRAIN_NAMES`.


2. **Checkpoint / Step:** Scans for existing intermediate weights in `CHECKPOINTS` / `STEPS_LIST`. Missing files are skipped automatically without interrupting execution.


3. **Prompt evaluation:** Generates images for each line in `prompts.csv` and saves them in structured folders:


```text
<OUTPUT_ROOT>/<experiment_name>/<checkpoint-step>/<sanitized_prompt>-<index>.png

```



### Implementation Specifics per Method

* **LoRA (`inference_lora.py`):** Loads the base model once per checkpoint, applies adapter weights via `pipe.load_lora_weights()`, and purges CUDA cache between checkpoints to prevent memory fragmentation.


* **Full Fine-Tuning (`inference_finetune.py`):** Loads the base pipeline (VAE, text encoder, scheduler) once per experiment, swapping only the `UNet2DConditionModel` at each checkpoint to reduce loading overhead.


* **Textual Inversion (`inference_TI.py`):** Keeps the base pipeline in memory and registers intermediate token embeddings via `pipe.load_textual_inversion()`. It calls `pipe.unload_textual_inversion()` after each step to prevent token leakage across steps.



---

## Configuration & Usage

All parameters are configured directly at the top of each script:

| Variable | Description |
| --- | --- |
| `BASE_DIR` | Directory where training outputs and checkpoints are stored

 |
| `OUTPUT_ROOT` | Destination directory for generated image samples

 |
| `PROMPTS_FILE` | Path to the `.csv` file containing text prompts

 |
| `TRAIN_NAMES` | List of experiment directory names to process

 |
| `CHECKPOINTS` / `STEPS_LIST` | Numerical step numbers to evaluate

 |
| `PLACEHOLDER_TOKEN` *(TI only)* | Special learned token to insert into prompts

 |

### Running the Scripts

Ensure your environment has access to a CUDA device:

```bash
python inference_lora.py
python inference_finetune.py
python inference_TI.py

```
