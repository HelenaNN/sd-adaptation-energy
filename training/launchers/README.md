# Experiment Launchers

This directory contains automation scripts to run batches of training experiments across three adaptation methods: **LoRA**, **Full Fine-Tuning**, and **Textual Inversion**.

The scripts manage the complete lifecycle of each experiment run, including training invocation via Hugging Face Accelerate, configuration serialization, continuous GPU monitoring across different training phases, and cooldown periods between runs.

---

## Launcher Structure & Logic

All launchers implement the exact same core pipeline and lifecycle:

1. **Global Configuration:** Defines base model paths, dataset root folders, base output directories, and shared hyperparameters (`RESOLUTION`, `BATCH_SIZE`, `LEARNING_RATE`, `CHECKP_STEPS`, etc.).


2. **Experiment Queue:** An ordered list of runs defining per-experiment variables (dataset name, maximum training steps, and method-specific arguments such as tokens or validation prompts).


3. **Reproducibility Tracking:** Before each run launches, its full configuration is exported into a `config.yaml` file inside the corresponding experiment directory.


4. **Execution & Log Capture:** The underlying training script is launched using `accelerate launch`, streaming stdout and stderr to both the terminal and an experiment-specific `train.log` file.


5. **GPU Monitoring Across Lifecycle:** GPU stats are recorded via `nvidia-smi` at discrete points (`before_training`, `after_training`, `training_failed`, `cooldown`, `after_cooldown`), alongside an asynchronous background process that samples continuous metrics during active training.


6. **Thermal Cooldown:** An optional cooldown period (e.g., 600 seconds) is enforced between consecutive experiments to return the hardware to an idle baseline state while continuing to log GPU metrics.



---

## Single Script vs. Hardware-Specific Variants

In this repository, separate script files are provided for different GPUs (e.g., NVIDIA A40 vs. NVIDIA TITAN RTX). These variants exist primarily due to:

* Target output and dataset directory paths on different machines.


* Different batch sizes adjusted to available VRAM (e.g., batch size 24 on A40 vs. 12 on TITAN for LoRA).


* Explicit device pinning when running alongside other workloads (e.g., `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`).



**Using a single launcher:** It is not strictly necessary to keep separate launcher files for different GPUs. Because all launchers share identical structure, you can maintain a single script and adjust paths, `BATCH_SIZE`, or GPU device IDs directly at the top of the file to fit your target setup.

---

## Script Inventory

| Script | Technique | Default Target GPU / Environment | Key Parameters |
| --- | --- | --- | --- |
| `sdv15_lora_a40-2.py`<br> | LoRA

 | NVIDIA A40

 | Batch size: 24, Rank: 8, LR: 1e-4

 |
| `sdv15_lora-2.py`<br> | LoRA

 | NVIDIA TITAN

 | Batch size: 12, Rank: 8, LR: 1e-4

 |
| `sdv15_finetuning_a40-2.py`<br> | Full Fine-Tuning

 | NVIDIA A40

 | Batch size: 6, LR: 5e-6

 |
| `sdv15_finetuning.py`<br> | Full Fine-Tuning

 | NVIDIA TITAN

 | Batch size: 6, LR: 5e-6

 |
| `sdv15_textual_inversion_a40-2.py`<br> | Textual Inversion

 | NVIDIA A40

 | Batch size: 6, LR: 5e-4

 |
| `sdv15_textual_inversion-2.py`<br> | Textual Inversion

 | NVIDIA TITAN

 | Batch size: 12, LR: 5e-4

 |

---

## Generated Metrics & Output Files

Each execution run produces two levels of outputs: global experiment batch files and isolated run directories.

### 1. Global GPU Log (`global_gpu_metrics_<variant>.csv`)

Created at the root of `BASE_OUTPUT_DIR` to record GPU behavior throughout the entire queue:

```text
timestamp,datetime,phase,experiment_name,dataset,steps,temperature_gpu,utilization_gpu,vram_used_mb,power_draw_w

```

The `phase` field categorizes the hardware context:

* `before_training`: Initial idle reading right before launching the job.


* `training`: Periodic readings sampled by the background monitor during script execution.


* `training_failed`: Logged if the training script terminates with a non-zero exit status.


* `after_training`: Immediate post-training reading.


* `cooldown`: Sampled every 10 seconds during the post-run idle period.


* `after_cooldown`: Final reading at the end of the cooldown interval.



### 2. Experiment Directory

Each entry in `EXPERIMENTS` creates an isolated subfolder named by index, model, technique, resolution, dataset, steps, and seed:

```text
outputs_<variant>/
├── global_gpu_metrics_<variant>.csv
└── <id>_sdv15_<technique>_res512_<dataset>_steps5000_seed1337/
    ├── config.yaml                     # Snapshot of all launcher-level parameters
    ├── train.log                       # Full terminal stdout/stderr stream
    ├── emissions.csv                   # CodeCarbon emissions log (from training script)
    ├── energy_per_checkpoint.csv       # Isolated checkpoint saving energy (from training script)
    ├── gpu_metrics.csv                 # In-training GPU log (from training script)
    ├── training_metrics.csv            # Loss and learning rate log (from training script)
    └── checkpoint-<step>/              # Saved model checkpoints

```

---

## How to Customize and Run

1. **Verify paths and parameters:** Open the selected launcher and ensure the paths to your model, datasets, and target output directory match your environment:


```python
MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5" #[cite: 1, 2, 4]
DATASETS_BASE = "/path/to/your/datasets" #[cite: 1, 2, 4]
BASE_OUTPUT_DIR = Path("/path/to/your/output_dir") #[cite: 1, 2, 4]

```


2. **Define your experiment sequence:** Add or remove items in the `EXPERIMENTS` list:


```python
EXPERIMENTS = [
    ("Battista", 5000, "A drawing of a tower."), #[cite: 2, 4]
    ("VanGogh",  3000, "A painting of a field."), #[cite: 4]
]

```


3. **Execute:**
```bash
python sdv15_lora_a40.py

```
