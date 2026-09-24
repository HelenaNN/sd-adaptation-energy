# Checkpoint Calibration Scripts

This directory contains benchmarking scripts designed to measure the energy overhead and time consumption associated with saving model checkpoints across **LoRA**, **Full Fine-Tuning**, and **Textual Inversion**.

Saving full models, adapters, or token embeddings incurs distinct computational and disk write overheads. These calibration runs isolate those operations in short, controlled training jobs to obtain consistent baseline measurements.

---

## Directory Overview

```text
energy/calibration/
├── calibration_lora.py         # Calibration runner for LoRA adapters
├── calibration_finetuning.py   # Calibration runner for Full Fine-Tuning
└── calibration_ti.py           # Calibration runner for Textual Inversion

```

---

## Methodology & Calibration Design

Unlike full training runs (typically 3,000 to 5,000 steps), the calibration runs are tailored to test checkpoint creation frequently under identical conditions:

1. **Short Iteration Horizon:** `MAX_TRAIN_STEPS = 300` ensures runs finish quickly while generating multiple sample events.
2. **Dense Checkpoint Frequency:** `CHECKP_STEPS = 100` triggers multiple checkpoint saves (at step 100, 200, and 300) per run.
3. **Repeated Runs:** Each script executes 3 identical runs in sequence to verify measurement repeatability.
4. **Zero Cooldown:** `COOLDOWN_SECONDS = 0` avoids artificial idle delays between calibration iterations.
5. **Isolated Task Profiling:** The underlying training scripts use CodeCarbon sub-tasks (`start_task("checkpoint")` and `stop_task("checkpoint")`) to record disk write energy independently from gradient updates into `energy_per_checkpoint.csv`.



---

## Single Script vs. Hardware-Specific Variants

In previous iterations of the repository, separate scripts were used for different GPUs (e.g., NVIDIA A40 vs. TITAN RTX) to handle varying machine paths and CUDA device IDs.

The scripts in this directory have been unified. All three methods follow the same structure and can be adapted to any machine by modifying the parameters at the top of the file:

* `BASE_OUTPUT_DIR` and `DATASETS_BASE` for local paths.


* `GPU_DEVICE_ID` to target specific GPUs in multi-GPU machines.



---

## Output Metrics

Each execution creates an output folder structure containing:

```text
calibration_outputs/<technique>/
├── global_gpu_metrics_calibration_<technique>.csv   # Host-level GPU stats throughout the run
└── <run_id>_calibration_<technique>_res512_.../
    ├── config.yaml                                  # Snapshot with "purpose": "checkpoint_duration_calibration"
    ├── train.log                                    # Execution terminal output
    ├── emissions.csv                                # Global CodeCarbon emission log
    ├── energy_per_checkpoint.csv                    # Isolated energy per checkpoint event
    ├── gpu_metrics.csv                              # In-training GPU readings
    ├── training_metrics.csv                         # Loss metrics per step
    └── checkpoint-<step>/                           # Serialized checkpoint states

```

The resulting `energy_per_checkpoint.csv` files provide the core data needed to benchmark checkpoint costs across different model architectures and storage setups.

---

## Running Calibration

Run the desired calibration script directly:

```bash
python calibration_lora.py
python calibration_finetuning.py
python calibration_ti.py

```
