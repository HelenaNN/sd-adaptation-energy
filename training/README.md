# Training

This directory contains the code and automation tools used to train and monitor Stable Diffusion models across three adaptation methods: **LoRA**, **Full Fine-Tuning**, and **Textual Inversion**.

The contents are organized into two main folders:

```text
training/
├── core/         # Core training scripts with integrated monitoring
└── launchers/    # Automation scripts to launch and monitor experiment batches

```

---

## Folder Overview

### [`core/`](https://www.google.com/search?q=./core&utm_source=gemini)

Contains the actual training scripts based on Hugging Face Diffusers examples. These scripts implement the training loops along with built-in instrumentation to record:

* Energy consumption and carbon emissions via CodeCarbon (`emissions.csv`).
* GPU metrics sampled during training via `nvidia-smi` (`gpu_metrics.csv`).
* Loss progression and learning rate updates (`training_metrics.csv`).
* Energy overhead measured specifically during checkpoint saves (`energy_per_checkpoint.csv`).



For details on the modifications made to the original Diffusers scripts and the generated CSV schemas, refer to the [core README](https://www.google.com/search?q=./core/README.md&utm_source=gemini).

### [`launchers/`](https://www.google.com/search?q=./launchers&utm_source=gemini)

Contains Python orchestrator scripts designed to run experiment queues automatically. Each launcher:

* Defines model paths, dataset locations, and hyperparameters (`batch_size`, `resolution`, `learning_rate`, etc.).
* Saves an experiment snapshot (`config.yaml`) for each run.
* Executes the underlying training script using `accelerate launch` while streaming outputs to a `train.log` file.
* Records global GPU usage across different run phases (pre-training, active training, post-training, failure, and cooldown).



For instructions on configuring queues, setting paths, and adapting scripts to your hardware, refer to the [launchers README](https://www.google.com/search?q=./launchers/README.md&utm_source=gemini).
