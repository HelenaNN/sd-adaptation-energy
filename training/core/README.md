# Modified Diffusers Training Scripts

This folder contains modified training scripts based on the official [examples provided by Hugging Face Diffusers](https://github.com/huggingface/diffusers/tree/main/examples?utm_source=gemini).

The scripts are used to adapt Stable Diffusion models across three distinct approaches:

* **LoRA (Low-Rank Adaptation)**
* **Full Fine-Tuning**
* **Textual Inversion**

While the core training pipelines and adaptation logic from Hugging Face Diffusers remain intact, additional functionality has been integrated across all three scripts to monitor hardware usage, power consumption, and environmental impact during training.

The original Hugging Face copyright notices and Apache 2.0 licenses are preserved in all script headers.

---

## Key Modifications

All scripts include a shared monitoring setup covering energy consumption, hardware metrics, training logs, and isolated measurements during checkpoint saves.

### 1. Energy and Emissions Monitoring (`emissions.csv`)

[CodeCarbon](https://github.com/mlco2/codecarbon?utm_source=gemini) is integrated to measure electrical consumption and estimated CO₂ equivalent emissions:

* **Process-Level Tracking & Safety:** Configured with `tracking_mode="process"` and instantiated strictly on the local main process (`accelerator.is_local_main_process`) to prevent duplicate writes and race conditions during multi-GPU runs.


* **Sampling & Flushing:** Configured with a 5-second sampling interval (`measure_power_secs=5`). It flushes data to disk every 10 seconds during optimization steps.


* **Fault-Tolerant Shutdown:** Wrapped inside a `try ... finally` block, ensuring that `codecarbon_tracker.stop()` is executed even if training is interrupted or fails with an error.



### 2. GPU Monitoring via `nvidia-smi` (`gpu_metrics.csv`)

Hardware performance is polled every 10 seconds using `nvidia-smi` and logged with the following fields:

| Column | Description |
| --- | --- |
| `timestamp` | Epoch Unix timestamp of the measurement |
| `global_step` | Current training optimization step |
| `temperature_gpu` | GPU core temperature (°C) |
| `utilization_gpu` | GPU compute utilization percentage (%) |
| `memory_used_mb` | Allocated VRAM (MB) |
| `power_draw_w` | Instantaneous power consumption (Watts) |

### 3. Standalone Training Metrics (`training_metrics.csv`)

To allow direct analysis without relying on TensorBoard or Weights & Biases, loss values and learning rates are saved directly to a CSV file:

```text
global_step,epoch,train_loss,step_loss,lr

```

* `train_loss`: Accumulated training loss averaged across processes.
* `step_loss`: Loss value computed on the current step.
* `lr`: Scheduled learning rate at that specific step.



### 4. Isolated Checkpoint Energy & Duration Measurement (`energy_per_checkpoint.csv`)

Saving model weights and optimizer states introduces non-negligible I/O and compute overhead. To separate this cost from actual training steps:

1. A dedicated sub-task is started right before saving via `codecarbon_tracker.start_task("checkpoint")`.
2. The task covers directory cleanup (`--checkpoints_total_limit`), state serialization (`accelerator.save_state`), and weight conversion/saving.
3. Once finished, `stop_task("checkpoint")` isolates and logs the energy consumed, emissions, and elapsed time strictly during the saving process.



**Recorded schema:**

```text
timestamp,global_step,energy_consumed_kwh,emissions_kg,total_energy_kwh,checkpoint_duration_s

```

* `checkpoint_duration_s`: Total duration in seconds spent exclusively on the checkpoint saving task (`ckpt_task_emissions.duration`).



### 5. CodeCarbon Compatibility Workaround

A fix was added for an issue observed in **CodeCarbon 3.2.8**, where calling `flush()` after completing sub-tasks crashes when `experiment_name=None`.

After stopping the task, the internal task registry is cleared:

```python
codecarbon_tracker._tasks.clear()

```

> **Note:** This workaround relies on internal attributes of CodeCarbon (`_tasks`) and may not be required in other releases of the library.
> 
> 

---

## Output Directory Structure

Each run saves the following files inside its `--output_dir`:

```text
output_dir/
├── emissions.csv               # Continuous emissions log from CodeCarbon
├── energy_per_checkpoint.csv   # Isolated energy cost and duration per checkpoint creation
├── gpu_metrics.csv             # GPU temperature, power, and utilization over time
├── training_metrics.csv        # Step-by-step loss and learning rate history
├── checkpoint-<step>/          # Periodic Accelerate checkpoints
└── pytorch_lora_weights.safetensors (or model weights depending on the method)

```

### Generated Files Summary

| File | Primary Use Case |
| --- | --- |
| `emissions.csv` | Cumulative energy footprint and emissions calculation |
| `gpu_metrics.csv` | Hardware load, power draw, and thermal throttling analysis |
| `training_metrics.csv` | Loss curves and learning rate progression |
| `energy_per_checkpoint.csv` | Benchmarking the energy overhead and execution time of saving checkpoints |

---

## Requirements

Install the additional dependency in your Python environment alongside Diffusers and Accelerate:

```bash
pip install codecarbon

```

> **Prerequisite:** Reading GPU metrics requires an NVIDIA driver with the `nvidia-smi` command available in your system's `PATH`.
> 
> 

---

## Training Methods

The scripts in this directory cover three common fine-tuning approaches:

1. **LoRA:** Injects low-rank decomposition matrices into the cross-attention layers of the UNet, training a minimal set of parameters while keeping the base model frozen.
2. **Full Fine-Tuning:** Directly updates the weights of the UNet/text encoders.
3. **Textual Inversion:** Freezes all model weights and optimizes new token embeddings inside the text encoder dictionary.



Regardless of which weights are updated, all three scripts use the exact same logging intervals, file formats, and monitoring logic.
