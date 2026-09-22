# Modified Diffusers Training Scripts

This folder contains modified training scripts based on the training examples provided by Hugging Face Diffusers.

The scripts are used to adapt Stable Diffusion models using three different approaches:

- **LoRA**
- **Fine-tuning**
- **Textual Inversion**

The original Diffusers training procedures have been kept as the basis of the implementations. Additional functionality has been added to monitor the computational and environmental characteristics of each training execution.

The original Hugging Face copyright and Apache 2.0 license notice are retained in the scripts.

## Modifications

The main modifications introduced for this project are related to energy monitoring, GPU monitoring, training metrics, and checkpoint analysis.

### Energy and emissions monitoring

[CodeCarbon](https://github.com/mlco2/codecarbon) is used to measure the energy consumed and associated emissions during training.

The tracker is created only by the local main process to avoid duplicated measurements when using distributed training.

The measurements are stored in:

```text
emissions.csv


The tracker is started at the beginning of training and periodically flushed every 10 seconds. It is stopped when the training execution finishes, including when an exception occurs.

The tracker is configured with a 5-second power measurement interval.

### GPU monitoring

GPU information is periodically collected using nvidia-smi.

The following metrics are recorded every 10 seconds:

```text
timestamp
global_step
temperature_gpu
utilization_gpu
memory_used_mb
power_draw_w


They are stored in:

gpu_metrics.csv

This provides information about GPU utilisation and power behaviour during the training execution.

### Training metrics

An additional CSV file is generated to store the main training metrics independently of TensorBoard or other logging systems:

training_metrics.csv

It contains:

global_step,epoch,train_loss,step_loss,lr

This allows the training evolution to be analysed together with the energy and GPU measurements.

### Checkpoint energy measurement

Checkpoint creation is measured separately from the rest of the training execution.

Before a checkpoint is saved, a CodeCarbon task named checkpoint is started. After the checkpoint has been completely saved, the task is stopped and its measurements are stored in:

energy_per_checkpoint.csv

The file contains:

timestamp,global_step,energy_consumed_kwh,emissions_kg,total_energy_kwh

This makes it possible to analyse the energy and emissions associated specifically with checkpoint creation.

The measured checkpoint operation includes the checkpoint management and saving operations performed between start_task() and stop_task(), including saving the training state and the model parameters.

### CodeCarbon compatibility workaround

A workaround was added for an issue observed with CodeCarbon 3.2.8.

After a checkpoint task is stopped, the internal task collection is cleared before subsequent CodeCarbon flushes:

codecarbon_tracker._tasks.clear()

This prevents an error observed when flush() is called while completed tasks remain in the internal task collection.

This workaround depends on CodeCarbon’s internal implementation and may need to be reviewed if a different CodeCarbon version is used.

## Output files

A typical training output directory can contain:

output_dir/
├── emissions.csv
├── gpu_metrics.csv
├── training_metrics.csv
├── energy_per_checkpoint.csv
├── model weights
└── checkpoint-<step>/

The four CSV files provide complementary information:

File	Description
emissions.csv	Energy consumption and emissions measured by CodeCarbon
gpu_metrics.csv	GPU temperature, utilisation, memory usage and power draw
training_metrics.csv	Training loss and learning-rate evolution
energy_per_checkpoint.csv	Energy and emissions measured for checkpoint creation

## Training methods

The folder contains scripts corresponding to the following adaptation methods:

LoRA: trains low-rank adapter parameters while keeping the original model parameters frozen.

Fine-tuning: updates the selected Stable Diffusion model parameters directly during training.

Textual Inversion: learns new textual embeddings while keeping the pretrained model weights fixed.

Although the parameters being trained differ between methods, the additional monitoring functionality follows the same purpose across the three implementations: collecting training, GPU, energy, emissions, and checkpoint-related information for subsequent analysis.

### Relation to the original Diffusers scripts

The training logic and model adaptation procedures are based on the corresponding Hugging Face Diffusers examples.

The project-specific additions are primarily:

* CodeCarbon energy and emissions tracking.
* Periodic GPU monitoring through nvidia-smi.
* CSV logging of training metrics.
* Separate checkpoint energy measurements.
* The CodeCarbon compatibility workaround described above.

The scripts should therefore be considered modified versions of the original Diffusers examples rather than completely independent implementations.

For the original implementation and licensing information, refer to the Hugging Face Diffusers repository.
