from pathlib import Path
import subprocess
import datetime
import yaml
import os
import shlex
import time
import signal

MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATASETS_BASE = "/mnt/rhome/hnn/datasets/wikiart"
BASE_OUTPUT_DIR = Path("/mnt/rhome/hnn/calibration_cp/sdv15/textual_inversion/calibration_a40")
GLOBAL_GPU_METRICS = BASE_OUTPUT_DIR / "global_gpu_metrics_calibration_a40.csv"

RESOLUTION = 512
BATCH_SIZE = 6
GRAD_ACCUM_STEPS = 1
CHECKP_STEPS = 100
MAX_TRAIN_STEPS = 300
SEED = 1337
LEARNING_RATE = "5e-4"
COOLDOWN_SECONDS = 0
GPU_LOG_INTERVAL_SECONDS = 10
LEARNABLE_PROPERTY = "style"

EXPERIMENTS = [
    ("Battista", "<piranesi-style>", "art", "A drawing of a tower."),
    ("Battista", "<piranesi-style>", "art", "A drawing of a tower."),
    ("Battista", "<piranesi-style>", "art", "A drawing of a tower."),
]

BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not GLOBAL_GPU_METRICS.exists():
    with open(GLOBAL_GPU_METRICS, "w") as f:
        f.write("timestamp,datetime,phase,experiment_name,dataset,steps,temperature_gpu,utilization_gpu,vram_used_mb,power_draw_w\n")


def log_global_gpu_metrics(phase, exp_name, dataset, steps):
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw",
         "--format=csv,noheader,nounits"], text=True).strip()
    temp, util, vram, power = [x.strip() for x in result.split(",")]
    with open(GLOBAL_GPU_METRICS, "a") as f:
        f.write(f"{time.time()},{datetime.datetime.now().isoformat()},{phase},{exp_name},{dataset},{steps},{temp},{util},{vram},{power}\n")


def start_global_gpu_monitor(exp_name, dataset, steps, interval_seconds=10):
    metrics_path = shlex.quote(str(GLOBAL_GPU_METRICS))
    monitor_command = f"""
    while true; do
        timestamp=$(date +%s.%N)
        datetime=$(date --iso-8601=seconds)
        gpu_data=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits)
        echo "$timestamp,$datetime,training,{shlex.quote(exp_name)},{shlex.quote(str(dataset))},{shlex.quote(str(steps))},$gpu_data" >> {metrics_path}
        sleep {interval_seconds}
    done
    """
    return subprocess.Popen(monitor_command, shell=True, executable="/bin/bash", preexec_fn=os.setsid)


def stop_global_gpu_monitor(process):
    if process is None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()


for i, (dataset_name, placeholder_token, initializer_token, validation_prompt) in enumerate(EXPERIMENTS):

    dataset_dir = f"{DATASETS_BASE}/textual_inversion_{RESOLUTION}_{dataset_name}_dataset"
    exp_name = f"{i}_calibration_ti_a40_res{RESOLUTION}_{dataset_name}_steps{MAX_TRAIN_STEPS}_seed{SEED}"
    output_dir = BASE_OUTPUT_DIR / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment_name": exp_name,
        "datetime": datetime.datetime.now().isoformat(),
        "base_model": MODEL_NAME,
        "dataset": dataset_dir,
        "technique": "Textual Inversion",
        "purpose": "checkpoint_duration_calibration",
        "learnable_property": LEARNABLE_PROPERTY,
        "placeholder_token": placeholder_token,
        "initializer_token": initializer_token,
        "resolution": RESOLUTION,
        "max_train_steps": MAX_TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "checkpointing_steps": CHECKP_STEPS,
        "seed": SEED,
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    command = [
        "accelerate", "launch", "textual_inversion_codecarbon.py",
        f"--pretrained_model_name_or_path={MODEL_NAME}",
        f"--train_data_dir={dataset_dir}",
        f"--learnable_property={LEARNABLE_PROPERTY}",
        f"--placeholder_token={placeholder_token}",
        f"--initializer_token={initializer_token}",
        f"--checkpointing_steps={CHECKP_STEPS}",
        f"--save_steps={CHECKP_STEPS}",
        f"--resolution={RESOLUTION}",
        f"--train_batch_size={BATCH_SIZE}",
        f"--gradient_accumulation_steps={GRAD_ACCUM_STEPS}",
        f"--max_train_steps={MAX_TRAIN_STEPS}",
        f"--learning_rate={LEARNING_RATE}",
        "--scale_lr",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--output_dir={output_dir}",
    ]

    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"

    log_path = output_dir / "train.log"
    command_str = " ".join(shlex.quote(arg) for arg in command)
    command_str = f"set -o pipefail; {command_str} 2>&1 | tee {shlex.quote(str(log_path))}"

    log_global_gpu_metrics("before_training", exp_name, dataset_name, MAX_TRAIN_STEPS)
    gpu_monitor = start_global_gpu_monitor(exp_name, dataset_name, MAX_TRAIN_STEPS, GPU_LOG_INTERVAL_SECONDS)

    try:
        subprocess.run(command_str, shell=True, executable="/bin/bash", check=True, env=env)
    except subprocess.CalledProcessError:
        log_global_gpu_metrics("training_failed", exp_name, dataset_name, MAX_TRAIN_STEPS)
        raise
    finally:
        stop_global_gpu_monitor(gpu_monitor)

    log_global_gpu_metrics("after_training", exp_name, dataset_name, MAX_TRAIN_STEPS)
    print(f"Finished: {exp_name}")
