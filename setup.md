# Environment Setup

This project uses two conda environments:

- **`sd-diffusers`** — for training and inference
- **`scores2`** — for metric computation and plotting

---

## Training Environment (`sd-diffusers`)

### 1. Create the environment

```bash
conda create -n sd-diffusers python=3.10 -y
conda activate sd-diffusers
```

### 2. Install PyTorch with CUDA

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.5.1+cu121 True
```

### 3. Install Diffusers from source

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd examples/text_to_image
pip install -r requirements.txt
cd ../../..
```

### 4. Install remaining dependencies

```bash
pip install codecarbon wandb pyyaml
```

### 5. Configure Accelerate for single GPU

```bash
CUDA_VISIBLE_DEVICES=0 accelerate config default
```

Or create the config manually:

```bash
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
use_cpu: false
EOF
```

### 6. Log in to Weights & Biases (optional)

```bash
wandb login
```

---

## Metrics & Plotting Environment (`scores2`)

### 1. Create the environment

```bash
conda create -n scores2 python=3.10 -y
conda activate scores2
```

### 2. Install PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install CLIP and metrics dependencies

```bash
pip install ftfy regex tqdm pandas pillow matplotlib scipy
pip install git+https://github.com/openai/CLIP.git
pip install torchmetrics[image]
```

### 4. Verify

```bash
python -c "import torch, clip, torchmetrics; print('OK')"
```

---

## Notes on Multi-GPU Systems

If your machine has multiple GPUs, always set `CUDA_VISIBLE_DEVICES` to avoid running on multiple GPUs unintentionally:

```bash
CUDA_VISIBLE_DEVICES=0 python your_script.py
```

Or set it inside the script at the top:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
