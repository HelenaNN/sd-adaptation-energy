# Environment Setup

This project uses two conda environments:

- **`sd-diffusers`** — for adaptation and inference
- **`scores`** — for metric computation and plotting

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

### 3. Install Diffusers from source

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
The next instructions depends on the technique selected: for LoRA and Fine-tuning is **cd examples/text_to_image** but for Textual Inversion is **cd example/textual_inversion**:

```bash
cd examples/text_to_image
pip install -r requirements.txt
cd ../../..
´´´

### 4. Install remaining dependencies

```bash
pip install codecarbon wandb pyyaml
```

### 5. Configure Accelerate for single GPU

This step is if you work in a Multi-GPU Environment.

```bash
CUDA_VISIBLE_DEVICES=0 accelerate config default
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

---

## Notes on Multi-GPU Environments

If your machine has multiple GPUs, always set `CUDA_VISIBLE_DEVICES` to avoid running on multiple GPUs unintentionally:

```bash
CUDA_VISIBLE_DEVICES=0 python your_script.py
```

Or set it inside the script at the top:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
