## Environment Setup

This project uses two conda environments:

* `sd-diffusers` — for adaptation and inference
* `scores` — for metric computation and plotting

The instructions in this section are orientative, as each environment may have slightly different requirements depending on your system. Treat this file as a guide. You can also use a single conda environment if you prefer — just be careful with library dependency conflicts.

---

### Adaptation & Inference Environment (`sd-diffusers`)

#### 1. Create the environment

```bash
conda create -n sd-diffusers python=3.10 -y
conda activate sd-diffusers
```

#### 2. Install PyTorch with CUDA

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Install Diffusers from source

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

The next step depends on the technique you want to use:

* For LoRA and Fine-tuning: `cd examples/text_to_image`
* For Textual Inversion: `cd examples/textual_inversion`

Then install the requirements and go back to the parent directory:

```bash
pip install -r requirements.txt
cd ../../..
```

#### 4. Install remaining dependencies

```bash
pip install codecarbon wandb pyyaml
```

#### 5. Configure Accelerate for single GPU

If you are working in a multi-GPU environment, run the following to avoid unintentional distributed training:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate config
```

#### 6. Log in to Weights & Biases (optional)

```bash
wandb login
```

---

### Metrics & Plotting Environment (`scores`)

#### 1. Create the environment

```bash
conda create -n scores python=3.10 -y
conda activate scores
```

#### 2. Install PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Install CLIP and metrics dependencies

```bash
pip install ftfy regex tqdm pandas pillow matplotlib scipy
pip install git+https://github.com/openai/CLIP.git
pip install torchmetrics[image]
```

---

### Notes on Multi-GPU Environments

If your machine has multiple GPUs, always set `CUDA_VISIBLE_DEVICES` to avoid running on multiple GPUs unintentionally:

```bash
CUDA_VISIBLE_DEVICES=0 python your_script.py
```

Or set it at the top of the script:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
