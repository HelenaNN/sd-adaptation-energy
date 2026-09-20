# sd-adaptation-energy

A research framework for measuring and analysing the **energy consumption** of Stable Diffusion model adaptation techniques. This repository contains all the code used to train, evaluate, and visualise results for three personalisation methods — **LoRA**, **Fine-tuning**, and **Textual Inversion** — across different GPUs, batch sizes, and tasks.

---

## Overview

Model adaptation techniques for diffusion models are increasingly used for personalisation tasks. However, their energy footprint is rarely studied. This project provides:

- Training scripts with **CodeCarbon** energy tracking integrated
- A pipeline to **clean and correct** energy and duration measurements (removing checkpoint-saving overhead)
- Scripts to compute **quality metrics** (CLIP-T, KID, CLIP-I, editability score)
- A full set of **visualisation scripts** for energy and quality analysis

---

## Techniques Covered

| Technique         | Task                            | Base model            |
| ----------------- | ------------------------------- | --------------------- |
| LoRA              | Style personalisation           | Stable Diffusion v1.5 |
| Fine-tuning       | Style personalisation           | Stable Diffusion v1.5 |
| Textual Inversion | Style & subject personalisation | Stable Diffusion v1.5 |

---

## Repository Structure

```
sd-adaptation-energy/
├── README.md
├── workflow.md               # Step-by-step execution guide
├── requirements.txt
├── setup.md                  # Environment setup instructions
│
├── training/                 # Training scripts with energy tracking
│   ├── README.md
│   ├── core/                 # Core training scripts (CodeCarbon integrated)
│   └── launchers/            # Experiment launcher scripts
│
├── inference/                # Image generation scripts
│   ├── README.md
│   └── evolution/            # Visual evolution grids per checkpoint
│
├── energy/                   # Energy & duration analysis pipeline
│   ├── README.md
│   └── calibration/          # Checkpoint-saving duration calibration
│
├── metrics/                  # Quality metric computation
│   ├── README.md
│   └── subject/              # Subject personalisation metrics
│
└── plots/                    # Visualisation scripts
    ├── README.md
    ├── energy/
    └── quality/
```

---

## Quick Start

1. Set up the environment — see [`setup.md`](setup.md)
2. Follow the full pipeline — see [`workflow.md`](workflow.md)
3. Each folder has its own `README.md` with detailed instructions

---

## Requirements

- Python 3.10
- PyTorch 2.5.1 + CUDA 12.1
- Diffusers (installed from source — see `setup.md`)
- CodeCarbon
- See [`requirements.txt`](requirements.txt) for the full list

---

## Hardware

Experiments were run on:

- **NVIDIA TITAN RTX** (24 GB VRAM)
- **NVIDIA A40** (46 GB VRAM)

---

## Citation

If you use this code for your research, please cite the original TFM thesis and the following works:

- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- Gal et al., *An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion*, ICLR 2023
- Courty et al., *CodeCarbon: Estimate and Track Carbon Emissions from Machine Learning Computing*, 2023

---

## License

MIT License
