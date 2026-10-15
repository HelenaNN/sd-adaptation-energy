import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------

base_energy = "/home/helena/Desktop/tfm/images_and_scores/sdv15"

output_dir = "results/barplot_optimal"
os.makedirs(output_dir, exist_ok=True)

configs = {
    "titan": [
        {
            "label": "Fine-tuning\n Batch 6 - Step 2500",
            "root": f"{base_energy}/finetuning/energy/outputs_titan_with_cp",
            "train_names": [
                "0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
                "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
                "2_sdv15_finetuning_res512_Battista_steps5000_seed1337",
            ],
            "checkpoint": 2500,
            "color": "#1B5E20",
        },
        {
            "label": "LoRA\n Batch 6 - Step 1000",
            "root": f"{base_energy}/lora/energy/outputs_titan_with_cp",
            "train_names": [
                "0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
                "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
                "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
            ],
            "checkpoint": 1000,
            "color": "#1565C0",
        },
        {
            "label": "Textual Inv.\n Batch 6 - Step 1000",
            "root": f"{base_energy}/textual_inversion/energy/outputs_titan_with_cp_50",
            "train_names": [
                "0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
                "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
                "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
            ],
            "checkpoint": 1000,
            "color": "#E65100",
        },
    ],
    "a40": [
        {
            "label": "Fine-tuning\n Batch 18 - Step 500",
            "root": f"{base_energy}/finetuning/energy/outputs_a40_with_cp",
            "train_names": [
                "0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
                "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
                "2_sdv15_finetuning_res512_Battista_steps5000_seed1337",
            ],
            "checkpoint": 500,
            "color": "#1B5E20",
        },
        {
            "label": "LoRA\n Batch 6 - Step 1000",
            "root": f"{base_energy}/lora/energy/outputs_a40_with_cp",
            "train_names": [
                "0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
                "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
                "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
            ],
            "checkpoint": 1000,
            "color": "#1565C0",
        },
        {
            "label": "Textual Inv.\n Batch 18 - Step 500",
            "root": f"{base_energy}/textual_inversion/energy/outputs_a40_with_cp",
            "train_names": [
                "0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
                "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
                "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
            ],
            "checkpoint": 500,
            "color": "#E65100",
        },
    ],
}

# ----------------------------------------------------------------------
# FUNCIÓN DE CARGA
# ----------------------------------------------------------------------

def get_energy_at_checkpoint(root, train_names, checkpoint):
    values = []
    for train_name in train_names:
        fpath = os.path.join(root, train_name, "energy_clean_per_checkpoint.csv")
        if not os.path.isfile(fpath):
            print(f"[WARNING] No existe {fpath}")
            continue
        df = pd.read_csv(fpath)
        row = df[df["global_step"] == checkpoint]
        if row.empty:
            print(f"[WARNING] Checkpoint {checkpoint} no encontrado en {fpath}")
            continue
        values.append(row["clean_energy_kwh"].values[0])
    if not values:
        return float("nan"), 0.0
    return np.mean(values), np.std(values)

# ----------------------------------------------------------------------
# FUNCIÓN DE PLOT (reutilizable para individual y subplot)
# ----------------------------------------------------------------------

def draw_barplot(ax, cfg_list):
    labels = [c["label"] for c in cfg_list]
    colors = [c["color"] for c in cfg_list]
    means, stds = [], []

    for c in cfg_list:
        mean, std = get_energy_at_checkpoint(c["root"], c["train_names"], c["checkpoint"])
        means.append(mean)
        stds.append(std)
        print(f"  {c['label'].replace(chr(10), ' ')}: {mean:.4f} +/- {std:.4f} kWh")

    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, yerr=stds, capsize=5,
                  error_kw={"elinewidth": 1.5}, width=0.5)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + max(means) * 0.01,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Energy (kWh)", fontsize=12)
    ax.set_xlabel("Technique", fontsize=12)
    ax.set_ylim(0, max(means) * 1.2)

# ----------------------------------------------------------------------
# GRÁFICAS INDIVIDUALES
# ----------------------------------------------------------------------

print("\n=== TITAN RTX ===")
fig, ax = plt.subplots(figsize=(8, 5))
draw_barplot(ax, configs["titan"])
ax.set_title("Energy at each checkpoint — Titan RTX", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "barplot_optimal_titan.svg"), format='svg')
print(f"Guardado en {output_dir}/barplot_optimal_titan.svg")
plt.show()

print("\n=== A40 ===")
fig, ax = plt.subplots(figsize=(8, 5))
draw_barplot(ax, configs["a40"])
ax.set_title("Energy at each checkpoint — A40", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "barplot_optimal_a40.svg"), format='svg')
print(f"Guardado en {output_dir}/barplot_optimal_a40.svg")
plt.show()

# ----------------------------------------------------------------------
# SUBPLOT COMBINADO
# ----------------------------------------------------------------------

print("\n=== SUBPLOT COMBINADO ===")
fig, (ax_titan, ax_a40) = plt.subplots(1, 2, figsize=(16, 5))

print("Titan RTX:")
draw_barplot(ax_titan, configs["titan"])
ax_titan.set_title("Titan RTX", fontsize=12)

print("A40:")
draw_barplot(ax_a40, configs["a40"])
ax_a40.set_title("A40", fontsize=12)

fig.suptitle("Energy at optimal checkpoint by technique and GPU", fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "barplot_optimal_combined.svg"), format='svg')
print(f"Guardado en {output_dir}/barplot_optimal_combined.svg")
plt.show()