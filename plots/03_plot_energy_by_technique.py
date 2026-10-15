import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------

base = "/home/helena/Desktop/tfm/images_and_scores/sdv15"

output_dir = "results/lineplot_default_batch"
os.makedirs(output_dir, exist_ok=True)

# Colores por técnica
technique_colors = {
    "LoRA":       "#1565C0",
    "Finetuning": "#1B5E20",
    "TI":         "#E65100",
}

# Series: (root, train_names, label, technique)
series = {
    "titan": [
        (
            f"{base}/lora/energy/outputs_titan_with_cp",
            ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
             "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
             "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337"],
            "LoRA batch12", "LoRA",
        ),
        (
            f"{base}/finetuning/energy/outputs_titan_with_cp",
            ["0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
             "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
             "2_sdv15_finetuning_res512_Battista_steps5000_seed1337"],
            "Finetuning batch6", "Finetuning",
        ),
        (
            f"{base}/textual_inversion/energy/outputs_titan_with_cp_50",
            ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
             "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
             "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337"],
            "Textual Inv. batch12", "TI",
        ),
    ],
    "a40": [
        (
            f"{base}/lora/energy/outputs_a40_with_cp",
            ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
             "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
             "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337"],
            "LoRA batch24", "LoRA",
        ),
        (
            f"{base}/finetuning/energy/outputs_a40_with_cp",
            ["0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
             "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
             "2_sdv15_finetuning_res512_Battista_steps5000_seed1337"],
            "Finetuning batch18", "Finetuning",
        ),
        (
            f"{base}/textual_inversion/energy/outputs_a40_with_cp",
            ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
             "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
             "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337"],
            "Textual Inv. batch24", "TI",
        ),
    ],
}

# ----------------------------------------------------------------------
# FUNCIÓN DE CARGA
# ----------------------------------------------------------------------

def load_grouped(root, train_names):
    frames = []
    for train_name in train_names:
        fpath = os.path.join(root, train_name, "energy_clean_per_checkpoint.csv")
        if not os.path.isfile(fpath):
            print(f"[WARNING] No existe {fpath}, se omite.")
            continue
        df = pd.read_csv(fpath)[["global_step", "clean_energy_kwh"]]
        df["train_name"] = train_name
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df.groupby("global_step").agg(
        mean=("clean_energy_kwh", "mean"),
        std =("clean_energy_kwh", "std"),
    ).reset_index()

# ----------------------------------------------------------------------
# FUNCIÓN DE PLOT EN UN EJE
# ----------------------------------------------------------------------

def draw_lines(ax, serie_list, show_bands=True):
    for root, train_names, label, technique in serie_list:
        grouped = load_grouped(root, train_names)
        if grouped is None:
            continue
        steps = grouped["global_step"].values
        mean  = grouped["mean"].values
        std   = grouped["std"].values
        color = technique_colors[technique]

        ax.plot(steps, mean, color=color, linewidth=2,
                marker="o", markersize=3, label=label)
        if show_bands:
            ax.fill_between(steps, mean - std, mean + std,
                            color=color, alpha=0.12)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Energy (kWh)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")

# ----------------------------------------------------------------------
# GRÁFICAS INDIVIDUALES
# ----------------------------------------------------------------------

for gpu_key, gpu_title in [("titan", "Titan RTX"), ("a40", "A40")]:
    for show_bands in [True, False]:
        suffix = "" if show_bands else "_no_bands"
        fig, ax = plt.subplots(figsize=(10, 5))
        draw_lines(ax, series[gpu_key], show_bands=show_bands)
        ax.set_title(f"Energy vs Step — {gpu_title}", fontsize=13, pad=12)
        plt.tight_layout()
        path = os.path.join(output_dir, f"lineplot_{gpu_key}{suffix}.png")
        plt.savefig(path.replace('.png', '.svg'), format='svg')
        print(f"Guardado en {path}")
        plt.show()

# ----------------------------------------------------------------------
# SUBPLOT COMBINADO
# ----------------------------------------------------------------------

for show_bands in [True, False]:
    suffix = "" if show_bands else "_no_bands"
    fig, (ax_titan, ax_a40) = plt.subplots(1, 2, figsize=(18, 5), sharey=False)

    draw_lines(ax_titan, series["titan"], show_bands=show_bands)
    ax_titan.set_title("Titan RTX", fontsize=12)

    draw_lines(ax_a40, series["a40"], show_bands=show_bands)
    ax_a40.set_title("A40", fontsize=12)

    fig.suptitle("Energy vs Step by Technique", fontsize=13, y=0.98)
    plt.tight_layout()
    path = os.path.join(output_dir, f"lineplot_combined{suffix}.png")
    plt.savefig(path.replace('.png', '.svg'), format='svg')
    print(f"Guardado en {path}")
    plt.show()
