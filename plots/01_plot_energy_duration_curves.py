import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------

base = "/home/helena/Desktop/tfm/images_and_scores/sdv15"

lora_titan = f"{base}/lora/energy/outputs_titan_with_cp"
lora_a40   = f"{base}/lora/energy/outputs_a40_with_cp"
ft_titan   = f"{base}/finetuning/energy/outputs_titan_with_cp"
ft_a40     = f"{base}/finetuning/energy/outputs_a40_with_cp"
ti_titan   = f"{base}/textual_inversion/energy/outputs_titan_with_cp_50"
ti_a40     = f"{base}/textual_inversion/energy/outputs_a40_with_cp"

# Colores por batch
batch_colors = {
    "batch6":  "#1565C0",
    "batch12": "#1B5E20",
    "batch18": "#E65100",
    "batch24": "#6A1B9A",
}

# Estilos por técnica (para gráficas por GPU)
technique_ls = {
    "LoRA": "-",
    "FT":   "--",
    "TI":   ":",
}

# Estilos por GPU (para gráficas por técnica)
gpu_ls = {
    "Titan": "-",
    "A40":   "--",
}

# ----------------------------------------------------------------------
# SERIES: (root, train_names, label, batch_key, technique, gpu)
# ----------------------------------------------------------------------

all_series = [
    # --- TITAN ---
    (lora_titan, ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
                  "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
                  "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337"],
     "LoRA batch12 (Titan)", "batch12", "LoRA", "Titan"),

    (lora_titan, ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
                  "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
                  "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6"],
     "LoRA batch6 (Titan)", "batch6", "LoRA", "Titan"),

    (ft_titan,   ["0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
                  "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
                  "2_sdv15_finetuning_res512_Battista_steps5000_seed1337"],
     "FT batch6 (Titan)", "batch6", "FT", "Titan"),

    (ti_titan,   ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
                  "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
                  "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337"],
     "TI batch12 (Titan)", "batch12", "TI", "Titan"),

    (ti_titan,   ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
                  "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
                  "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6"],
     "TI batch6 (Titan)", "batch6", "TI", "Titan"),

    # --- A40 ---
    (lora_a40,   ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
                  "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
                  "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337"],
     "LoRA batch24 (A40)", "batch24", "LoRA", "A40"),

    (lora_a40,   ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18",
                  "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18",
                  "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18"],
     "LoRA batch18 (A40)", "batch18", "LoRA", "A40"),

    (lora_a40,   ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
                  "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
                  "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6"],
     "LoRA batch6 (A40)", "batch6", "LoRA", "A40"),

    (ft_a40,     ["0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
                  "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
                  "2_sdv15_finetuning_res512_Battista_steps5000_seed1337"],
     "FT batch18 (A40)", "batch18", "FT", "A40"),

    (ti_a40,     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
                  "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
                  "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337"],
     "TI batch24 (A40)", "batch24", "TI", "A40"),

    (ti_a40,     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
                  "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
                  "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18"],
     "TI batch18 (A40)", "batch18", "TI", "A40"),

    (ti_a40,     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
                  "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
                  "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6"],
     "TI batch6 (A40)", "batch6", "TI", "A40"),
]

output_dir = "results/lineplot_batch_color"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------------------
# FUNCIÓN DE CARGA
# ----------------------------------------------------------------------

def load_grouped(root, train_names, value_col, csv_file):
    frames = []
    for train_name in train_names:
        fpath = os.path.join(root, train_name, csv_file)
        if not os.path.isfile(fpath):
            print(f"[WARNING] No existe {fpath}, se omite.")
            continue
        df = pd.read_csv(fpath)[["global_step", value_col]]
        df["train_name"] = train_name
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df.groupby("global_step").agg(
        mean=(value_col, "mean"),
        std =(value_col, "std"),
    ).reset_index()

# ----------------------------------------------------------------------
# FUNCIÓN DE PLOT GENÉRICA
# ----------------------------------------------------------------------

def plot_lines(series_list, title, value_col, csv_file, ylabel, output_path,
               show_bands=False, xlim=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    for root, train_names, label, batch_key, technique, gpu in series_list:
        grouped = load_grouped(root, train_names, value_col, csv_file)
        if grouped is None:
            continue
        steps = grouped["global_step"].values
        mean  = grouped["mean"].values
        std   = grouped["std"].values
        color = batch_colors[batch_key]
        ls    = technique_ls.get(technique, gpu_ls.get(gpu, "-"))

        ax.plot(steps, mean, color=color, linestyle=ls,
                linewidth=2, marker="o", markersize=3, label=label)
        if show_bands:
            ax.fill_between(steps, mean - std, mean + std,
                            color=color, alpha=0.1)

    if xlim is not None:
        ax.set_xlim(0, xlim)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    metric_prefix = "Duration" if "duration" in value_col else "Energy"
    full_title = title.replace("Energy vs Step", f"{metric_prefix} vs Step")
    plt.title(full_title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '.svg'), format='svg')
    print(f"Gráfica guardada en {output_path}")
    plt.show()


def plot_combined_gpu(value_col, csv_file, ylabel, output_path, show_bands=False, xlim=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    gpu_filters = {
        "Titan RTX": lambda s: s[5] == "Titan",
        "A40":       lambda s: s[5] == "A40",
    }

    for ax, (gpu_name, gpu_filter) in zip(axes, gpu_filters.items()):
        filtered = [s for s in all_series if gpu_filter(s)]
        for root, train_names, label, batch_key, technique, gpu in filtered:
            grouped = load_grouped(root, train_names, value_col, csv_file)
            if grouped is None:
                continue
            steps = grouped["global_step"].values
            mean  = grouped["mean"].values
            std   = grouped["std"].values
            color = batch_colors[batch_key]
            ls    = technique_ls.get(technique, "-")

            ax.plot(steps, mean, color=color, linestyle=ls,
                    linewidth=2, marker="o", markersize=3, label=label)
            if show_bands:
                ax.fill_between(steps, mean - std, mean + std,
                                color=color, alpha=0.1)

        if xlim is not None:
            ax.set_xlim(0, xlim)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_title(gpu_name, fontsize=12)
        ax.legend(fontsize=9)

    axes[0].set_ylabel(ylabel, fontsize=12)
    metric_prefix = "Duration" if "duration" in value_col else "Energy"
    fig.suptitle(f"{metric_prefix} vs Step by GPU (color=batch, style=technique)",
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '.svg'), format='svg')
    print(f"Gráfica guardada en {output_path}")
    plt.show()



    
# ----------------------------------------------------------------------
# DEFINIR GRUPOS
# ----------------------------------------------------------------------

groups = {
    # Por GPU (color=batch, estilo=técnica)
    "titan": {
        "title": "Energy vs Step — Titan RTX (color=batch, style=technique)",
        "filter": lambda s: s[5] == "Titan",
        "ls_field": "technique",
    },
    "a40": {
        "title": "Energy vs Step — A40 (color=batch, style=technique)",
        "filter": lambda s: s[5] == "A40",
        "ls_field": "technique",
    },
    # Por técnica (color=batch, estilo=GPU)
    "lora": {
        "title": "Energy vs Step — LoRA (color=batch, style=GPU)",
        "filter": lambda s: s[4] == "LoRA",
        "ls_field": "gpu",
    },
    "ft": {
        "title": "Energy vs Step — Finetuning (color=batch, style=GPU)",
        "filter": lambda s: s[4] == "FT",
        "ls_field": "gpu",
    },
    "ti": {
        "title": "Energy vs Step — Textual Inversion (color=batch, style=GPU)",
        "filter": lambda s: s[4] == "TI",
        "ls_field": "gpu",
    },
}

metrics = [
    ("clean_energy_kwh", "energy_clean_per_checkpoint.csv", "Energy (kWh)"),
    ("clean_duration_s", "duration_clean_per_checkpoint.csv", "Duration (s)"),
]

# ----------------------------------------------------------------------
# GENERAR GRÁFICAS
# ----------------------------------------------------------------------

for group_key, group_cfg in groups.items():
    filtered = [s for s in all_series if group_cfg["filter"](s)]

    # Ajustar ls según el campo (técnica o GPU)
    ls_field = group_cfg["ls_field"]
    filtered_with_ls = []
    for s in filtered:
        root, train_names, label, batch_key, technique, gpu = s
        if ls_field == "technique":
            ls = technique_ls.get(technique, "-")
        else:
            ls = gpu_ls.get(gpu, "-")
        filtered_with_ls.append((root, train_names, label, batch_key, technique, gpu))

    for value_col, csv_file, ylabel in metrics:
        plot_lines(filtered_with_ls, group_cfg["title"], value_col, csv_file, ylabel,
                   output_path=os.path.join(output_dir, f"{group_key}_{value_col}.png"),
                   show_bands=False)
        plot_lines(filtered_with_ls, group_cfg["title"], value_col, csv_file, ylabel,
                   output_path=os.path.join(output_dir, f"{group_key}_{value_col}_to1500.png"),
                   show_bands=False, xlim=1500)

for value_col, csv_file, ylabel in metrics:
    plot_combined_gpu(value_col, csv_file, ylabel,
                    output_path=os.path.join(output_dir, f"combined_gpu_{value_col}.png"))
    plot_combined_gpu(value_col, csv_file, ylabel,
                    output_path=os.path.join(output_dir, f"combined_gpu_{value_col}_to1500.png"),
                    xlim=1500)