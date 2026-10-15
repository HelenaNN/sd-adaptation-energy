import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# RUTAS BASE
# ----------------------------------------------------------------------

ti_a40_old        = "/home/helena/Desktop/tfm/images_and_scores/sdv15/textual_inversion/energy/outputs_a40_with_cp"
ti_a40_isolated   = "/home/helena/Desktop/tfm/images_and_scores/sdv15/textual_inversion/energy/outputs_a40_isolated_gpu"
ti_a40_multi      = "/home/helena/Desktop/tfm/images_and_scores/sdv15/textual_inversion/energy/outputs_a40_multi_gpu"
cat_statue_dir    = "/home/helena/Desktop/tfm/images_and_scores/subject/energy/outputs_a40_isolated_gpu/cat_statue"
dog6_dir          = "/home/helena/Desktop/tfm/images_and_scores/subject/energy/outputs_a40_isolated_gpu/dog6"

# Train names
ti_batch6_old = [
    "0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
    "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
    "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
]
ti_batch6_new = [
    "0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
    "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
    "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
]
cat_statue_names = [
    "0_sdv15_textual_inversion_res512_cat_statue_steps5000_seed1337",
    "1_sdv15_textual_inversion_res512_cat_statue_steps5000_seed1337",
    "2_sdv15_textual_inversion_res512_cat_statue_steps5000_seed1337",
]
dog6_names = [
    "0_sdv15_textual_inversion_res512_dog6_steps5000_seed1337",
    "1_sdv15_textual_inversion_res512_dog6_steps5000_seed1337",
    "2_sdv15_textual_inversion_res512_dog6_steps5000_seed1337",
]

# ----------------------------------------------------------------------
# COMPARATIVAS
# ----------------------------------------------------------------------

comparisons = {
    # --- Comparativas de sistema (mismo batch6, distinto sistema) ---
    "1_old_vs_isolated": {
        "title": "TI A40 batch6 — Old system vs Isolated GPU",
        "series": [
            (ti_a40_old,      ti_batch6_old, "Old system (batch6)",      "#1565C0", "-"),
            (ti_a40_isolated, ti_batch6_new, "Isolated GPU (batch6)",    "#42A5F5", "--"),
        ],
    },
    "2_isolated_vs_multi": {
        "title": "TI A40 batch6 — Isolated GPU vs Multi-GPU",
        "series": [
            (ti_a40_isolated, ti_batch6_new, "Isolated GPU (batch6)", "#1565C0", "-"),
            (ti_a40_multi,    ti_batch6_new, "Multi-GPU (batch6)",    "#42A5F5", "--"),
        ],
    },
    "3_old_vs_multi": {
        "title": "TI A40 batch6 — Dedicated-GPU vs Multi-GPU Environments",
        "series": [
            (ti_a40_old,   ti_batch6_old, "Dedicated-GPU (batch6)", "#1565C0", "-"),
            (ti_a40_multi, ti_batch6_new, "Multi-GPU (batch6)",  "#42A5F5", "--"),
        ],
    },
    # --- Comparativas de tarea ---
    "4_style_vs_subjects": {

        "title": "TI A40 batch6 — Style vs Subject Personalization",
        "series": [

            (ti_a40_isolated, ti_batch6_new,   "Style - Piranesi",     "#1565C0", "-"),
            (cat_statue_dir,  cat_statue_names, "Subject - cat_statue", "#E65100", "--"),
            (dog6_dir,        dog6_names,        "Subject - dog6",       "#1B5E20", "-."),
        ],
    },
    "5_cat_statue_vs_dog6": {
        "title": "TI A40 batch6 — Subject cat_statue vs dog6",
        "series": [
            (cat_statue_dir, cat_statue_names, "Subject - cat_statue", "#E65100", "-"),
            (dog6_dir,       dog6_names,       "Subject - dog6",       "#1B5E20", "--"),
        ],
    },
}

output_dir = "results/comparisons_TI_2"
os.makedirs(output_dir, exist_ok=True)

metrics = [
    ("clean_energy_kwh", "energy_clean_per_checkpoint.csv", "Energy (kWh)"),
    ("clean_duration_s", "duration_clean_per_checkpoint.csv", "Duration (s)"),
]

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
# FUNCIÓN DE PLOT
# ----------------------------------------------------------------------

def plot_comparison(comp_key, comp_cfg, value_col, csv_file, ylabel, output_path, show_bands=True):
    fig, ax = plt.subplots(figsize=(12, 6))

    for root, train_names, label, color, ls in comp_cfg["series"]:
        grouped = load_grouped(root, train_names, value_col, csv_file)
        if grouped is None:
            continue
        steps = grouped["global_step"].values
        mean  = grouped["mean"].values
        std   = grouped["std"].values

        ax.plot(steps, mean, color=color, linestyle=ls, linewidth=2,
                marker="o", markersize=4, label=label)
        if show_bands:
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.1)

    metric_prefix = "Duration" if "duration" in value_col else "Energy"
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    plt.title(f"{metric_prefix} vs Step - {comp_cfg['title']}", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '.svg'), format='svg')
    print(f"Gráfica guardada en {output_path}")
    plt.show()

# ----------------------------------------------------------------------
# GENERAR GRÁFICAS
# ----------------------------------------------------------------------

for comp_key, comp_cfg in comparisons.items():
    for value_col, csv_file, ylabel in metrics:
        plot_comparison(comp_key, comp_cfg, value_col, csv_file, ylabel,
                        output_path=os.path.join(output_dir, f"{comp_key}_{value_col}.png"),
                        show_bands=True)
        plot_comparison(comp_key, comp_cfg, value_col, csv_file, ylabel,
                        output_path=os.path.join(output_dir, f"{comp_key}_{value_col}_no_bands.png"),
                        show_bands=False)