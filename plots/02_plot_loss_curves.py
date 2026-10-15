import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------

# Carpeta donde están los CSVs exportados de wandb (uno por config,
# columna "Step" + una columna "<run_name> - loss" o "<run_name> - train_loss").
# Ajusta esta ruta a donde tengas guardados los exports.
base = "/home/helena/Desktop/tfm/images_and_scores/sdv15/loss_curves_csv/sdv15"

# Suavizado (mismo alpha que se validó visualmente contra wandb)
EMA_ALPHA = 0.95

# Colores por batch (idénticos a 2_plot_lineplot_batch_color.py para mantener
# consistencia visual entre las gráficas de energía y las de loss)
batch_colors = {
    "batch6":  "#1565C0",
    "batch12": "#1B5E20",
    "batch18": "#E65100",
    "batch24": "#6A1B9A",
}

# ----------------------------------------------------------------------
# SERIES: (csv_file, label, batch_key, technique, gpu)
# Nombres de fichero según lo exportado de wandb. Ajustar si cambian.
# ----------------------------------------------------------------------

all_series = [
    # --- TITAN ---
    ("lora_titan_batch12.csv",             "LoRA batch12 (Titan)", "batch12", "LoRA", "Titan"),
    ("lora_titan_batch6.csv",              "LoRA batch6 (Titan)",  "batch6",  "LoRA", "Titan"),
    ("finetuning_titan_batch6.csv",        "FT batch6 (Titan)",    "batch6",  "FT",   "Titan"),
    ("textual_inversion_titan_batch12.csv","TI batch12 (Titan)",   "batch12", "TI",   "Titan"),
    ("textual_inversion_titan_batch6.csv", "TI batch6 (Titan)",    "batch6",  "TI",   "Titan"),

    # --- A40 ---
    ("lora_a40_batch24.csv",               "LoRA batch24 (A40)",   "batch24", "LoRA", "A40"),
    ("lora_a40_batch18.csv",               "LoRA batch18 (A40)",   "batch18", "LoRA", "A40"),
    ("lora_a40_batch6.csv",                "LoRA batch6 (A40)",    "batch6",  "LoRA", "A40"),
    ("finetuning_a40_batch18.csv",         "FT batch18 (A40)",     "batch18", "FT",   "A40"),
    # NOTA: falta finetuning_a40_batch6.csv - pendiente de exportar de wandb
    ("textual_inversion_a40_batch24.csv",  "TI batch24 (A40)",     "batch24", "TI",   "A40"),
    ("textual_inversion_a40_batch18.csv",  "TI batch18 (A40)",     "batch18", "TI",   "A40"),
    ("textual_inversion_a40_batch6.csv",   "TI batch6 (A40)",      "batch6",  "TI",   "A40"),
]

output_dir = "results/lineplot_loss"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------------------
# SUAVIZADO: Time Weighted EMA con debiasing (misma fórmula que usa wandb)
# ----------------------------------------------------------------------

def ema_debiased(series, alpha=EMA_ALPHA):
    smoothed = []
    last = 0.0
    num_acc = 0.0
    for x in series:
        last = last * alpha + x * (1 - alpha)
        num_acc = num_acc * alpha + (1 - alpha)
        smoothed.append(last / num_acc)
    return pd.Series(smoothed, index=series.index)

# ----------------------------------------------------------------------
# FUNCIÓN DE CARGA
# ----------------------------------------------------------------------

def load_loss(csv_file):
    fpath = os.path.join(base, csv_file)
    if not os.path.isfile(fpath):
        print(f"[WARNING] No existe {fpath}, se omite.")
        return None
    df = pd.read_csv(fpath)
    df.columns = [c.strip() for c in df.columns]
    # columna de loss: la que termina en "loss" y no es __MIN/__MAX
    loss_cols = [c for c in df.columns if c.endswith("loss") and "__" not in c]
    if not loss_cols:
        print(f"[WARNING] No se encontró columna de loss en {fpath}, se omite.")
        return None
    loss_col = loss_cols[0]
    out = df[["Step", loss_col]].rename(columns={loss_col: "loss"})
    out["loss_smoothed"] = ema_debiased(out["loss"])
    return out

# ----------------------------------------------------------------------
# FUNCIÓN DE PLOT (línea continua, color=batch, sin distinguir GPU por estilo)
# ----------------------------------------------------------------------

def plot_lines(series_list, title, output_path, xlim=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    for csv_file, label, batch_key, technique, gpu in series_list:
        data = load_loss(csv_file)
        if data is None:
            continue
        color = batch_colors[batch_key]

        ax.plot(data["Step"], data["loss"], color=color, linewidth=0.4, alpha=0.15)
        ax.plot(data["Step"], data["loss_smoothed"], color=color, linestyle="-",
                linewidth=0.7, label=label)

    if xlim is not None:
        ax.set_xlim(0, xlim)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel(f"Loss (EMA α={EMA_ALPHA}, debiased)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    plt.title(title, fontsize=13, pad=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '.svg'), format='svg')
    print(f"Gráfica guardada en {output_path}")
    plt.show()

# ----------------------------------------------------------------------
# GRUPOS: solo por técnica (color=batch), sin mezclar técnicas entre sí
# ----------------------------------------------------------------------

groups = {
    "lora": {
        "title": "Loss vs Step — LoRA (color=batch)",
        "filter": lambda s: s[3] == "LoRA",
    },
    "ft": {
        "title": "Loss vs Step — Finetuning (color=batch)",
        "filter": lambda s: s[3] == "FT",
    },
    "ti": {
        "title": "Loss vs Step — Textual Inversion (color=batch)",
        "filter": lambda s: s[3] == "TI",
    },
}

# ----------------------------------------------------------------------
# GENERAR GRÁFICAS
# ----------------------------------------------------------------------

for group_key, group_cfg in groups.items():
    filtered = [s for s in all_series if group_cfg["filter"](s)]

    plot_lines(filtered, group_cfg["title"],
               output_path=os.path.join(output_dir, f"{group_key}_loss.png"))
    plot_lines(filtered, group_cfg["title"],
               output_path=os.path.join(output_dir, f"{group_key}_loss_to1500.png"),
               xlim=1500)