import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------

scores_csv = "scores_finetuning_a40_batch6.csv"
energy_root = "/mnt/rhome/hnn/images_and_scores/sdv15/finetuning/energy/outputs_titan_with_cp"

train_names = [
    "0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
    "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
    "2_sdv15_finetuning_res512_Battista_steps5000_seed1337",
]

checkpoints = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

output_dir = "results/outputs_titan_batch6"
os.makedirs(output_dir, exist_ok=True)

output_csv  = f"{output_dir}/scores_energy_finetuning_titan_batch6.csv"
output_plot = f"{output_dir}/scores_energy_finetuning_titan_batch6.png"
output_plot_no_bands = f"{output_dir}/scores_energy_finetuning_titan_batch6_no_bands.png"

# ----------------------------------------------------------------------
# CARGAR Y CRUZAR DATOS
# ----------------------------------------------------------------------

scores_df = pd.read_csv(scores_csv)

energy_frames = []
for train_name in train_names:
    energy_file = os.path.join(energy_root, train_name, "energy_clean_per_checkpoint.csv")

    if not os.path.isfile(energy_file):
        print(f"[WARNING] No existe {energy_file}, se omite.")
        continue

    edf = pd.read_csv(energy_file)
    edf = edf[edf["global_step"].isin(checkpoints)][["global_step", "clean_energy_kwh"]]
    edf["train_name"] = train_name
    energy_frames.append(edf)

energy_df = pd.concat(energy_frames, ignore_index=True)

merged = pd.merge(
    scores_df,
    energy_df,
    left_on=["train_name", "checkpoint"],
    right_on=["train_name", "global_step"],
).drop(columns=["global_step"])

# ----------------------------------------------------------------------
# CALCULAR MEDIAS POR CHECKPOINT (promediando los 3 seeds)
# ----------------------------------------------------------------------

grouped = merged.groupby("checkpoint").agg(
    CLIP_T_mean      = ("CLIP_T_score",    "mean"),
    CLIP_T_std       = ("CLIP_T_score",    "std"),
    KID_mean_avg     = ("KID_mean",        "mean"),
    KID_mean_std     = ("KID_mean",        "std"),
    energy_mean_kwh  = ("clean_energy_kwh","mean"),
    energy_std_kwh   = ("clean_energy_kwh","std"),
    n_runs           = ("train_name",      "count"),
).reset_index()

print(grouped[["checkpoint", "energy_mean_kwh", "CLIP_T_mean", "KID_mean_avg"]].to_string(index=False))

merged.to_csv(output_csv, index=False)
print(f"\nDatos cruzados guardados en {output_csv}")

# ----------------------------------------------------------------------
# FUNCIÓN DE PLOT
# ----------------------------------------------------------------------

def plot_scores_energy(grouped, output_path, show_bands=True):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_clip = "#2196F3"
    color_kid  = "#E53935"

    energy = grouped["energy_mean_kwh"].values

    # --- Eje izquierdo: CLIP-T ---
    ax1.set_xlabel("Energy (kWh)", fontsize=12)
    ax1.set_ylabel("CLIP-T", fontsize=12)
    ax1.plot(energy, grouped["CLIP_T_mean"], color=color_clip,
             marker="o", linewidth=2, label="CLIP-T")
    if show_bands:
        ax1.fill_between(energy,
                         grouped["CLIP_T_mean"] - grouped["CLIP_T_std"],
                         grouped["CLIP_T_mean"] + grouped["CLIP_T_std"],
                         color=color_clip, alpha=0.15)
    ax1.tick_params(axis="y")

    # --- Eje derecho: KID ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("KID", fontsize=12)
    ax2.plot(energy, grouped["KID_mean_avg"], color=color_kid,
             marker="s", linewidth=2, linestyle="--", label="KID")
    if show_bands:
        ax2.fill_between(energy,
                         grouped["KID_mean_avg"] - grouped["KID_mean_std"],
                         grouped["KID_mean_avg"] + grouped["KID_mean_std"],
                         color=color_kid, alpha=0.15)
    ax2.tick_params(axis="y")

    # Eje X superior con los checkpoints
    ax_top = ax1.twiny()
    ax_top.set_xlim(ax1.get_xlim())
    ax_top.set_xticks(grouped["energy_mean_kwh"].values)
    ax_top.set_xticklabels([str(int(c)) for c in grouped["checkpoint"].values],
                            fontsize=9, color="black")
    ax_top.set_xlabel("Steps", fontsize=11, color="black", labelpad=8)
    ax_top.tick_params(axis="x", length=4, color="gray")

    # --- Leyenda combinada ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10, bbox_to_anchor=(0.05, 0.2))

    plt.title("CLIP-T and KID vs Energy (TITAN RTX - Finetuning - Batch 6)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Gráfica guardada en {output_path}")
    plt.show()

# ----------------------------------------------------------------------
# GENERAR AMBAS GRÁFICAS
# ----------------------------------------------------------------------

plot_scores_energy(grouped, output_plot,          show_bands=True)
plot_scores_energy(grouped, output_plot_no_bands, show_bands=False)