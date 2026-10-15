import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------

base_scores = "/home/helena/Desktop/tfm/images_and_scores/sdv15"
base_energy = "/home/helena/Desktop/tfm/images_and_scores/sdv15"

output_dir = "results/bubble_scores_b"
os.makedirs(output_dir, exist_ok=True)

batch_colors = {
    "batch6":  "#1565C0",
    "batch12": "#1B5E20",
    "batch18": "#E65100",
    "batch24": "#6A1B9A",
}

gpu_ls = {
    "Titan": "-",
    "A40":   "--",
}

checkpoints_lora = [500, 1000, 2000, 3000, 4000, 5000]
checkpoints_ti   = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
checkpoints_ft   = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500,
                    2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000]

BUBBLE_SCALE = 300

series = {
    "LoRA Titan batch12": (
        f"{base_scores}/lora/scores_lora_titan_batch12.csv",
        f"{base_energy}/lora/energy/outputs_titan_with_cp",
        [
            "0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
            "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
            "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
        ],
        "batch12", "Titan", "LoRA", checkpoints_lora
    ),

    "LoRA Titan batch6": (
        f"{base_scores}/lora/scores_lora_titan_batch6.csv",
        f"{base_energy}/lora/energy/outputs_titan_with_cp",
        [
            "0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
            "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
            "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
        ],
        "batch6", "Titan", "LoRA", checkpoints_lora
    ),

    "LoRA A40 batch24": (
        f"{base_scores}/lora/scores_lora_a40_batch24.csv",
        f"{base_energy}/lora/energy/outputs_a40_with_cp",
        [
            "0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
            "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
            "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
        ],
        "batch24", "A40", "LoRA", checkpoints_lora
    ),

    "LoRA A40 batch18": (
        f"{base_scores}/lora/scores_lora_a40_batch18.csv",
        f"{base_energy}/lora/energy/outputs_a40_with_cp",
        [
            "0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18",
            "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18",
            "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18",
        ],
        "batch18", "A40", "LoRA", checkpoints_lora
    ),

    "LoRA A40 batch6": (
        f"{base_scores}/lora/scores_lora_a40_batch6.csv",
        f"{base_energy}/lora/energy/outputs_a40_with_cp",
        [
            "0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
            "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
            "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
        ],
        "batch6", "A40", "LoRA", checkpoints_lora
    ),

    "Fine-tuning Titan batch6": (
        f"{base_scores}/finetuning/scores_finetuning_titan_batch6.csv",
        f"{base_energy}/finetuning/energy/outputs_titan_with_cp",
        [
            "0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
            "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
            "2_sdv15_finetuning_res512_Battista_steps5000_seed1337",
        ],
        "batch6", "Titan", "FT", checkpoints_ft
    ),

    "Fine-tuning A40 batch18": (
        f"{base_scores}/finetuning/scores_finetuning_a40.csv",
        f"{base_energy}/finetuning/energy/outputs_a40_with_cp",
        [
            "0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
            "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
            "2_sdv15_finetuning_res512_Battista_steps5000_seed1337",
        ],
        "batch18", "A40", "FT", checkpoints_ft
    ),

    "Fine-tuning A40 batch6": (
        f"{base_scores}/finetuning/scores_finetuning_a40_batch6.csv",
        f"{base_energy}/finetuning/energy/outputs_a40_with_cp",
        [
            "0_sdv15_finetuning_res512_Battista_steps5000_seed1337_batch6",
            "1_sdv15_finetuning_res512_Battista_steps5000_seed1337_batch6",
            "2_sdv15_finetuning_res512_Battista_steps5000_seed1337_batch6",
        ],
        "batch6", "A40", "FT", checkpoints_ft
    ),

    "Textual Inversion Titan batch12": (
        f"{base_scores}/textual_inversion/scores_TI_titan_batch12.csv",
        f"{base_energy}/textual_inversion/energy/outputs_titan_with_cp_50",
        [
            "0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
            "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
            "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
        ],
        "batch12", "Titan", "TI", checkpoints_ti
    ),

    "Textual Inversion Titan batch6": (
        f"{base_scores}/textual_inversion/scores_TI_titan_batch6.csv",
        f"{base_energy}/textual_inversion/energy/outputs_titan_with_cp_50",
        [
            "0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
            "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
            "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
        ],
        "batch6", "Titan", "TI", checkpoints_ti
    ),

    "Textual Inversion A40 batch24": (
        f"{base_scores}/textual_inversion/scores_TI_a40_batch24.csv",
        f"{base_energy}/textual_inversion/energy/outputs_a40_with_cp",
        [
            "0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
            "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
            "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
        ],
        "batch24", "A40", "TI", checkpoints_ti
    ),

    "Textual Inversion A40 batch18": (
        f"{base_scores}/textual_inversion/scores_TI_a40_batch18.csv",
        f"{base_energy}/textual_inversion/energy/outputs_a40_with_cp",
        [
            "0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
            "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
            "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
        ],
        "batch18", "A40", "TI", checkpoints_ti
    ),

    "Textual Inversion A40 batch6": (
        f"{base_scores}/textual_inversion/scores_TI_a40_batch6.csv",
        f"{base_energy}/textual_inversion/energy/outputs_a40_with_cp",
        [
            "0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
            "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
            "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
        ],
        "batch6", "A40", "TI", checkpoints_ti
    ),
}

technique_groups = {
    "LoRA": {
        k: v for k, v in series.items()
        if v[5] == "LoRA"
    },
    "Fine-tuning": {
        k: v for k, v in series.items()
        if v[5] == "FT"
    },
    "Textual Inversion": {
        k: v for k, v in series.items()
        if v[5] == "TI"
    },
}

technique_gpu_groups = {
    "LoRA_Titan": {
        k: v for k, v in series.items()
        if v[5] == "LoRA" and v[4] == "Titan"
    },

    "LoRA_A40": {
        k: v for k, v in series.items()
        if v[5] == "LoRA" and v[4] == "A40"
    },

    "Fine-tuning_Titan": {
        k: v for k, v in series.items()
        if v[5] == "FT" and v[4] == "Titan"
    },

    "Fine-tuning_A40": {
        k: v for k, v in series.items()
        if v[5] == "FT" and v[4] == "A40"
    },

    "Textual Inversion_Titan": {
        k: v for k, v in series.items()
        if v[5] == "TI" and v[4] == "Titan"
    },

    "Textual Inversion_A40": {
        k: v for k, v in series.items()
        if v[5] == "TI" and v[4] == "A40"
    },
}

# ----------------------------------------------------------------------
# FUNCIÓN DE CARGA
# ----------------------------------------------------------------------

def load_grouped(label, cfg):

    scores_csv, energy_root, train_names, batch_key, gpu, technique, checkpoints = cfg

    if not os.path.isfile(scores_csv):
        print(f"[WARNING] No existe {scores_csv}, se omite.")
        return None

    scores_df = pd.read_csv(scores_csv)

    energy_frames = []

    for train_name in train_names:

        fpath = os.path.join(
            energy_root,
            train_name,
            "energy_clean_per_checkpoint.csv"
        )

        if not os.path.isfile(fpath):
            print(f"[WARNING] No existe {fpath}, se omite.")
            continue

        edf = pd.read_csv(fpath)

        edf = edf[
            edf["global_step"].isin(checkpoints)
        ][
            ["global_step", "clean_energy_kwh"]
        ]

        edf["train_name"] = train_name

        energy_frames.append(edf)

    if not energy_frames:
        return None

    energy_df = pd.concat(
        energy_frames,
        ignore_index=True
    )

    merged = pd.merge(
        scores_df,
        energy_df,
        left_on=["train_name", "checkpoint"],
        right_on=["train_name", "global_step"]
    ).drop(
        columns=["global_step"]
    )

    grouped = merged.groupby(
        "checkpoint"
    ).agg(
        CLIP_T_mean=(
            "CLIP_T_score",
            "mean"
        ),

        KID_mean_avg=(
            "KID_mean",
            "mean"
        ),

        energy_mean_kwh=(
            "clean_energy_kwh",
            "mean"
        ),

    ).reset_index()

    return grouped


# ----------------------------------------------------------------------
# FUNCIÓN DE BUBBLE PLOT
#
# Dos subplots:
# izquierda = CLIP-T
# derecha   = KID
#
# Burbujas = energía en ese checkpoint
# Color     = batch
# Línea     = GPU
#
# legend_outside=True:
#     leyenda fuera de la figura, a la derecha
#
# legend_outside=False:
#     leyenda dentro del subplot izquierdo
# ----------------------------------------------------------------------

def plot_bubble(
    group_dict,
    title,
    output_path,
    legend_outside=False
):

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(25, 6.5),
        sharex=True
    )

    plt.subplots_adjust(
        wspace=0.1,
        bottom=0.22
    )

    for label, cfg in group_dict.items():

        batch_key = cfg[3]
        gpu       = cfg[4]

        color = batch_colors[batch_key]
        ls    = gpu_ls[gpu]

        grouped = load_grouped(
            label,
            cfg
        )

        if grouped is None:
            continue

        steps = grouped[
            "checkpoint"
        ].values

        clip_t = grouped[
            "CLIP_T_mean"
        ].values

        kid = grouped[
            "KID_mean_avg"
        ].values

        energy = grouped[
            "energy_mean_kwh"
        ].values

        sizes = energy * BUBBLE_SCALE

        # --------------------------------------------------------------
        # CLIP-T
        # --------------------------------------------------------------

        ax_left.plot(
            steps,
            clip_t,
            color=color,
            linestyle=ls,
            linewidth=1.5,
            alpha=0.7,
            zorder=1
        )

        ax_left.scatter(
            steps,
            clip_t,
            s=sizes,
            color=color,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
            label=label
        )

        # --------------------------------------------------------------
        # KID
        # --------------------------------------------------------------

        ax_right.plot(
            steps,
            kid,
            color=color,
            linestyle=ls,
            linewidth=1.5,
            alpha=0.7,
            zorder=1
        )

        ax_right.scatter(
            steps,
            kid,
            s=sizes,
            color=color,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            zorder=2
        )

    # ------------------------------------------------------------------
    # EJES
    # ------------------------------------------------------------------

    ax_left.set_ylabel(
        "CLIP-T",
        fontsize=18
    )

    ax_right.set_ylabel(
        "KID mean",
        fontsize=18
    )

    ax_left.set_xlabel(
        "Steps",
        fontsize=18
    )

    ax_right.set_xlabel(
        "Steps",
        fontsize=18
    )

    # ------------------------------------------------------------------
    # LEYENDA
    # ------------------------------------------------------------------

    handles, labels = ax_left.get_legend_handles_labels()

    if legend_outside:

        fig.legend(
            handles,
            labels,
            fontsize=18,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(labels),
            labelspacing=1.2,
            columnspacing=1.8,
            handletextpad=0.8
        )

    else:

        ax_left.legend(
            handles,
            labels,
            fontsize=18,
            loc="upper right",
            labelspacing=1.5
        )

    # ------------------------------------------------------------------
    # TÍTULO
    # ------------------------------------------------------------------

    fig.suptitle(
        f"{title}\n(bubble size = energy)",
        fontsize=18,
        y=0.98
    )

    plt.tight_layout()

    # ------------------------------------------------------------------
    # GUARDAR
    # ------------------------------------------------------------------

    svg_path = output_path.replace(
        ".png",
        ".svg"
    )

    plt.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight"
    )

    print(
        f"Gráfica guardada en {svg_path}"
    )

    plt.show()


# ----------------------------------------------------------------------
# GENERAR GRÁFICAS CONJUNTAS POR TÉCNICA
#
# Aquí la leyenda se coloca FUERA
# ----------------------------------------------------------------------

for technique, group_dict in technique_groups.items():

    plot_bubble(
        group_dict,
        title=f"CLIP-T and KID vs Steps — {technique}",
        output_path=os.path.join(
            output_dir,
            f"bubble_{technique}.png"
        ),
        legend_outside=True
    )


# ----------------------------------------------------------------------
# GENERAR GRÁFICAS POR TÉCNICA + GPU
#
# Aquí la leyenda se mantiene DENTRO
# ----------------------------------------------------------------------

for group_key, group_dict in technique_gpu_groups.items():

    technique, gpu = group_key.split(
        "_",
        1
    )

    plot_bubble(
        group_dict,
        title=f"CLIP-T and KID vs Steps — {technique} {gpu}",
        output_path=os.path.join(
            output_dir,
            f"bubble_{group_key}.png"
        ),
        legend_outside=False
    )