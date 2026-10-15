import os
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------

base = "/home/helena/Desktop/tfm/images_and_scores/sdv15"
base_subject = "/home/helena/Desktop/tfm/images_and_scores/subject"

output_dir = "results/slopes"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------------------
# DEFINICIÓN DE TODOS LOS EXPERIMENTOS
# (group, label, root, train_names)
# ----------------------------------------------------------------------

experiments = [

    # ── FIGURA 1: Técnica (batch6 Titan / batch18 A40) ──────────────
    ("technique", "FT Titan batch6",
     f"{base}/finetuning/energy/outputs_titan_with_cp",
     ["0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
      "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
      "2_sdv15_finetuning_res512_Battista_steps5000_seed1337"]),

    ("technique", "LoRA Titan batch6",
     f"{base}/lora/energy/outputs_titan_with_cp",
     ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
      "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
      "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6"]),

    ("technique", "TI Titan batch6",
     f"{base}/textual_inversion/energy/outputs_titan_with_cp_50",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6"]),

    ("technique", "FT A40 batch18",
     f"{base}/finetuning/energy/outputs_a40_with_cp",
     ["0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
      "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
      "2_sdv15_finetuning_res512_Battista_steps5000_seed1337"]),

    ("technique", "LoRA A40 batch18",
     f"{base}/lora/energy/outputs_a40_with_cp",
     ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18",
      "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18",
      "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18"]),

    ("technique", "TI A40 batch18",
     f"{base}/textual_inversion/energy/outputs_a40_with_cp",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18"]),

    # ── FIGURA 2: Iter/Batch ─────────────────────────────────────────
    ("iter_batch", "LoRA Titan batch12",
     f"{base}/lora/energy/outputs_titan_with_cp",
     ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
      "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
      "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337"]),

    ("iter_batch", "LoRA Titan batch6",
     f"{base}/lora/energy/outputs_titan_with_cp",
     ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
      "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
      "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6"]),

    ("iter_batch", "TI Titan batch12",
     f"{base}/textual_inversion/energy/outputs_titan_with_cp_50",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337"]),

    ("iter_batch", "TI Titan batch6",
     f"{base}/textual_inversion/energy/outputs_titan_with_cp_50",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6"]),

    ("iter_batch", "LoRA A40 batch24",
     f"{base}/lora/energy/outputs_a40_with_cp",
     ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
      "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
      "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337"]),

    ("iter_batch", "LoRA A40 batch18",
     f"{base}/lora/energy/outputs_a40_with_cp",
     ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18",
      "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18",
      "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch18"]),

    ("iter_batch", "LoRA A40 batch6",
     f"{base}/lora/energy/outputs_a40_with_cp",
     ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
      "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
      "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6"]),

    ("iter_batch", "FT A40 batch18",
     f"{base}/finetuning/energy/outputs_a40_with_cp",
     ["0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
      "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
      "2_sdv15_finetuning_res512_Battista_steps5000_seed1337"]),

    ("iter_batch", "TI A40 batch24",
     f"{base}/textual_inversion/energy/outputs_a40_with_cp",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337"]),

    ("iter_batch", "TI A40 batch18",
     f"{base}/textual_inversion/energy/outputs_a40_with_cp",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch18"]),

    ("iter_batch", "TI A40 batch6",
     f"{base}/textual_inversion/energy/outputs_a40_with_cp",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6"]),

    # ── FIGURA 3: Tarea ──────────────────────────────────────────────
    ("task", "TI A40 batch6 Style Piranesi",
     f"{base}/textual_inversion/energy/outputs_a40_isolated_gpu",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337"]),

    ("task", "TI A40 batch6 Subject cat_statue",
     f"{base_subject}/energy/outputs_a40_isolated_gpu/cat_statue",
     ["0_sdv15_textual_inversion_res512_cat_statue_steps5000_seed1337",
      "1_sdv15_textual_inversion_res512_cat_statue_steps5000_seed1337",
      "2_sdv15_textual_inversion_res512_cat_statue_steps5000_seed1337"]),

    ("task", "TI A40 batch6 Subject dog6",
     f"{base_subject}/energy/outputs_a40_isolated_gpu/dog6",
     ["0_sdv15_textual_inversion_res512_dog6_steps5000_seed1337",
      "1_sdv15_textual_inversion_res512_dog6_steps5000_seed1337",
      "2_sdv15_textual_inversion_res512_dog6_steps5000_seed1337"]),

    # ── FIGURA 4: GPU ────────────────────────────────────────────────
    ("gpu", "LoRA Titan batch6",
     f"{base}/lora/energy/outputs_titan_with_cp",
     ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
      "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
      "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6"]),

    ("gpu", "LoRA A40 batch6",
     f"{base}/lora/energy/outputs_a40_with_cp",
     ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
      "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6",
      "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337_batch6"]),

    ("gpu", "TI Titan batch6",
     f"{base}/textual_inversion/energy/outputs_titan_with_cp_50",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6"]),

    ("gpu", "TI A40 batch6",
     f"{base}/textual_inversion/energy/outputs_a40_with_cp",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6"]),

    # ── FIGURA 5: Sistema ────────────────────────────────────────────
    ("system", "TI A40 batch6 Dedicated-GPU",
     f"{base}/textual_inversion/energy/outputs_a40_with_cp",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337_batch6"]),

    ("system", "TI A40 batch6 Multi-GPU",
     f"{base}/textual_inversion/energy/outputs_a40_multi_gpu",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337"]),

    # ── FIGURA 6: Memory budget (default batch) ──────────────────────
    ("memory_budget", "LoRA Titan batch12",
     f"{base}/lora/energy/outputs_titan_with_cp",
     ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
      "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
      "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337"]),

    ("memory_budget", "FT Titan batch6",
     f"{base}/finetuning/energy/outputs_titan_with_cp",
     ["0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
      "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
      "2_sdv15_finetuning_res512_Battista_steps5000_seed1337"]),

    ("memory_budget", "TI Titan batch12",
     f"{base}/textual_inversion/energy/outputs_titan_with_cp_50",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337"]),

    ("memory_budget", "LoRA A40 batch24",
     f"{base}/lora/energy/outputs_a40_with_cp",
     ["0_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
      "1_sdv15_lora_rank8_res512_Battista_steps5000_seed1337",
      "2_sdv15_lora_rank8_res512_Battista_steps5000_seed1337"]),

    ("memory_budget", "FT A40 batch18",
     f"{base}/finetuning/energy/outputs_a40_with_cp",
     ["0_sdv15_finetuning_res512_Battista_steps5000_seed1337",
      "1_sdv15_finetuning_res512_Battista_steps5000_seed1337",
      "2_sdv15_finetuning_res512_Battista_steps5000_seed1337"]),

    ("memory_budget", "TI A40 batch24",
     f"{base}/textual_inversion/energy/outputs_a40_with_cp",
     ["0_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "1_sdv15_textual_inversion_res512_Battista_steps5000_seed1337",
      "2_sdv15_textual_inversion_res512_Battista_steps5000_seed1337"]),
]

# ----------------------------------------------------------------------
# FUNCIÓN: calcular pendiente (kWh/step) con regresión lineal
# ----------------------------------------------------------------------

def compute_slope(root, train_names):
    frames = []
    for train_name in train_names:
        fpath = os.path.join(root, train_name, "energy_clean_per_checkpoint.csv")
        if not os.path.isfile(fpath):
            print(f"  [WARNING] No existe {fpath}")
            continue
        df = pd.read_csv(fpath)[["global_step", "clean_energy_kwh"]]
        df["train_name"] = train_name
        frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    grouped = df.groupby("global_step")["clean_energy_kwh"].mean().reset_index()

    slope, intercept, r, p, se = stats.linregress(
        grouped["global_step"], grouped["clean_energy_kwh"]
    )
    return {
        "slope_kwh_per_step": slope,
        "slope_kwh_per_1000steps": slope * 1000,
        "intercept": intercept,
        "r_squared": r**2,
        "n_points": len(grouped),
    }

# ----------------------------------------------------------------------
# CALCULAR PENDIENTES
# ----------------------------------------------------------------------

records = []
for group, label, root, train_names in experiments:
    print(f"  {group} | {label}")
    result = compute_slope(root, train_names)
    if result is None:
        continue
    records.append({
        "group":                  group,
        "label":                  label,
        "slope_kwh_per_step":     result["slope_kwh_per_step"],
        "slope_kwh_per_1000steps": result["slope_kwh_per_1000steps"],
        "intercept":              result["intercept"],
        "r_squared":              result["r_squared"],
        "n_points":               result["n_points"],
    })

df_out = pd.DataFrame(records)

# ----------------------------------------------------------------------
# GUARDAR: un CSV global y uno por grupo
# ----------------------------------------------------------------------

global_path = os.path.join(output_dir, "slopes_all.csv")
df_out.to_csv(global_path, index=False)
print(f"\nGuardado: {global_path}")

for group in df_out["group"].unique():
    subset = df_out[df_out["group"] == group]
    path = os.path.join(output_dir, f"slopes_{group}.csv")
    subset.to_csv(path, index=False)
    print(f"Guardado: {path}")

print("\n=== RESUMEN ===")
print(df_out[["group", "label", "slope_kwh_per_1000steps", "r_squared"]].to_string(index=False))
