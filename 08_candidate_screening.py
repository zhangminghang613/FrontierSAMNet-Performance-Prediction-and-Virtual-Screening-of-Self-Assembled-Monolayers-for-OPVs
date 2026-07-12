from pathlib import Path

import numpy as np
import pandas as pd

from sam_core import ensure_dirs


TARGETS = ["pce", "voc", "jsc", "ff"]


def main():
    ensure_dirs()

    pred_path = Path("results/tables/predictions_multimodal.csv")

    if pred_path.exists():
        pred = pd.read_csv(pred_path)
    else:
        pred = pd.read_csv("data/processed/sam_clean.csv")

        for t in TARGETS:
            pred[f"pred_{t}"] = pred[t]
            pred[f"unc_{t}"] = pred[t].std() * 0.1

    required_cols = [
        "name",
        "SMILES",
        "pred_pce",
        "pred_voc",
        "pred_jsc",
        "pred_ff",
    ]

    for c in required_cols:
        if c not in pred.columns:
            raise ValueError(f"Missing required column: {c}")

    
    for t in TARGETS:
        unc_col = f"unc_{t}"

        if unc_col not in pred.columns:
            pred[unc_col] = pred[f"pred_{t}"].std() * 0.1

    
    
    group = pred.groupby(["name", "SMILES"], dropna=False)

    rows = []

    global_mean_voc = float(pred["pred_voc"].mean())
    global_mean_ff = float(pred["pred_ff"].mean())

    for (name, smiles), g in group:
        pce = float(g["pred_pce"].mean())
        unc_pce = float(g["unc_pce"].mean())

        voc = float(g["pred_voc"].mean())
        jsc = float(g["pred_jsc"].mean())
        ff = float(g["pred_ff"].mean())

        unc_voc = float(g["unc_voc"].mean())
        unc_jsc = float(g["unc_jsc"].mean())
        unc_ff = float(g["unc_ff"].mean())

        
        robust_score_old = (
            pce
            - 0.75 * unc_pce
            + 0.2 * (voc - global_mean_voc)
            + 0.03 * (ff - global_mean_ff)
        )

        
        pce_lcb_2sigma = pce - 2.0 * unc_pce
        pce_ucb_2sigma = pce + 2.0 * unc_pce

        rows.append(
            {
                "name": name,
                "SMILES": smiles,
                "mean_pred_pce": pce,
                "uncertainty_pce": unc_pce,
                "pce_lcb_2sigma": pce_lcb_2sigma,
                "pce_ucb_2sigma": pce_ucb_2sigma,
                "mean_pred_voc": voc,
                "uncertainty_voc": unc_voc,
                "mean_pred_jsc": jsc,
                "uncertainty_jsc": unc_jsc,
                "mean_pred_ff": ff,
                "uncertainty_ff": unc_ff,
                "robust_score_old": robust_score_old,
                "robust_score": pce_lcb_2sigma,
                "observations": int(len(g)),
            }
        )

    out = pd.DataFrame(rows)

    
    out = out.sort_values(
        "pce_lcb_2sigma",
        ascending=False,
    ).reset_index(drop=True)

    out["rank"] = np.arange(1, len(out) + 1)

    
    keep_cols = [
        "rank",
        "name",
        "SMILES",
        "observations",
        "mean_pred_pce",
        "uncertainty_pce",
        "pce_lcb_2sigma",
        "pce_ucb_2sigma",
        "mean_pred_voc",
        "uncertainty_voc",
        "mean_pred_jsc",
        "uncertainty_jsc",
        "mean_pred_ff",
        "uncertainty_ff",
        "robust_score",
        "robust_score_old",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]

    out = out[keep_cols]

if __name__ == "__main__":
    main()
