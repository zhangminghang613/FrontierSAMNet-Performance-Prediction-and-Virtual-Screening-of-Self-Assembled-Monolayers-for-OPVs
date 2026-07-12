from pathlib import Path

import numpy as np

from sam_core import build_clean_frame, ensure_dirs, load_config, save_json, split_by_group


def main():
    config = load_config()
    ensure_dirs()
    df, manifest = build_clean_frame(config)
    split = split_by_group(df, config)
    df["split"] = split
    
    audit = df.copy()
    audit["pce_from_voc_jsc_ff"] = audit["voc"] * audit["jsc"] * audit["ff"] / 100.0
    audit["pce_abs_error"] = (audit["pce"] - audit["pce_from_voc_jsc_ff"]).abs()

    audit["target_flag"] = (
        (audit["pce"] > 25)
        | (audit["jsc"] > 60)
        | (audit["voc"] > 1.5)
        | (audit["ff"] > 90)
        | (audit["pce_abs_error"] > 5)
    )

    audit_cols = [
        "name",
        "SMILES",
        "doi",
        "pce",
        "voc",
        "jsc",
        "ff",
        "pce_from_voc_jsc_ff",
        "pce_abs_error",
        "target_flag",
    ]

    audit[audit_cols].to_csv(
        "results/tables/target_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    flagged = audit[audit["target_flag"]]
    if len(flagged):
        print("Warning: suspicious target rows found.")
        print(flagged[audit_cols].sort_values("pce", ascending=False).head(20))
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/sam_clean.csv", index=False)
    for part in ["train", "val", "test"]:
        df[df["split"] == part].to_csv(f"data/processed/{part}.csv", index=False)
    summary = {
        "rows": int(len(df)),
        "unique_smiles": int(df["SMILES"].nunique()),
        "unique_sam_group": int(df["sam_group"].nunique()) if "sam_group" in df else 0,
        "unique_names": int(df["name"].nunique()) if "name" in df else 0,
        "unique_doi": int(df["doi"].nunique()) if "doi" in df else 0,
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "split_group_counts": {
            part: int(df.loc[df["split"] == part, config["split"]["group_column"]].nunique())
            for part in ["train", "val", "test"]
        },
        "target_mean_by_split": {
            part: {
                t: float(df.loc[df["split"] == part, t].mean())
                for t in manifest["targets"]
            }
            for part in ["train", "val", "test"]
        },
        "feature_counts": {k: len(v) for k, v in manifest.items() if isinstance(v, list)}
    }
    save_json(manifest, "data/processed/feature_manifest.json")
    save_json(summary, "data/processed/data_summary.json")


if __name__ == "__main__":
    main()
