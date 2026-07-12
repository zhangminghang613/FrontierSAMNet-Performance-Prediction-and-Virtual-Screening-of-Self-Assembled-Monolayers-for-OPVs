import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sam_core import ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    with open("data/processed/feature_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    df = pd.read_csv("data/processed/sam_clean.csv")
    rows = []
    cols = manifest["descriptors"] + manifest["numeric_context"] + manifest["bits"]
    limit = 350 if args.smoke else len(cols)
    for c in cols[:limit]:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.nunique(dropna=True) < 2:
            continue
        for t in manifest["targets"]:
            y = pd.to_numeric(df[t], errors="coerce")
            v = x.corr(y, method="spearman")
            if pd.notna(v):
                rows.append({"feature": c, "target": t, "score": float(abs(v)), "signed_score": float(v)})
    imp = pd.DataFrame(rows).sort_values(["target", "score"], ascending=[True, False])
    imp.to_csv("results/tables/feature_importance.csv", index=False)
    proc = []
    for c in manifest["numeric_context"]:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() > 20:
            q = pd.qcut(x.rank(method="first"), 8, labels=False, duplicates="drop")
            g = df.groupby(q)["pce"].mean()
            for k, v in g.items():
                proc.append({"feature": c, "bin": int(k), "pce": float(v)})
    pd.DataFrame(proc).to_csv("results/tables/process_response.csv", index=False)
    print({"features_ranked": len(imp)})


if __name__ == "__main__":
    main()
