import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sam_core import ensure_dirs, load_config, metrics_frame


def feature_blocks_raw(df, manifest):
    frames = []
    block_cols = {}

    if manifest["descriptors"]:
        desc = (
            df[manifest["descriptors"]]
            .apply(pd.to_numeric, errors="coerce")
            .add_prefix("desc_")
        )
        frames.append(desc)
        block_cols["descriptors"] = list(desc.columns)

    if manifest["bits"]:
        bits = (
            df[manifest["bits"]]
            .apply(pd.to_numeric, errors="coerce")
            .add_prefix("bit_")
        )
        frames.append(bits)
        block_cols["bits"] = list(bits.columns)

    process_frames = []

    if manifest["numeric_context"]:
        num = (
            df[manifest["numeric_context"]]
            .apply(pd.to_numeric, errors="coerce")
            .add_prefix("num_")
        )
        process_frames.append(num)

    if manifest["categorical_context"]:
        cat = pd.get_dummies(
            df[manifest["categorical_context"]]
            .fillna("missing")
            .astype(str),
            dummy_na=False,
        ).add_prefix("cat_")
        process_frames.append(cat)

    if process_frames:
        process = pd.concat(process_frames, axis=1)
        frames.append(process)
        block_cols["process"] = list(process.columns)

    if not frames:
        return pd.DataFrame(index=df.index), block_cols

    X = pd.concat(frames, axis=1)

    return X, block_cols

def feature_blocks(df, manifest, keep):

    x_all, block_cols = feature_blocks_raw(df, manifest)

    selected_cols = []

    for block in ["descriptors", "bits", "process"]:
        if block in keep:
            selected_cols.extend(block_cols.get(block, []))

    if len(selected_cols) == 0:
        return pd.DataFrame(index=df.index)

    x = x_all.reindex(columns=selected_cols)

    x = (
        x
        .replace([np.inf, -np.inf], np.nan)
        .fillna(x.median(numeric_only=True))
        .fillna(0)
    )

    return x

def prepare_full_features(train, test, manifest):


    x_train, block_cols_train = feature_blocks_raw(train, manifest)
    x_test, block_cols_test = feature_blocks_raw(test, manifest)

    
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    
    med = (
        x_train
        .replace([np.inf, -np.inf], np.nan)
        .median(numeric_only=True)
    )

    x_train = (
        x_train
        .replace([np.inf, -np.inf], np.nan)
        .fillna(med)
        .fillna(0)
    )

    x_test = (
        x_test
        .replace([np.inf, -np.inf], np.nan)
        .fillna(med)
        .fillna(0)
    )

    
    block_cols = {}
    for block, cols in block_cols_train.items():
        block_cols[block] = [c for c in cols if c in x_train.columns]

    return x_train, x_test, block_cols, med

def align(train_x, other_x):
    return train_x, other_x.reindex(columns=train_x.columns, fill_value=0)


def train_baselines(train, test, manifest, smoke):
    try:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
        from sklearn.linear_model import ElasticNet
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import ElasticNet
        from sklearn.svm import SVR
        from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor

    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    y_train = train[manifest["targets"]].to_numpy(float)
    y_test = test[manifest["targets"]].to_numpy(float)
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=80 if smoke else 200, random_state=17, n_jobs=-1, min_samples_leaf=2),
        "ExtraTrees": ExtraTreesRegressor(
    n_estimators=80 if smoke else 350,
    random_state=19,
    n_jobs=-1,
    min_samples_leaf=3,
    max_depth=12,
    max_features=0.6,
    min_samples_split=6,
),
        "HistGB": MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=80 if smoke else 200, learning_rate=0.045, l2_regularization=0.02, random_state=23)),
        "ElasticNet": make_pipeline(StandardScaler(with_mean=False), MultiOutputRegressor(ElasticNet(alpha=0.004, l1_ratio=0.25, max_iter=6000, random_state=29)))
    }
    x_train = feature_blocks(train, manifest, {"descriptors", "bits", "process"})
    x_test = feature_blocks(test, manifest, {"descriptors", "bits", "process"})
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    x_train, x_test = align(x_train, x_test)
    metrics = []
    preds = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        p = model.predict(x_test)
        mf = metrics_frame(y_test, p, manifest["targets"])
        mf["model"] = name
        metrics.append(mf)
        block = test[["row_id", "name", "SMILES", "doi"]].copy()
        block["model"] = name
        for j, t in enumerate(manifest["targets"]):
            block[f"pred_{t}"] = p[:, j]
            block[t] = y_test[:, j]
        preds.append(block)
    return pd.concat(metrics), pd.concat(preds)


def export_frontiersamnet_all_split_predictions(manifest):


    src = Path("results/tables/predictions_multimodal.csv")
    dst = Path("results/tables/frontiersamnet_all_split_predictions.csv")

    if not src.exists():
        print("Figure 9 all-split export skipped: results/tables/predictions_multimodal.csv not found.")
        return pd.DataFrame()

    pred = pd.read_csv(src)

    targets = manifest.get("targets", [])
    required = ["split"]
    for t in targets:
        required.extend([t, f"pred_{t}"])

    missing = [c for c in required if c not in pred.columns]
    if missing:
        print(f"Figure 9 all-split export warning: missing columns {missing}; table was not exported.")
        return pred

    keep_cols = []
    for c in ["row_id", "split", "name", "SMILES", "doi", "sam_group"]:
        if c in pred.columns:
            keep_cols.append(c)

    for t in targets:
        keep_cols.extend([t, f"pred_{t}"])
        unc_col = f"unc_{t}"
        if unc_col in pred.columns:
            keep_cols.append(unc_col)

    out = pred[keep_cols].copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)

    split_counts = (
        out["split"].astype(str).value_counts().to_dict()
        if "split" in out.columns else {}
    )

    print({
        "all_split_predictions": str(dst),
        "rows": int(len(out)),
        "split_counts": split_counts,
    })

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    config = load_config()
    with open("data/processed/feature_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    df = pd.read_csv("data/processed/sam_clean.csv")
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    metrics, preds = train_baselines(train, test, manifest, args.smoke)
    if len(metrics):
        metrics.to_csv("results/tables/baseline_metrics.csv", index=False)
        preds.to_csv("results/tables/baseline_predictions.csv", index=False)
    all_split_pred = export_frontiersamnet_all_split_predictions(manifest)

    print({"baseline_models": int(metrics["model"].nunique()) if len(metrics) else 0})


if __name__ == "__main__":
    main()
