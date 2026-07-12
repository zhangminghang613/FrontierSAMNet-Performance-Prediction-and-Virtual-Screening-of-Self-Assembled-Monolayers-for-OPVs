import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sam_core import (
    FrontierSAMNet,
    SAMDataset,
    SmilesTokenizer,
    batch_to_device,
    device_from_config,
    ensure_dirs,
    metrics_frame,
    set_seed,
)


def restore_prep(d):
    return {
        "desc_mu": np.array(d["desc_mu"], dtype=np.float32),
        "desc_sd": np.array(d["desc_sd"], dtype=np.float32),
        "num_mu": np.array(d["num_mu"], dtype=np.float32),
        "num_sd": np.array(d["num_sd"], dtype=np.float32),
        "target_mu": np.array(d["target_mu"], dtype=np.float32),
        "target_sd": np.array(d["target_sd"], dtype=np.float32),
        "cat_maps": d["cat_maps"],
    }


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_trained_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find {model_path}. Run 04_train_multimodal_model_train_only.py first.")

    
    ck_cpu = safe_torch_load(model_path, torch.device("cpu"))
    config = ck_cpu["config"]
    device = device_from_config(config)

    ck = safe_torch_load(model_path, device)
    manifest = ck["manifest"]
    prep = restore_prep(ck["preprocessor"])

    if Path("models/smiles_tokenizer.json").exists():
        tok = SmilesTokenizer.load("models/smiles_tokenizer.json")
    else:
        tok = SmilesTokenizer(vocab=ck["vocab"], max_len=config["model"]["max_smiles_length"])

    cat_cards = [len(prep["cat_maps"][c]) for c in manifest["categorical_context"]]
    model = FrontierSAMNet(
        config,
        len(tok.vocab),
        len(manifest["descriptors"]),
        len(manifest["numeric_context"]),
        len(manifest["bits"]),
        cat_cards,
    ).to(device)

    model.load_state_dict(ck["model"])
    model.eval()
    return model, config, manifest, prep, tok, device


def evaluate(model, loader, device, prep, manifest, mc=1):
    model.eval()
    preds = []
    trues = []
    rows = []
    gates = []

    if mc > 1:
        model.train()

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            draws = []
            aux0 = None

            for _ in range(mc):
                pred, aux = model(batch, aux=True)
                draws.append(pred.detach().cpu().numpy())
                aux0 = aux

            arr = np.stack(draws, 0)
            mean = arr.mean(0)
            std = arr.std(0)
            raw = mean * prep["target_sd"] + prep["target_mu"]
            raw_std = std * prep["target_sd"]

            preds.append(raw)
            trues.append(batch["target_raw"].detach().cpu().numpy())

            rid = batch["row_id"].detach().cpu().numpy()
            for i in range(len(rid)):
                item = {"row_id": int(rid[i])}
                for j, t in enumerate(manifest["targets"]):
                    item[f"pred_{t}"] = float(raw[i, j])
                    item[f"unc_{t}"] = float(raw_std[i, j])
                rows.append(item)

            g = aux0["gates"].detach().cpu().numpy()
            for i in range(len(rid)):
                gates.append(
                    {
                        "row_id": int(rid[i]),
                        "graph_gate": float(g[i, 0]),
                        "smiles_gate": float(g[i, 1]),
                        "tabular_gate": float(g[i, 2]),
                    }
                )

    model.eval()
    return np.vstack(trues), np.vstack(preds), pd.DataFrame(rows), pd.DataFrame(gates)


def _coverage(abs_error, sigma, multiplier):
    valid = np.isfinite(abs_error) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(valid):
        return np.nan
    return float((abs_error[valid] <= multiplier * sigma[valid]).mean())


def calibrate_uncertainty(predictions, targets, calibration_split="val", min_scale=1.0):
    out = predictions.copy()
    summary_rows = []

    for target in targets:
        pred_col = f"pred_{target}"
        y_col = target
        unc_col = f"unc_{target}"
        raw_unc_col = f"raw_unc_{target}"

        if unc_col not in out.columns:
            continue

        out[raw_unc_col] = out[unc_col].astype(float)

        cal = out[out["split"] == calibration_split].copy()
        required = [pred_col, y_col, raw_unc_col]
        if not all(c in cal.columns for c in required):
            scale = 1.0
            n = 0
            abs_error = np.array([], dtype=float)
            raw_sigma = np.array([], dtype=float)
        else:
            abs_error = (cal[pred_col].astype(float) - cal[y_col].astype(float)).abs().to_numpy()
            raw_sigma = cal[raw_unc_col].astype(float).to_numpy()
            valid = np.isfinite(abs_error) & np.isfinite(raw_sigma) & (raw_sigma > 1e-8)
            n = int(valid.sum())

            if n:
                scale = float(np.sqrt(np.mean((abs_error[valid] / raw_sigma[valid]) ** 2)))
                scale = max(scale, float(min_scale))
            else:
                scale = 1.0

        out[unc_col] = out[raw_unc_col].astype(float) * scale
        cal_sigma = raw_sigma * scale

        summary_rows.append(
            {
                "target": target,
                "calibration_split": calibration_split,
                "n": n,
                "uncertainty_scale": scale,
                "raw_mean_uncertainty": float(np.nanmean(raw_sigma)) if n else np.nan,
                "calibrated_mean_uncertainty": float(np.nanmean(cal_sigma)) if n else np.nan,
                "calibration_mae": float(np.nanmean(abs_error)) if n else np.nan,
                "raw_coverage_1sigma": _coverage(abs_error, raw_sigma, 1.0),
                "calibrated_coverage_1sigma": _coverage(abs_error, cal_sigma, 1.0),
                "raw_coverage_2sigma": _coverage(abs_error, raw_sigma, 2.0),
                "calibrated_coverage_2sigma": _coverage(abs_error, cal_sigma, 2.0),
            }
        )

    return out, pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description="Predict/evaluate with a trained FrontierSAMNet checkpoint only.")
    parser.add_argument("--model-path", default="models/frontier_sam_net.pt", help="Path to trained model checkpoint.")
    parser.add_argument("--input-csv", default="data/processed/sam_clean.csv", help="Input CSV with train/val/test splits.")
    parser.add_argument("--output-dir", default="results/tables", help="Directory for prediction and metric CSV files.")
    parser.add_argument("--mc-dropout-passes", type=int, default=None, help="Override MC-dropout passes. Default uses checkpoint config.")
    parser.add_argument("--calibration-split", default="val", choices=["train", "val", "test"], help="Split used to calibrate MC-dropout uncertainty scale.")
    parser.add_argument("--disable-uncertainty-calibration", action="store_true", help="Keep raw MC-dropout uncertainties without validation-set scaling.")
    parser.add_argument("--smoke", action="store_true", help="Use 4 MC-dropout passes for a quick prediction test.")
    args = parser.parse_args()

    ensure_dirs()
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, manifest, prep, tok, device = load_trained_model(model_path)
    set_seed(config["seed"])

    df = pd.read_csv(args.input_csv)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)

    datasets = {
        "train": (train, SAMDataset(train, manifest, prep, config, tok)),
        "val": (val, SAMDataset(val, manifest, prep, config, tok)),
        "test": (test, SAMDataset(test, manifest, prep, config, tok)),
    }

    loaders = {
        name: DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for name, (_, ds) in datasets.items()
    }

    if args.mc_dropout_passes is not None:
        mc = args.mc_dropout_passes
    elif args.smoke:
        mc = 4
    else:
        mc = config["training"]["mc_dropout_passes"]

    all_frames = []
    all_gates = []
    metric_frames = []

    for name in ["train", "val", "test"]:
        data, _ = datasets[name]
        loader = loaders[name]
        y, p, pred_df, gate_df = evaluate(model, loader, device, prep, manifest, mc=mc)

        pred_df["split"] = name
        pred_df = pred_df.merge(
            data[["row_id", "name", "SMILES", "doi"] + manifest["targets"]],
            on="row_id",
            how="left",
        )

        gate_df["split"] = name
        all_frames.append(pred_df)
        all_gates.append(gate_df)

        mf = metrics_frame(y, p, manifest["targets"])
        mf["split"] = name
        metric_frames.append(mf)

    predictions = pd.concat(all_frames).reset_index(drop=True)
    uncertainty_calibration = pd.DataFrame()
    if not args.disable_uncertainty_calibration:
        predictions, uncertainty_calibration = calibrate_uncertainty(
            predictions,
            manifest["targets"],
            calibration_split=args.calibration_split,
        )

    predictions_path = output_dir / "predictions_multimodal.csv"
    gates_path = output_dir / "modality_gates.csv"
    metrics_path = output_dir / "metrics_multimodal.csv"
    calibration_path = output_dir / "uncertainty_calibration.csv"

    predictions.to_csv(predictions_path, index=False)
    pd.concat(all_gates).to_csv(gates_path, index=False)
    pd.concat(metric_frames).to_csv(metrics_path, index=False)
    if not uncertainty_calibration.empty:
        uncertainty_calibration.to_csv(calibration_path, index=False)

    print(
        {
            "model_path": str(model_path),
            "mc_dropout_passes": int(mc),
            "uncertainty_calibration": None if uncertainty_calibration.empty else str(calibration_path),
            "predictions": str(predictions_path),
            "modality_gates": str(gates_path),
            "metrics": str(metrics_path),
        }
    )


if __name__ == "__main__":
    main()
