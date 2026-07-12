import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from sam_core import (
    FrontierSAMNet,
    SAMDataset,
    SmilesTokenizer,
    batch_to_device,
    device_from_config,
    ensure_dirs,
    load_config,
    make_category_maps,
    metrics_frame,
    multitask_loss,
    save_json,
    set_seed,
    standardize_fit,
)


def smoke_config(config):
    d = json.loads(json.dumps(config))
    d["model"]["hidden_dim"] = 96
    d["model"]["graph_layers"] = 2
    d["model"]["smiles_layers"] = 2
    d["model"]["tabular_layers"] = 1
    d["model"]["fusion_layers"] = 1
    d["model"]["heads"] = 4
    d["model"]["experts"] = 2
    d["model"]["max_atoms"] = 96
    d["model"]["max_smiles_length"] = 128
    d["training"]["batch_size"] = 32
    return d


def build_prep(train, manifest):
    desc = (
        train[manifest["descriptors"]].to_numpy(np.float64)
        if manifest["descriptors"]
        else np.zeros((len(train), 0), dtype=np.float32)
    )
    num = (
        train[manifest["numeric_context"]].to_numpy(np.float64)
        if manifest["numeric_context"]
        else np.zeros((len(train), 0), dtype=np.float32)
    )
    y = train[manifest["targets"]].to_numpy(np.float64)

    desc_mu, desc_sd = (
        standardize_fit(desc)
        if desc.shape[1]
        else (np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32))
    )
    num_mu, num_sd = (
        standardize_fit(num)
        if num.shape[1]
        else (np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32))
    )
    target_mu, target_sd = standardize_fit(y)

    return {
        "desc_mu": desc_mu,
        "desc_sd": desc_sd,
        "num_mu": num_mu,
        "num_sd": num_sd,
        "target_mu": target_mu,
        "target_sd": target_sd,
        "cat_maps": make_category_maps(train, manifest["categorical_context"]),
    }


def prep_to_json(prep):
    return {
        "desc_mu": prep["desc_mu"].tolist(),
        "desc_sd": prep["desc_sd"].tolist(),
        "num_mu": prep["num_mu"].tolist(),
        "num_sd": prep["num_sd"].tolist(),
        "target_mu": prep["target_mu"].tolist(),
        "target_sd": prep["target_sd"].tolist(),
        "cat_maps": prep["cat_maps"],
    }


def evaluate_for_validation(model, loader, device, prep, manifest):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            pred = model(batch)
            pred_np = pred.detach().cpu().numpy()
            raw_pred = pred_np * prep["target_sd"] + prep["target_mu"]
            preds.append(raw_pred)
            trues.append(batch["target_raw"].detach().cpu().numpy())

    return np.vstack(trues), np.vstack(preds)


def standardized_rmse_score(y, p, prep):
    y_scaled = (y - prep["target_mu"]) / prep["target_sd"]
    p_scaled = (p - prep["target_mu"]) / prep["target_sd"]
    err = p_scaled - y_scaled
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Train FrontierSAMNet only; prediction is handled by the separate predict script.")
    parser.add_argument("--smoke", action="store_true", help="Run a small smoke-test configuration.")
    args = parser.parse_args()

    config = load_config()
    if args.smoke:
        config = smoke_config(config)

    ensure_dirs()
    set_seed(config["seed"])
    device = device_from_config(config)

    with open("data/processed/feature_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    df = pd.read_csv("data/processed/sam_clean.csv")
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)

    prep = build_prep(train, manifest)
    save_json(prep_to_json(prep), "models/preprocessor.json")

    if Path("models/smiles_tokenizer.json").exists():
        tok = SmilesTokenizer.load("models/smiles_tokenizer.json")
    else:
        tok = SmilesTokenizer(max_len=config["model"]["max_smiles_length"]).fit(df["SMILES"].astype(str))
        tok.save("models/smiles_tokenizer.json")

    batch_size = config["training"]["batch_size"]
    train_ds = SAMDataset(train, manifest, prep, config, tok)
    val_ds = SAMDataset(val, manifest, prep, config, tok)
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=config["training"]["num_workers"])
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=config["training"]["num_workers"])

    cat_cards = [len(prep["cat_maps"][c]) for c in manifest["categorical_context"]]
    model = FrontierSAMNet(
        config,
        len(tok.vocab),
        len(manifest["descriptors"]),
        len(manifest["numeric_context"]),
        len(manifest["bits"]),
        cat_cards,
    ).to(device)

    if Path("models/smiles_pretrainer.pt").exists():
        ck = torch.load("models/smiles_pretrainer.pt", map_location=device)
        try:
            model.smiles.load_state_dict(ck["encoder"], strict=False)
        except Exception:
            pass

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
    )

    epochs = config["training"]["smoke_epochs"] if args.smoke else config["training"]["epochs"]
    best = float("inf")
    wait = 0
    history = []
    model_path = Path("models/frontier_sam_net.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0

        for batch in dl_train:
            batch = batch_to_device(batch, device)
            pred = model(batch)
            loss = multitask_loss(pred, batch["target"], model.log_vars)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            opt.step()

            train_loss += float(loss.detach().cpu()) * batch["target"].size(0)
            n += batch["target"].size(0)

        yv, pv = evaluate_for_validation(model, dl_val, device, prep, manifest)
        val_metrics = metrics_frame(yv, pv, manifest["targets"])
        val_rmse_raw_mean = float(val_metrics["rmse"].mean())
        val_score = standardized_rmse_score(yv, pv, prep)

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n, 1),
            "val_score_norm_rmse": val_score,
            "val_rmse_raw_mean": val_rmse_raw_mean,
            "lr": float(opt.param_groups[0]["lr"]),
        }

        for _, r in val_metrics.iterrows():
            target = r["target"]
            row[f"val_{target}_rmse"] = float(r["rmse"])
            row[f"val_{target}_mae"] = float(r["mae"])
            row[f"val_{target}_r2"] = float(r["r2"])
            row[f"val_{target}_spearman"] = float(r["spearman"])

        history.append(row)
        sched.step(val_score)
        print(row)

        if np.isfinite(val_score) and val_score < best:
            best = val_score
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "manifest": manifest,
                    "preprocessor": prep_to_json(prep),
                    "vocab": tok.vocab,
                },
                model_path,
            )
        else:
            wait += 1

        if epoch == epochs and not model_path.exists():
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "manifest": manifest,
                    "preprocessor": prep_to_json(prep),
                    "vocab": tok.vocab,
                },
                model_path,
            )

        if not args.smoke and wait >= config["training"]["patience"]:
            break

    pd.DataFrame(history).to_csv("results/tables/training_history.csv", index=False)
    print({"saved_model": str(model_path), "training_history": "results/tables/training_history.csv"})


if __name__ == "__main__":
    main()
