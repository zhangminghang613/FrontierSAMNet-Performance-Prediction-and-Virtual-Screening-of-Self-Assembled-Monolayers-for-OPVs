import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from sam_core import (
    ExpertHead,
    FrontierSAMNet,
    GraphEncoder,
    MLP,
    SAMDataset,
    SmilesEncoder,
    SmilesTokenizer,
    TabularEncoder,
    batch_to_device,
    device_from_config,
    metrics_frame,
    multitask_loss,
    set_seed,
)


ALL_VARIANTS = (
    "tabular_only",
    "graph_smiles_only",
    "without_adaptive_gate",
)


VARIANT_SPECS = {
    "tabular_only": {
        "modalities": ("tabular",),
        "adaptive_gate": False,
    },
    "graph_smiles_only": {
        "modalities": ("graph", "smiles"),
        "use_fingerprint": False,
        "adaptive_gate": True,
    },
    "without_adaptive_gate": {
        "modalities": ("graph", "smiles", "tabular"),
        "adaptive_gate": False,
    },
}

DEFAULT_MODEL_DIR = Path("models/module_ablation")
DEFAULT_TABLE_DIR = Path("results/tables")


def output_layout(model_dir, table_dir, smoke):
    model_dir = Path(model_dir)
    table_dir = Path(table_dir)
    if smoke:
        model_dir = model_dir / "smoke"
    prefix = "module_ablation_smoke" if smoke else "module_ablation"
    return model_dir, table_dir, prefix


def table_path(table_dir, prefix, name):
    return table_dir / f"{prefix}_{name}.csv"


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def restore_prep(data):
    return {
        "desc_mu": np.asarray(data["desc_mu"], dtype=np.float32),
        "desc_sd": np.asarray(data["desc_sd"], dtype=np.float32),
        "num_mu": np.asarray(data["num_mu"], dtype=np.float32),
        "num_sd": np.asarray(data["num_sd"], dtype=np.float32),
        "target_mu": np.asarray(data["target_mu"], dtype=np.float32),
        "target_sd": np.asarray(data["target_sd"], dtype=np.float32),
        "cat_maps": data["cat_maps"],
    }


def smoke_config(config):
    out = copy.deepcopy(config)
    out["model"].update(
        {
            "hidden_dim": 96,
            "graph_layers": 2,
            "smiles_layers": 2,
            "tabular_layers": 1,
            "fusion_layers": 1,
            "heads": 4,
            "experts": 2,
            "max_atoms": 96,
            "max_smiles_length": 128,
        }
    )
    out["training"]["batch_size"] = 32
    return out


class AblationFusion(nn.Module):

    def __init__(self, config, modalities, adaptive_gate):
        super().__init__()
        self.modalities = tuple(modalities)
        self.adaptive_gate = bool(adaptive_gate)
        dim = config["model"]["hidden_dim"]
        heads = config["model"]["heads"]
        dropout = config["model"]["dropout"]
        layers = config["model"]["fusion_layers"]

        encoder_layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.gate = (
            MLP([dim * len(self.modalities), dim, len(self.modalities)], dropout)
            if self.adaptive_gate and len(self.modalities) > 1
            else None
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, representations):
        base = torch.stack([representations[name] for name in self.modalities], dim=1)
        encoded = self.transformer(base)
        if self.gate is None:
            gates = encoded.new_full(
                (encoded.size(0), len(self.modalities)),
                1.0 / len(self.modalities),
            )
        else:
            raw = torch.cat([representations[name] for name in self.modalities], dim=-1)
            gates = torch.softmax(self.gate(raw), dim=-1)
        fused = (encoded * gates.unsqueeze(-1)).sum(dim=1)
        return self.norm(fused), gates


class AblatedFrontierSAMNet(nn.Module):
    def __init__(
        self,
        config,
        vocab_size,
        desc_dim,
        num_dim,
        bit_dim,
        cat_cards,
        spec,
    ):
        super().__init__()
        self.spec = dict(spec)
        self.modalities = tuple(spec["modalities"])
        self.fingerprint_enabled = (
            bool(spec.get("use_fingerprint", True))
            and "tabular" in self.modalities
        )
        self.target_indices = (0, 1, 2, 3)

        self.graph = GraphEncoder(config) if "graph" in self.modalities else None
        self.smiles = SmilesEncoder(vocab_size, config) if "smiles" in self.modalities else None
        self.tabular = (
            TabularEncoder(
                desc_dim,
                num_dim,
                bit_dim if self.fingerprint_enabled else 0,
                cat_cards,
                config,
            )
            if "tabular" in self.modalities
            else None
        )
        self.fusion = AblationFusion(
            config,
            self.modalities,
            adaptive_gate=spec["adaptive_gate"],
        )
        self.head = ExpertHead(config, 4)
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(self, batch, aux=False):
        representations = {}
        if self.graph is not None:
            representations["graph"], _ = self.graph(
                batch["atoms"],
                batch["degree"],
                batch["aromatic"],
                batch["charge"],
                batch["adj"],
                batch["atom_mask"],
            )
        if self.smiles is not None:
            representations["smiles"], _ = self.smiles(
                batch["smiles_ids"], batch["smiles_mask"]
            )
        if self.tabular is not None:
            tabular_bits = (
                batch["bits"]
                if self.fingerprint_enabled
                else batch["bits"][:, :0]
            )
            representations["tabular"], _ = self.tabular(
                batch["desc"], batch["num"], tabular_bits, batch["cat"]
            )

        fused, gates = self.fusion(representations)
        prediction, route = self.head(fused)
        if aux:
            return prediction, {"gates": gates, "route": route}
        return prediction


def parse_variants(text):
    requested = list(ALL_VARIANTS) if text.strip().lower() == "all" else [
        item.strip() for item in text.split(",") if item.strip()
    ]
    unknown = sorted(set(requested) - set(ALL_VARIANTS))
    if unknown:
        raise ValueError(
            f"Unknown variants: {unknown}. Supported variants: {list(ALL_VARIANTS)}"
        )
    return list(dict.fromkeys(requested))


def make_loaders(df, manifest, prep, config, tokenizer):
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 0)
    frames = {
        split: df[df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    datasets = {
        split: SAMDataset(frame, manifest, prep, config, tokenizer)
        for split, frame in frames.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "train_eval": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return frames, loaders


def target_names_and_indices(model, manifest):
    indices = tuple(getattr(model, "target_indices", range(len(manifest["targets"]))))
    names = [manifest["targets"][i] for i in indices]
    return names, indices


def enable_mc_dropout(model):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def evaluate_loader(model, loader, device, prep, manifest, mc_dropout_passes=1):
    mc_dropout_passes = int(mc_dropout_passes)
    if mc_dropout_passes < 1:
        raise ValueError("mc_dropout_passes must be at least 1.")

    if mc_dropout_passes > 1:
        enable_mc_dropout(model)
    else:
        model.eval()
    target_names, target_indices = target_names_and_indices(model, manifest)
    index_array = np.asarray(target_indices, dtype=int)
    predictions = []
    uncertainties = []
    truths = []
    row_ids = []

    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch_to_device(batch, device)
                draws = torch.stack(
                    [model(batch) for _ in range(mc_dropout_passes)], dim=0
                )
                pred_scaled = draws.mean(dim=0).detach().cpu().numpy()
                unc_scaled = draws.std(dim=0, unbiased=False).detach().cpu().numpy()
                raw_pred = (
                    pred_scaled * prep["target_sd"][index_array]
                    + prep["target_mu"][index_array]
                )
                raw_unc = unc_scaled * prep["target_sd"][index_array]
                raw_true = (
                    batch["target_raw"][:, list(target_indices)].detach().cpu().numpy()
                )
                predictions.append(raw_pred)
                uncertainties.append(raw_unc)
                truths.append(raw_true)
                row_ids.extend(
                    batch["row_id"].detach().cpu().numpy().astype(int).tolist()
                )
    finally:
        model.eval()

    y_true = np.vstack(truths)
    y_pred = np.vstack(predictions)
    y_unc = np.vstack(uncertainties)
    pred_frame = pd.DataFrame({"row_id": row_ids})
    for index, target in enumerate(target_names):
        pred_frame[f"pred_{target}"] = y_pred[:, index]
        pred_frame[f"raw_unc_{target}"] = y_unc[:, index]
        pred_frame[target] = y_true[:, index]
    return y_true, y_pred, pred_frame, target_names


def validation_score(model, loader, device, prep, manifest):
    y_true, y_pred, _, _ = evaluate_loader(model, loader, device, prep, manifest)
    _, indices = target_names_and_indices(model, manifest)
    idx = np.asarray(indices, dtype=int)
    y_scaled = (y_true - prep["target_mu"][idx]) / prep["target_sd"][idx]
    p_scaled = (y_pred - prep["target_mu"][idx]) / prep["target_sd"][idx]
    return float(np.sqrt(np.mean((p_scaled - y_scaled) ** 2)))


def compute_loss(model, prediction, target):
    selected = target[:, list(model.target_indices)]
    return multitask_loss(prediction, selected, model.log_vars)


def maybe_load_smiles_pretraining(model, pretrain_path, device):
    if model.smiles is None or not pretrain_path.exists():
        return False
    checkpoint = safe_torch_load(pretrain_path, device)
    try:
        model.smiles.load_state_dict(checkpoint["encoder"], strict=False)
        return True
    except (KeyError, RuntimeError, ValueError):
        return False


def save_checkpoint(path, model, config, manifest, prep_json, vocab, variant, spec):
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "manifest": manifest,
            "preprocessor": prep_json,
            "vocab": vocab,
            "variant": variant,
            "variant_spec": spec,
            "target_indices": list(model.target_indices),
        },
        path,
    )


def normalized_variant_spec(spec):
    if not spec:
        return None
    return {
        "modalities": tuple(spec.get("modalities", ())),
        "use_fingerprint": bool(spec.get("use_fingerprint", True)),
        "adaptive_gate": bool(spec.get("adaptive_gate", True)),
    }


def train_variant(
    variant,
    spec,
    config,
    manifest,
    prep,
    prep_json,
    tokenizer,
    loaders,
    device,
    checkpoint_path,
    history_path,
    pretrain_path,
    overwrite,
    smoke,
):
    cat_cards = [len(prep["cat_maps"][c]) for c in manifest["categorical_context"]]
    model = AblatedFrontierSAMNet(
        config,
        len(tokenizer.vocab),
        len(manifest["descriptors"]),
        len(manifest["numeric_context"]),
        len(manifest["bits"]),
        cat_cards,
        spec,
    ).to(device)

    if checkpoint_path.exists() and not overwrite:
        checkpoint = safe_torch_load(checkpoint_path, device)
        saved_spec = normalized_variant_spec(checkpoint.get("variant_spec"))
        current_spec = normalized_variant_spec(spec)
        if saved_spec == current_spec:
            model.load_state_dict(checkpoint["model"])
            return model, False
        print(
            {
                "variant": variant,
                "checkpoint_spec_mismatch": True,
                "saved_spec": saved_spec,
                "current_spec": current_spec,
                "action": "retrain_and_replace_stale_ablation_checkpoint",
            }
        )

    pretrained_smiles_loaded = maybe_load_smiles_pretraining(model, pretrain_path, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
    )
    epochs = config["training"]["smoke_epochs"] if smoke else config["training"]["epochs"]
    patience = config["training"]["patience"]
    best = float("inf")
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_rows = 0
        for batch in loaders["train"]:
            batch = batch_to_device(batch, device)
            prediction = model(batch)
            loss = compute_loss(model, prediction, batch["target"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), config["training"]["gradient_clip"]
            )
            optimizer.step()
            batch_n = batch["target"].size(0)
            running_loss += float(loss.detach().cpu()) * batch_n
            n_rows += batch_n

        score = validation_score(model, loaders["val"], device, prep, manifest)
        scheduler.step(score)
        row = {
            "variant": variant,
            "epoch": epoch,
            "train_loss": running_loss / max(n_rows, 1),
            "val_score_norm_rmse": score,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "smiles_pretraining_loaded": pretrained_smiles_loaded,
        }
        history.append(row)
        print(row)

        if np.isfinite(score) and score < best:
            best = score
            wait = 0
            save_checkpoint(
                checkpoint_path,
                model,
                config,
                manifest,
                prep_json,
                tokenizer.vocab,
                variant,
                spec,
            )
        else:
            wait += 1

        if not smoke and wait >= patience:
            break

    pd.DataFrame(history).to_csv(history_path, index=False)
    if not checkpoint_path.exists():
        save_checkpoint(
            checkpoint_path,
            model,
            config,
            manifest,
            prep_json,
            tokenizer.vocab,
            variant,
            spec,
        )
    best_checkpoint = safe_torch_load(checkpoint_path, device)
    model.load_state_dict(best_checkpoint["model"])
    return model, True


def evaluate_all_splits(
    model,
    frames,
    loaders,
    device,
    prep,
    manifest,
    variant,
    mc_dropout_passes=1,
):
    metric_frames = []
    prediction_frames = []
    for split, loader_key in (("train", "train_eval"), ("val", "val"), ("test", "test")):
        y_true, y_pred, pred_frame, target_names = evaluate_loader(
            model,
            loaders[loader_key],
            device,
            prep,
            manifest,
            mc_dropout_passes=mc_dropout_passes,
        )
        metrics = metrics_frame(y_true, y_pred, target_names)
        metrics["split"] = split
        metrics["variant"] = variant
        metrics["evaluation_mode"] = (
            "mc_dropout_mean" if mc_dropout_passes > 1 else "deterministic"
        )
        metrics["mc_dropout_passes"] = int(mc_dropout_passes)
        metrics["fingerprint_enabled"] = bool(
            getattr(model, "fingerprint_enabled", True)
        )
        metric_frames.append(metrics)

        metadata_cols = [
            column
            for column in ("row_id", "sam_group", "name", "SMILES", "doi")
            if column in frames[split].columns
        ]
        pred_frame = pred_frame.merge(
            frames[split][metadata_cols], on="row_id", how="left"
        )
        pred_frame["split"] = split
        pred_frame["variant"] = variant
        pred_frame["evaluation_mode"] = (
            "mc_dropout_mean" if mc_dropout_passes > 1 else "deterministic"
        )
        pred_frame["mc_dropout_passes"] = int(mc_dropout_passes)
        pred_frame["fingerprint_enabled"] = bool(
            getattr(model, "fingerprint_enabled", True)
        )
        prediction_frames.append(pred_frame)
    return pd.concat(metric_frames, ignore_index=True), pd.concat(
        prediction_frames, ignore_index=True
    )


def count_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))


def r2(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denominator = float(np.sum((y - y.mean()) ** 2))
    if denominator <= 0:
        return np.nan
    return float(1.0 - np.sum((y - pred) ** 2) / denominator)


def paired_group_bootstrap(full_predictions, ablated_predictions, repeats, seed):
    full = full_predictions[
        (full_predictions["split"] == "test")
        & full_predictions["pred_pce"].notna()
    ][["row_id", "sam_group", "pce", "pred_pce"]].rename(
        columns={"pred_pce": "pred_full"}
    )
    ablated = ablated_predictions[
        (ablated_predictions["split"] == "test")
        & ablated_predictions["pred_pce"].notna()
    ][["row_id", "pred_pce"]].rename(columns={"pred_pce": "pred_ablated"})
    paired = full.merge(ablated, on="row_id", how="inner")
    groups = paired["sam_group"].dropna().astype(str).unique()
    if len(groups) < 2 or repeats <= 0:
        return {}

    by_group = {
        group: paired[paired["sam_group"].astype(str) == group]
        for group in groups
    }
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat([by_group[group] for group in sampled_groups], ignore_index=True)
        y = sampled["pce"].to_numpy(float)
        full_pred = sampled["pred_full"].to_numpy(float)
        ablated_pred = sampled["pred_ablated"].to_numpy(float)
        draws.append(
            {
                "delta_rmse_vs_full": rmse(y, ablated_pred) - rmse(y, full_pred),
                "delta_mae_vs_full": mae(y, ablated_pred) - mae(y, full_pred),
                "delta_r2_vs_full": r2(y, ablated_pred) - r2(y, full_pred),
            }
        )

    draw_frame = pd.DataFrame(draws)
    summary = {}
    for column in draw_frame.columns:
        values = draw_frame[column].dropna().to_numpy(float)
        summary[f"{column}_bootstrap_mean"] = float(np.mean(values))
        summary[f"{column}_ci_low"] = float(np.quantile(values, 0.025))
        summary[f"{column}_ci_high"] = float(np.quantile(values, 0.975))
    return summary


def pce_test_summary(metrics, parameter_counts, predictions, bootstrap_repeats, seed):
    selected = metrics[(metrics["split"] == "test") & (metrics["target"] == "pce")].copy()
    selected["parameters"] = selected["variant"].map(parameter_counts)
    full_row = selected[selected["variant"] == "full"].iloc[0]
    selected["delta_rmse_vs_full"] = selected["rmse"] - float(full_row["rmse"])
    selected["delta_mae_vs_full"] = selected["mae"] - float(full_row["mae"])
    selected["delta_r2_vs_full"] = selected["r2"] - float(full_row["r2"])

    full_predictions = predictions[predictions["variant"] == "full"]
    bootstrap_rows = []
    for index, variant in enumerate(selected["variant"]):
        row = {"variant": variant}
        if variant != "full":
            row.update(
                paired_group_bootstrap(
                    full_predictions,
                    predictions[predictions["variant"] == variant],
                    repeats=bootstrap_repeats,
                    seed=seed + index,
                )
            )
        bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)
    return selected.merge(bootstrap, on="variant", how="left")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Retrain FrontierSAMNet module ablations while reusing the existing "
            "full-model checkpoint as the untouched reference."
        )
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variants or 'all'.",
    )
    parser.add_argument(
        "--full-model-path",
        default="models/frontier_sam_net.pt",
        help="Existing full FrontierSAMNet checkpoint; it is never overwritten.",
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/sam_clean.csv",
        help="Processed data containing the fixed train/val/test split.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory for trained ablation checkpoints.",
    )
    parser.add_argument(
        "--table-dir",
        default=str(DEFAULT_TABLE_DIR),
        help="Directory for all generated CSV tables.",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=2000,
        help="SAM-group paired bootstrap repeats for PCE test deltas.",
    )
    parser.add_argument(
        "--mc-dropout-passes",
        type=int,
        default=None,
        help=(
            "Stochastic forward passes used for final train/val/test metrics. "
            "Default uses training.mc_dropout_passes from the full checkpoint; "
            "use 1 for deterministic evaluation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain and replace checkpoints inside the ablation model directory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use the small smoke configuration, a smoke checkpoint subdirectory, "
            "and smoke-prefixed CSV filenames."
        ),
    )
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    full_model_path = Path(args.full_model_path)
    if not full_model_path.exists():
        raise FileNotFoundError(
            f"Full checkpoint not found: {full_model_path}. The full model will not be retrained."
        )

    full_checkpoint_cpu = safe_torch_load(full_model_path, torch.device("cpu"))
    full_config = full_checkpoint_cpu["config"]
    config = smoke_config(full_config) if args.smoke else copy.deepcopy(full_config)
    if args.mc_dropout_passes is not None:
        mc_dropout_passes = int(args.mc_dropout_passes)
    elif args.smoke:
        mc_dropout_passes = 4
    else:
        mc_dropout_passes = int(config["training"].get("mc_dropout_passes", 1))
    if mc_dropout_passes < 1:
        raise ValueError("--mc-dropout-passes must be at least 1.")
    evaluation_seed = int(config["seed"])
    manifest = full_checkpoint_cpu["manifest"]
    prep_json = full_checkpoint_cpu["preprocessor"]
    prep = restore_prep(prep_json)
    tokenizer = SmilesTokenizer(
        vocab=full_checkpoint_cpu["vocab"],
        max_len=config["model"]["max_smiles_length"],
    )
    device = device_from_config(config)
    set_seed(config["seed"])

    model_dir, table_dir, output_prefix = output_layout(
        args.model_dir,
        args.table_dir,
        args.smoke,
    )
    for directory in (model_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input_csv)
    expected_splits = {"train", "val", "test"}
    if "split" not in data.columns or set(data["split"].dropna().unique()) != expected_splits:
        raise ValueError("Input CSV must contain the fixed train/val/test split labels.")
    frames, loaders = make_loaders(data, manifest, prep, config, tokenizer)

    all_metrics = []
    all_predictions = []
    parameter_counts = {}

    if args.smoke:
        print(
            {
                "full_reference": "skipped in smoke mode because smoke dimensions differ",
                "full_checkpoint_untouched": str(full_model_path),
            }
        )
    else:
        cat_cards = [len(prep["cat_maps"][c]) for c in manifest["categorical_context"]]
        full_model = FrontierSAMNet(
            full_config,
            len(tokenizer.vocab),
            len(manifest["descriptors"]),
            len(manifest["numeric_context"]),
            len(manifest["bits"]),
            cat_cards,
        ).to(device)
        full_checkpoint = safe_torch_load(full_model_path, device)
        full_model.load_state_dict(full_checkpoint["model"])
        set_seed(evaluation_seed)
        full_metrics, full_predictions = evaluate_all_splits(
            full_model,
            frames,
            loaders,
            device,
            prep,
            manifest,
            "full",
            mc_dropout_passes=mc_dropout_passes,
        )
        full_metrics.to_csv(
            table_path(table_dir, output_prefix, "full_metrics"),
            index=False,
        )
        full_predictions.to_csv(
            table_path(table_dir, output_prefix, "full_predictions"),
            index=False,
        )
        all_metrics.append(full_metrics)
        all_predictions.append(full_predictions)
        parameter_counts["full"] = count_parameters(full_model)
        del full_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pretrain_path = Path("models/smiles_pretrainer.pt")
    run_rows = []
    for variant in variants:
        set_seed(config["seed"])
        checkpoint_path = model_dir / f"{variant}.pt"
        history_path = table_path(
            table_dir,
            output_prefix,
            f"{variant}_training_history",
        )
        model, trained_now = train_variant(
            variant,
            VARIANT_SPECS[variant],
            config,
            manifest,
            prep,
            prep_json,
            tokenizer,
            loaders,
            device,
            checkpoint_path,
            history_path,
            pretrain_path,
            args.overwrite,
            args.smoke,
        )
        set_seed(evaluation_seed)
        metrics, predictions = evaluate_all_splits(
            model,
            frames,
            loaders,
            device,
            prep,
            manifest,
            variant,
            mc_dropout_passes=mc_dropout_passes,
        )
        metrics.to_csv(
            table_path(table_dir, output_prefix, f"{variant}_metrics"),
            index=False,
        )
        predictions.to_csv(
            table_path(table_dir, output_prefix, f"{variant}_predictions"),
            index=False,
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        parameter_counts[variant] = count_parameters(model)
        run_rows.append(
            {
                "variant": variant,
                "trained_now": trained_now,
                "checkpoint": str(checkpoint_path),
                "parameters": parameter_counts[variant],
                "evaluation_mode": (
                    "mc_dropout_mean" if mc_dropout_passes > 1 else "deterministic"
                ),
                "mc_dropout_passes": mc_dropout_passes,
                "evaluation_seed": evaluation_seed,
                **VARIANT_SPECS[variant],
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    metrics_path = table_path(table_dir, output_prefix, "metrics")
    predictions_path = table_path(table_dir, output_prefix, "predictions")
    manifest_path = table_path(table_dir, output_prefix, "run_manifest")
    combined_metrics.to_csv(metrics_path, index=False)
    combined_predictions.to_csv(predictions_path, index=False)
    pd.DataFrame(run_rows).to_csv(manifest_path, index=False)

    summary_path = None
    if not args.smoke:
        summary = pce_test_summary(
            combined_metrics,
            parameter_counts,
            combined_predictions,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=config["seed"],
        )
        summary_path = table_path(table_dir, output_prefix, "pce_test_summary")
        summary.to_csv(summary_path, index=False)

    print(
        {
            "full_checkpoint_retrained": False,
            "full_checkpoint": str(full_model_path),
            "evaluation_mode": (
                "mc_dropout_mean" if mc_dropout_passes > 1 else "deterministic"
            ),
            "mc_dropout_passes": mc_dropout_passes,
            "evaluation_seed": evaluation_seed,
            "variants": variants,
            "model_dir": str(model_dir),
            "table_dir": str(table_dir),
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "run_manifest": str(manifest_path),
            "pce_test_summary": None if summary_path is None else str(summary_path),
        }
    )


if __name__ == "__main__":
    main()
