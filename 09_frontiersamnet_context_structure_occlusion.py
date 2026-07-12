import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from sam_core import (
    FrontierSAMNet,
    SAMDataset,
    SmilesTokenizer,
    batch_to_device,
    device_from_config,
    ensure_dirs,
    set_seed,
)


FIXED_CONDITIONS = [
    "full",
    "active_layer_masked",
    "all_context_masked",
    "molecular_tabular_masked",
    "all_tabular_inputs_masked",
]

PERMUTATION_CONDITIONS = [
    "context_permuted_global",
    "context_permuted_within_sam",
    "molecular_permuted_within_active_layer",
    "both_context_and_molecular_permuted",
]

CONDITION_LABELS = {
    "full": "Full model",
    "active_layer_masked": "Active layer masked",
    "all_context_masked": "All context masked",
    "molecular_tabular_masked": "Descriptor + fingerprint masked",
    "all_tabular_inputs_masked": "All tabular inputs masked",
    "context_permuted_global": "Context permuted globally",
    "context_permuted_within_sam": "Context permuted within SAM",
    "molecular_permuted_within_active_layer": "Molecule permuted within active layer",
    "both_context_and_molecular_permuted": "Context + molecule permuted",
}

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
    ck_cpu = safe_torch_load(model_path, torch.device("cpu"))
    config = ck_cpu["config"]
    device = device_from_config(config)
    ck = safe_torch_load(model_path, device)
    manifest = ck["manifest"]
    prep = restore_prep(ck["preprocessor"])

    tokenizer_path = model_path.parent / "smiles_tokenizer.json"
    if tokenizer_path.exists():
        tok = SmilesTokenizer.load(tokenizer_path)
    else:
        tok = SmilesTokenizer(
            vocab=ck["vocab"],
            max_len=config["model"]["max_smiles_length"],
        )

    cat_cards = [
        len(prep["cat_maps"][c])
        for c in manifest["categorical_context"]
    ]
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


def pce_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    den = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / den) if den > 0 else np.nan
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "spearman": float(spearman) if pd.notna(spearman) else np.nan,
    }


def masked_batch(batch, condition, active_layer_index):
    if condition == "full":
        return batch

    out = dict(batch)
    if condition == "active_layer_masked":
        out["cat"] = batch["cat"].clone()
        out["cat"][:, active_layer_index] = 0
    elif condition == "all_context_masked":
        out["num"] = torch.zeros_like(batch["num"])
        out["cat"] = torch.zeros_like(batch["cat"])
    elif condition == "molecular_tabular_masked":
        out["desc"] = torch.zeros_like(batch["desc"])
        out["bits"] = torch.zeros_like(batch["bits"])
    elif condition == "all_tabular_inputs_masked":
        out["desc"] = torch.zeros_like(batch["desc"])
        out["num"] = torch.zeros_like(batch["num"])
        out["bits"] = torch.zeros_like(batch["bits"])
        out["cat"] = torch.zeros_like(batch["cat"])
    else:
        raise ValueError(f"Unknown fixed-mask condition: {condition}")
    return out


def evaluate_dataframe(
    model,
    df,
    manifest,
    prep,
    config,
    tokenizer,
    device,
    batch_size,
    fixed_condition="full",
):
    dataset = SAMDataset(
        df,
        manifest,
        prep,
        config,
        fit_tokenizer=tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if "pce" not in manifest["targets"]:
        raise KeyError("PCE target is required for this analysis.")
    if "Active_Layer" not in manifest["categorical_context"]:
        raise KeyError("Active_Layer must be a categorical context feature.")

    pce_index = manifest["targets"].index("pce")
    active_layer_index = manifest["categorical_context"].index("Active_Layer")
    y_true = []
    y_pred = []
    row_ids = []
    gate_rows = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            eval_batch = masked_batch(
                batch,
                fixed_condition,
                active_layer_index,
            )
            pred_scaled, aux = model(eval_batch, aux=True)
            pred_raw = (
                pred_scaled.detach().cpu().numpy() * prep["target_sd"]
                + prep["target_mu"]
            )
            true_raw = batch["target_raw"].detach().cpu().numpy()
            gates = aux["gates"].detach().cpu().numpy()
            rid = batch["row_id"].detach().cpu().numpy().astype(int)

            y_true.append(true_raw[:, pce_index])
            y_pred.append(pred_raw[:, pce_index])
            row_ids.append(rid)
            gate_rows.append(gates)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    row_ids = np.concatenate(row_ids)
    gates = np.concatenate(gate_rows, axis=0)
    metrics = pce_metrics(y_true, y_pred)
    metrics.update(
        {
            "graph_gate_mean": float(gates[:, 0].mean()),
            "smiles_gate_mean": float(gates[:, 1].mean()),
            "tabular_gate_mean": float(gates[:, 2].mean()),
        }
    )

    pred_frame = pd.DataFrame(
        {
            "row_id": row_ids,
            "true_pce": y_true,
            "pred_pce": y_pred,
            "graph_gate": gates[:, 0],
            "smiles_gate": gates[:, 1],
            "tabular_gate": gates[:, 2],
        }
    )
    input_meta = df[["row_id", "sam_group", "Active_Layer"]].copy()
    input_meta = input_meta.rename(columns={"Active_Layer": "input_active_layer"})
    pred_frame = pred_frame.merge(input_meta, on="row_id", how="left")
    return metrics, pred_frame


def nonidentity_permutation(n, rng):
    if n <= 1:
        return np.arange(n)
    perm = rng.permutation(n)
    if np.all(perm == np.arange(n)):
        perm = np.roll(perm, 1)
    return perm


def permute_block_global(df, columns, rng):
    out = df.copy(deep=True)
    perm = nonidentity_permutation(len(df), rng)
    out.loc[:, columns] = df.iloc[perm][columns].to_numpy()
    return out, perm


def count_changed_rows(before, after, columns):
    left = before[columns].reset_index(drop=True)
    right = after[columns].reset_index(drop=True)
    equal = left.eq(right) | (left.isna() & right.isna())
    return int((~equal.all(axis=1)).sum())


def permute_block_within_groups(df, columns, group_col, rng):
    out = df.copy(deep=True)
    moved_rows = 0
    for _, indices in df.groupby(group_col, sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        if len(idx) <= 1:
            continue
        local_perm = nonidentity_permutation(len(idx), rng)
        src = idx[local_perm]
        out.loc[idx, columns] = df.loc[src, columns].to_numpy()
        moved_rows += int(np.sum(src != idx))
    return out, moved_rows


def permute_molecule_within_active_layer(df, molecular_cols, rng):
    out = df.copy(deep=True)
    moved_rows = 0
    eligible_active_layers = 0

    for _, layer_frame in df.groupby("Active_Layer", sort=False):
        groups = list(layer_frame["sam_group"].drop_duplicates())
        if len(groups) <= 1:
            continue
        eligible_active_layers += 1
        perm = nonidentity_permutation(len(groups), rng)
        source_groups = [groups[i] for i in perm]
        representatives = {
            g: layer_frame[layer_frame["sam_group"] == g].iloc[0]
            for g in groups
        }
        for destination_group, source_group in zip(groups, source_groups):
            destination_idx = layer_frame.index[
                layer_frame["sam_group"] == destination_group
            ]
            source_values = representatives[source_group][molecular_cols].to_numpy()
            out.loc[destination_idx, molecular_cols] = source_values
            if source_group != destination_group:
                moved_rows += len(destination_idx)

    return out, moved_rows, eligible_active_layers


def summarize_metrics(metrics_frame, full_metrics):
    out = metrics_frame.copy()
    out["delta_mae_vs_full"] = out["mae"] - full_metrics["mae"]
    out["delta_mse_vs_full"] = out["mse"] - full_metrics["mse"]
    out["delta_rmse_vs_full"] = out["rmse"] - full_metrics["rmse"]
    out["delta_r2_vs_full"] = out["r2"] - full_metrics["r2"]

    rows = []
    metric_cols = [
        "mae",
        "mse",
        "rmse",
        "r2",
        "spearman",
        "delta_mae_vs_full",
        "delta_mse_vs_full",
        "delta_rmse_vs_full",
        "delta_r2_vs_full",
        "graph_gate_mean",
        "smiles_gate_mean",
        "tabular_gate_mean",
        "moved_rows",
    ]
    for condition, group in out.groupby("condition", sort=False):
        row = {
            "condition": condition,
            "label": CONDITION_LABELS[condition],
            "analysis_kind": group["analysis_kind"].iloc[0],
            "n_repeats": int(len(group)),
        }
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna().to_numpy(float)
            if len(values) == 0:
                continue
            row[f"{col}_mean"] = float(np.mean(values))
            row[f"{col}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_q025"] = float(np.quantile(values, 0.025))
            row[f"{col}_q975"] = float(np.quantile(values, 0.975))
        row["fraction_rmse_worse_than_full"] = float(
            np.mean(group["delta_rmse_vs_full"].to_numpy(float) > 0)
        )
        rows.append(row)
    return out, pd.DataFrame(rows)


def interaction_table(metrics_frame, full_mse):
    permutation = metrics_frame[
        metrics_frame["analysis_kind"] == "permutation"
    ].copy()
    pivot = permutation.pivot(index="repeat", columns="condition", values="mse")
    required = {
        "context_permuted_global",
        "molecular_permuted_within_active_layer",
        "both_context_and_molecular_permuted",
    }
    missing = required - set(pivot.columns)
    if missing:
        raise KeyError(f"Missing interaction conditions: {sorted(missing)}")

    out = pd.DataFrame(index=pivot.index)
    out["full_mse"] = full_mse
    out["context_permuted_mse"] = pivot["context_permuted_global"]
    out["molecular_permuted_mse"] = pivot[
        "molecular_permuted_within_active_layer"
    ]
    out["both_permuted_mse"] = pivot[
        "both_context_and_molecular_permuted"
    ]
    out["interaction_mse"] = (
        out["both_permuted_mse"]
        - out["context_permuted_mse"]
        - out["molecular_permuted_mse"]
        + out["full_mse"]
    )
    return out.reset_index()



def main():
    parser = argparse.ArgumentParser(
        description="Fixed-model context-structure occlusion analysis for FrontierSAMNet."
    )
    parser.add_argument("--model-path", default="models/frontier_sam_net.pt")
    parser.add_argument("--input-csv", default="data/processed/sam_clean.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--table-dir", default="results/tables")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    model_path = Path(args.model_path)
    input_path = Path(args.input_csv)
    table_dir = Path(args.table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    model, config, manifest, prep, tokenizer, device = load_trained_model(model_path)
    data = pd.read_csv(input_path)
    analysis_df = data[data["split"] == args.split].reset_index(drop=True)
    if len(analysis_df) == 0:
        raise ValueError(f"No rows found for split={args.split!r}.")

    context_cols = manifest["numeric_context"] + manifest["categorical_context"]
    molecular_cols = ["SMILES"] + manifest["descriptors"] + manifest["bits"]
    original_active_layer = analysis_df.set_index("row_id")["Active_Layer"].to_dict()

    metric_rows = []
    prediction_frames = []

    for condition in FIXED_CONDITIONS:
        metrics, predictions = evaluate_dataframe(
            model,
            analysis_df,
            manifest,
            prep,
            config,
            tokenizer,
            device,
            args.batch_size,
            fixed_condition=condition,
        )
        metrics.update(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "analysis_kind": "fixed_mask",
                "repeat": 0,
                "moved_rows": 0,
            }
        )
        metric_rows.append(metrics)
        predictions["condition"] = condition
        predictions["analysis_kind"] = "fixed_mask"
        predictions["repeat"] = 0
        predictions["original_active_layer"] = predictions["row_id"].map(original_active_layer)
        prediction_frames.append(predictions)

    full_metrics = metric_rows[0].copy()

    audit_repeat_rows = []
    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat + 1)

        context_global, context_permutation = permute_block_global(
            analysis_df,
            context_cols,
            rng,
        )
        context_within_sam, _ = permute_block_within_groups(
            analysis_df,
            context_cols,
            "sam_group",
            rng,
        )
        molecule_within_layer, _, eligible_layers = (
            permute_molecule_within_active_layer(
                analysis_df,
                molecular_cols,
                rng,
            )
        )
        both_permuted = molecule_within_layer.copy(deep=True)
        both_permuted.loc[:, context_cols] = analysis_df.iloc[
            context_permutation
        ][context_cols].to_numpy()

        moved_context_global = count_changed_rows(
            analysis_df,
            context_global,
            context_cols,
        )
        moved_context_within_sam = count_changed_rows(
            analysis_df,
            context_within_sam,
            context_cols,
        )
        moved_molecule = count_changed_rows(
            analysis_df,
            molecule_within_layer,
            molecular_cols,
        )
        moved_both = count_changed_rows(
            analysis_df,
            both_permuted,
            context_cols + molecular_cols,
        )

        conditions = [
            (
                "context_permuted_global",
                context_global,
                moved_context_global,
            ),
            (
                "context_permuted_within_sam",
                context_within_sam,
                moved_context_within_sam,
            ),
            (
                "molecular_permuted_within_active_layer",
                molecule_within_layer,
                moved_molecule,
            ),
            (
                "both_context_and_molecular_permuted",
                both_permuted,
                moved_both,
            ),
        ]

        for condition, permuted_df, moved_rows in conditions:
            metrics, predictions = evaluate_dataframe(
                model,
                permuted_df,
                manifest,
                prep,
                config,
                tokenizer,
                device,
                args.batch_size,
                fixed_condition="full",
            )
            metrics.update(
                {
                    "condition": condition,
                    "label": CONDITION_LABELS[condition],
                    "analysis_kind": "permutation",
                    "repeat": repeat + 1,
                    "moved_rows": moved_rows,
                }
            )
            metric_rows.append(metrics)
            predictions["condition"] = condition
            predictions["analysis_kind"] = "permutation"
            predictions["repeat"] = repeat + 1
            predictions["original_active_layer"] = predictions["row_id"].map(original_active_layer)
            prediction_frames.append(predictions)

        audit_repeat_rows.append(
            {
                "repeat": repeat + 1,
                "moved_context_global_rows": moved_context_global,
                "moved_context_within_sam_rows": moved_context_within_sam,
                "moved_molecular_rows": moved_molecule,
                "moved_both_rows": moved_both,
                "eligible_active_layers_for_molecular_permutation": eligible_layers,
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame, summary = summarize_metrics(metrics_frame, full_metrics)
    interaction = interaction_table(metrics_frame, full_metrics["mse"])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    repeat_audit = pd.DataFrame(audit_repeat_rows)

    metrics_path = table_dir / "frontiersamnet_context_structure_occlusion_metrics.csv"
    summary_path = table_dir / "frontiersamnet_context_structure_occlusion_summary.csv"
    prediction_path = table_dir / "frontiersamnet_context_structure_occlusion_predictions.csv"
    interaction_path = table_dir / "frontiersamnet_context_structure_interaction.csv"
    repeat_audit_path = table_dir / "frontiersamnet_context_structure_permutation_audit.csv"
    audit_path = table_dir / "frontiersamnet_context_structure_occlusion_audit.json"

    metrics_frame.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    interaction.to_csv(interaction_path, index=False)
    repeat_audit.to_csv(repeat_audit_path, index=False)

    interaction_values = interaction["interaction_mse"].to_numpy(float)
    audit = {
        "model_path": str(model_path),
        "input_csv": str(input_path),
        "split": args.split,
        "seed": args.seed,
        "repeats": args.repeats,
        "device": str(device),
        "test_rows": int(len(analysis_df)),
        "test_sam_groups": int(analysis_df["sam_group"].nunique()),
        "test_active_layers": int(analysis_df["Active_Layer"].nunique()),
        "numeric_context_features": manifest["numeric_context"],
        "categorical_context_features": manifest["categorical_context"],
        "descriptor_count": len(manifest["descriptors"]),
        "fingerprint_bit_count": len(manifest["bits"]),
        "fixed_mask_semantics": {
            "numeric_context_zero": "training-mean reference in standardized space",
            "categorical_context_zero": "reserved unknown/padding embedding index",
            "descriptor_zero": "training-mean reference in standardized space",
            "fingerprint_zero": "all bits absent",
        },
        "interaction_mse_mean": float(np.mean(interaction_values)),
        "interaction_mse_q025": float(np.quantile(interaction_values, 0.025)),
        "interaction_mse_q975": float(np.quantile(interaction_values, 0.975)),
        "outputs": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "predictions": str(prediction_path),
            "interaction": str(interaction_path),
            "permutation_audit": str(repeat_audit_path),
        },
        "interpretation_boundary": (
            "Fixed-model occlusion and permutation quantify predictive sensitivity; "
            "they do not establish causal feature contributions or retrained-model capacity."
        ),
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
