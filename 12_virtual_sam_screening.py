from pathlib import Path
import json
import warnings
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from sam_core import (
    FrontierSAMNet,
    SAMDataset,
    SmilesTokenizer,
    batch_to_device,
    device_from_config,
    ensure_dirs,
    load_config,
)
warnings.filterwarnings("ignore")

RANDOM_SEED = 42


def set_global_seed(seed=RANDOM_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass

TOP_N_FIGURE = 12
AD_TANIMOTO_THRESHOLD = 0.35
EXTERNAL_LIBRARY_PATH = Path("data/virtual_sam_library.csv")

OUT_DIR = Path("results")
TAB_DIR = OUT_DIR / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)
UNCERTAINTY_CALIBRATION_PATH = TAB_DIR / "uncertainty_calibration.csv"


def get_mc_dropout_passes(config):
    try:
        passes = int(config["training"]["mc_dropout_passes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "config.yaml must define training.mc_dropout_passes as a positive integer."
        ) from exc

    if passes < 1:
        raise ValueError("training.mc_dropout_passes must be at least 1.")
    return passes


def load_uncertainty_scales(targets, path=UNCERTAINTY_CALIBRATION_PATH):
    scales = {t: 1.0 for t in targets}
    if not path.exists():
        return scales, pd.DataFrame()

    try:
        cal = pd.read_csv(path)
    except Exception:
        return scales, pd.DataFrame()

    required = {"target", "uncertainty_scale"}
    if not required.issubset(cal.columns):
        return scales, cal

    for row in cal.itertuples(index=False):
        target = str(getattr(row, "target"))
        if target not in scales:
            continue
        try:
            scale = float(getattr(row, "uncertainty_scale"))
        except Exception:
            scale = 1.0
        if np.isfinite(scale) and scale > 0:
            scales[target] = scale

    return scales, cal


def apply_uncertainty_calibration(pred, targets, scales):
    out = pred.copy()
    for target in targets:
        unc_col = f"unc_{target}"
        raw_col = f"raw_unc_{target}"
        if unc_col not in out.columns:
            continue
        out[raw_col] = out[unc_col].astype(float)
        out[unc_col] = out[raw_col] * float(scales.get(target, 1.0))
    return out



def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonical_smiles_rdkit(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def parse_bit_index(bit_name):
    s = str(bit_name)
    chunks = "".join(ch if ch.isdigit() else " " for ch in s).split()
    if not chunks:
        return None
    return int(chunks[-1])


def infer_morgan_fp_size(bit_cols, default=1024):
    idxs = [parse_bit_index(c) for c in bit_cols]
    idxs = [i for i in idxs if i is not None]
    if not idxs:
        return default
    return max(default, max(idxs) + 1)


_MORGAN_GENERATOR_CACHE = {}


def get_morgan_generator(n_bits):
    n_bits = int(n_bits)
    if n_bits not in _MORGAN_GENERATOR_CACHE:
        _MORGAN_GENERATOR_CACHE[n_bits] = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=n_bits,
        )
    return _MORGAN_GENERATOR_CACHE[n_bits]


def morgan_bit_array(smiles, n_bits=1024):
    mol = mol_from_smiles(smiles)
    arr = np.zeros(n_bits, dtype=np.float32)
    if mol is None:
        return arr
    fp = get_morgan_generator(n_bits).GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def morgan_fp(smiles, n_bits=1024):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return get_morgan_generator(n_bits).GetFingerprint(mol)


def morgan_bit_dict_for_manifest(smiles, bit_cols):
    fp_size = infer_morgan_fp_size(bit_cols, default=1024)
    arr = morgan_bit_array(smiles, n_bits=fp_size)
    parsed = [parse_bit_index(c) for c in bit_cols]
    no_indices = all(i is None for i in parsed)

    out = {}
    for j, c in enumerate(bit_cols):
        idx = j if no_indices else parse_bit_index(c)
        if idx is None or idx < 0 or idx >= len(arr):
            out[c] = 0.0
        else:
            out[c] = float(arr[idx])
    return out


def build_training_fps(train_smiles, n_bits=1024):
    fps = []
    for smi in train_smiles:
        fp = morgan_fp(smi, n_bits=n_bits)
        if fp is not None:
            fps.append((smi, fp))
    return fps


def tanimoto_to_training(candidate_smiles, train_fps, n_bits=1024):
    fp = morgan_fp(candidate_smiles, n_bits=n_bits)
    if fp is None or not train_fps:
        return 0.0, None

    best_sim = -1.0
    best_smiles = None
    for smi, tfp in train_fps:
        sim = DataStructs.TanimotoSimilarity(fp, tfp)
        if sim > best_sim:
            best_sim = sim
            best_smiles = smi

    if best_sim < 0:
        return 0.0, None
    return float(best_sim), best_smiles


def _add_candidate(candidates, core, substituent, anchor, spacer, smiles, design_family):
    candidates.append(
        {
            "core": core,
            "substituent": substituent,
            "anchor": anchor,
            "spacer": spacer,
            "smiles": smiles,
            "design_family": design_family,
        }
    )


def build_builtin_virtual_sam_library():
    candidates = []

    spacer_prefix = {"C2": "CC", "C3": "CCC", "C4": "CCCC"}
    spacer_tail = {
        "C2": "CCP(=O)(O)O",
        "C3": "CCCP(=O)(O)O",
        "C4": "CCCCP(=O)(O)O",
    }

    for spacer_name, sp in spacer_prefix.items():
        templates = [
            ("carbazole", "H", f"O=P(O)(O){sp}n1c2ccccc2c2ccccc21", "PACz-like"),
            ("carbazole", "di-F", f"O=P(O)(O){sp}n1c2ccc(F)cc2c2cc(F)ccc21", "halogenated PACz-like"),
            ("carbazole", "di-Cl", f"O=P(O)(O){sp}n1c2ccc(Cl)cc2c2cc(Cl)ccc21", "halogenated PACz-like"),
            ("carbazole", "di-Br", f"O=P(O)(O){sp}n1c2ccc(Br)cc2c2cc(Br)ccc21", "halogenated PACz-like"),
            ("diphenyl-carbazole", "phenyl", f"O=P(O)(O){sp}n1c2ccc(-c3ccccc3)cc2c2cc(-c3ccccc3)ccc21", "aryl-extended carbazole"),
            ("dithiophene-carbazole", "thiophene", f"O=P(O)(O){sp}n1c2cc(-c3ccsc3)ccc2c2ccc(-c3ccsc3)cc21", "thiophene-extended carbazole"),
            ("thiophene-carbazole", "bromothiophene", f"O=P(O)(O){sp}n1c2ccccc2c2ccc(-c3csc(Br)c3)cc21", "thiophene-extended carbazole"),
            ("dinaphthyl-carbazole", "naphthyl", f"O=P(O)(O){sp}n1c2cc(-c3cccc4ccccc34)ccc2c2ccc(-c3cccc4ccccc34)cc21", "fused-aryl-extended carbazole"),
        ]
        for core, substituent, smiles, family in templates:
            _add_candidate(candidates, core, substituent, "phosphonic acid", spacer_name, smiles, family)

    for spacer_name, tail in spacer_tail.items():
        templates = [
            ("dimethoxy-carbazole", "di-OMe", f"COc1ccc2c(c1)c1ccc(OC)cc1n2{tail}", "electron-rich carbazole"),
            ("dimethyl-carbazole", "di-Me", f"Cc1ccc2c(c1)c1ccc(C)cc1n2{tail}", "alkylated carbazole"),
            ("benzocarbazole", "anisyl", f"COc1cc(OC)cc(-c2ccc3c(c2)c2c4ccccc4ccc2n3{tail})c1", "fused-aryl carbazole"),
        ]
        for core, substituent, smiles, family in templates:
            _add_candidate(candidates, core, substituent, "phosphonic acid", spacer_name, smiles, family)

    negative_controls = [
        ("alkyl acid", "C6-chain", "carboxylic acid", "C6", "CCCCCCC(=O)O", "negative-control acid"),
        ("fluoroalkyl acid", "perfluoroalkyl", "carboxylic acid", "long-chain", "O=C(O)CCCCCCCC(F)(F)F", "negative-control acid"),
    ]
    for core, substituent, anchor, spacer, smiles, family in negative_controls:
        _add_candidate(candidates, core, substituent, anchor, spacer, smiles, family)

    return pd.DataFrame(candidates)


def load_external_virtual_library(path):
    raw = pd.read_csv(path)
    if "smiles" not in raw.columns and "SMILES" in raw.columns:
        raw = raw.rename(columns={"SMILES": "smiles"})
    if "smiles" not in raw.columns:
        raise ValueError(f"{path} must contain a 'smiles' or 'SMILES' column.")

    defaults = {
        "core": "external_core",
        "spacer": "external_spacer",
        "anchor": "external_anchor",
        "substituent": "external_substituent",
        "design_family": "external_library",
    }
    for c, default in defaults.items():
        if c not in raw.columns:
            raw[c] = default

    if "candidate_name" not in raw.columns:
        raw["candidate_name"] = [f"external_candidate_{i + 1:04d}" for i in range(len(raw))]

    return raw[["candidate_name", "core", "spacer", "anchor", "substituent", "design_family", "smiles"]].copy()


def finalize_virtual_library(raw_candidates):
    out = raw_candidates.copy()
    for c in ["core", "spacer", "anchor", "substituent", "design_family"]:
        if c not in out.columns:
            out[c] = "unspecified"

    if "candidate_name" not in out.columns:
        out["candidate_name"] = (
            out["core"].astype(str) + "_" + out["spacer"].astype(str) + "_" + out["substituent"].astype(str)
        )

    out["canonical_smiles"] = out["smiles"].apply(canonical_smiles_rdkit)
    out["valid_smiles"] = out["canonical_smiles"].notna()
    out = out[out["valid_smiles"]].copy()
    out = out.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    out["candidate_id"] = [f"VSAM-{i + 1:04d}" for i in range(len(out))]
    return out


def build_virtual_sam_library():
    if EXTERNAL_LIBRARY_PATH.exists():
        raw = load_external_virtual_library(EXTERNAL_LIBRARY_PATH)
    else:
        raw = build_builtin_virtual_sam_library()
    return finalize_virtual_library(raw)


def calc_rdkit_descriptors_for_manifest(smiles, descriptor_cols, fallback_values):
    mol = mol_from_smiles(smiles)
    values = {}

    if mol is None:
        for c in descriptor_cols:
            values[c] = fallback_values.get(c, 0.0)
        return values

    try:
        desc_dict = Descriptors.CalcMolDescriptors(mol)
    except Exception:
        desc_dict = {}

    for c in descriptor_cols:
        v = desc_dict.get(c, fallback_values.get(c, 0.0))
        try:
            v = float(v)
        except Exception:
            v = fallback_values.get(c, 0.0)
        if not np.isfinite(v):
            v = fallback_values.get(c, 0.0)
        values[c] = v
    return values


def choose_representative_context(df, manifest):
    if "split" in df.columns:
        source = df[df["split"] == "train"].copy()
    else:
        source = df.copy()
    if len(source) == 0:
        source = df.copy()

    q75 = source["pce"].quantile(0.75)
    high = source[source["pce"] >= q75].copy()
    if len(high) == 0:
        high = source.copy()

    context = {}
    for c in manifest["numeric_context"]:
        if c not in high.columns:
            context[c] = 0.0
            continue
        val = pd.to_numeric(high[c], errors="coerce").median()
        if not np.isfinite(val):
            val = pd.to_numeric(source[c], errors="coerce").median()
        if not np.isfinite(val):
            val = 0.0
        context[c] = float(val)

    for c in manifest["categorical_context"]:
        if c not in high.columns:
            context[c] = "missing"
            continue
        s = high[c].fillna("missing").astype(str)
        context[c] = s.mode().iloc[0] if len(s.mode()) else "missing"

    context["_reference_context_source"] = "train_upper_quartile_pce"
    context["_reference_context_pce_q75"] = float(q75)
    return context


def build_virtual_feature_frame(candidates, df_train_all, manifest):
    descriptor_fallback = {}
    for c in manifest["descriptors"]:
        if c not in df_train_all.columns:
            descriptor_fallback[c] = 0.0
            continue
        vals = pd.to_numeric(df_train_all[c], errors="coerce")
        med = vals.median()
        descriptor_fallback[c] = float(med) if np.isfinite(med) else 0.0

    context = choose_representative_context(df_train_all, manifest)
    rows = []

    for i, row in candidates.iterrows():
        smi = row["canonical_smiles"]
        item = {
            "row_id": int(1_000_000 + i),
            "name": row["candidate_name"],
            "SMILES": smi,
            "doi": "virtual_design",
            "sam_group": smi,
            "pce": 0.0,
            "voc": 0.0,
            "jsc": 0.0,
            "ff": 0.0,
        }

        item.update(calc_rdkit_descriptors_for_manifest(smi, manifest["descriptors"], descriptor_fallback))
        item.update(morgan_bit_dict_for_manifest(smi, manifest["bits"]))

        for c in manifest["numeric_context"]:
            item[c] = context.get(c, 0.0)
        for c in manifest["categorical_context"]:
            item[c] = context.get(c, "missing")

        item["candidate_id"] = row["candidate_id"]
        item["core"] = row["core"]
        item["spacer"] = row["spacer"]
        item["anchor"] = row["anchor"]
        item["substituent"] = row["substituent"]
        item["design_family"] = row.get("design_family", "unspecified")
        rows.append(item)

    return pd.DataFrame(rows), context






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


def load_frontier_model(config=None):
    if config is None:
        config = load_config()
    device = device_from_config(config)
    model_path = Path("models/frontier_sam_net.pt")
    if not model_path.exists():
        raise FileNotFoundError("Cannot find models/frontier_sam_net.pt. Run 04_train_multimodal_model.py first.")

    try:
        ck = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(model_path, map_location=device)

    model_config = ck["config"]
    manifest = ck["manifest"]
    prep = restore_prep(ck["preprocessor"])

    if Path("models/smiles_tokenizer.json").exists():
        tok = SmilesTokenizer.load("models/smiles_tokenizer.json")
    else:
        tok = SmilesTokenizer(vocab=ck["vocab"], max_len=model_config["model"]["max_smiles_length"])

    cat_cards = [len(prep["cat_maps"][c]) for c in manifest["categorical_context"]]

    model = FrontierSAMNet(
        model_config,
        len(tok.vocab),
        len(manifest["descriptors"]),
        len(manifest["numeric_context"]),
        len(manifest["bits"]),
        cat_cards,
    ).to(device)

    model.load_state_dict(ck["model"])
    model.eval()
    return model, model_config, manifest, prep, tok, device


def predict_virtual_candidates(virt_df, model, config, manifest, prep, tok, device, mc):
    
    
    set_global_seed(RANDOM_SEED)

    ds = SAMDataset(virt_df, manifest, prep, config, tok)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    if mc > 1:
        model.train()
    else:
        model.eval()

    rows = []
    gate_rows = []

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            draws = []
            aux_last = None
            for _ in range(mc):
                pred_scaled, aux = model(batch, aux=True)
                draws.append(pred_scaled.detach().cpu().numpy())
                aux_last = aux

            arr = np.stack(draws, axis=0)
            mean_scaled = arr.mean(axis=0)
            std_scaled = arr.std(axis=0)
            pred_raw = mean_scaled * prep["target_sd"] + prep["target_mu"]
            unc_raw = std_scaled * prep["target_sd"]
            row_ids = batch["row_id"].detach().cpu().numpy()
            gates = aux_last["gates"].detach().cpu().numpy()

            for i, rid in enumerate(row_ids):
                item = {"row_id": int(rid)}
                for j, t in enumerate(manifest["targets"]):
                    item[f"pred_{t}"] = float(pred_raw[i, j])
                    item[f"unc_{t}"] = float(unc_raw[i, j])
                rows.append(item)
                gate_rows.append({
                    "row_id": int(rid),
                    "graph_gate": float(gates[i, 0]),
                    "smiles_gate": float(gates[i, 1]),
                    "tabular_gate": float(gates[i, 2]),
                })

    model.eval()
    return pd.DataFrame(rows), pd.DataFrame(gate_rows)






def mark_existing_molecules(result, df):
    existing = set()
    for smi in df["SMILES"].dropna().astype(str):
        cs = canonical_smiles_rdkit(smi)
        if cs is not None:
            existing.add(cs)
    result["is_existing_dataset_sam"] = result["SMILES"].isin(existing)
    return result


def add_applicability_domain(result, train_df, bit_cols=None):
    if "split" in train_df.columns:
        train = train_df[train_df["split"] == "train"].copy()
    else:
        train = train_df.copy()

    train_smiles = sorted(train["SMILES"].dropna().astype(str).unique())
    fp_size = infer_morgan_fp_size(bit_cols, default=1024) if bit_cols is not None else 1024
    train_fps = build_training_fps(train_smiles, n_bits=fp_size)

    sims = []
    nn_smiles = []
    for smi in result["SMILES"]:
        sim, near = tanimoto_to_training(smi, train_fps, n_bits=fp_size)
        sims.append(sim)
        nn_smiles.append(near)

    result["nearest_train_tanimoto"] = sims
    result["nearest_train_smiles"] = nn_smiles
    result["in_applicability_domain"] = result["nearest_train_tanimoto"] >= AD_TANIMOTO_THRESHOLD
    return result


def add_design_rule_score(result):
    out = result.copy()

    anchor_s = out["anchor"].astype(str)
    spacer_s = out["spacer"].astype(str)
    core_s = out["core"].astype(str)
    subst_s = out["substituent"].astype(str)

    if "design_family" in out.columns:
        family_s = out["design_family"].astype(str)
    else:
        family_s = pd.Series([""] * len(out), index=out.index)

    combined_s = (core_s + " " + subst_s + " " + family_s).str.lower()

    out["rule_phosphonic_anchor"] = anchor_s.eq("phosphonic acid").astype(int)
    out["rule_short_spacer"] = spacer_s.isin(["C2", "C3", "C4"]).astype(int)
    out["rule_carbazole_like_core"] = core_s.str.contains("carbazole", case=False, regex=False).astype(int)

    out["rule_aryl_substituent"] = combined_s.str.contains(
        r"phenyl|thiophene|naphthyl|anisyl|aryl|heteroaryl|\bph\b|\bth\b",
        case=False,
        regex=True,
    ).astype(int)

    out["rule_no_carboxylic_acid"] = (
        ~anchor_s.str.contains("carboxylic", case=False, regex=False)
    ).astype(int)

    
    
    
    positive_rule_cols = [
        "rule_phosphonic_anchor",
        "rule_short_spacer",
        "rule_carbazole_like_core",
        "rule_aryl_substituent",
    ]

    out["design_rule_match_4"] = out[positive_rule_cols].sum(axis=1)
    out["design_rule_match_4_norm"] = out["design_rule_match_4"] / len(positive_rule_cols)

    
    
    
    rule_cols = positive_rule_cols + [
        "rule_no_carboxylic_acid",
    ]
    out["design_rule_score"] = out[rule_cols].sum(axis=1)
    out["design_rule_score_norm"] = out["design_rule_score"] / len(rule_cols)
    return out


def select_top_new_indomain_candidates(result, top_n=12, allow_fallback=True):
    plot_df = (
        result[(~result["is_existing_dataset_sam"]) & (result["in_applicability_domain"])]
        .sort_values(["pce_lcb_2sigma", "nearest_train_tanimoto"], ascending=[False, False])
        .head(top_n)
        .copy()
    )

    if allow_fallback and len(plot_df) < 5:
        plot_df = (
            result[~result["is_existing_dataset_sam"]]
            .sort_values(["pce_lcb_2sigma", "nearest_train_tanimoto"], ascending=[False, False])
            .head(top_n)
            .copy()
        )

    if allow_fallback and len(plot_df) < 5:
        plot_df = (
            result
            .sort_values(["pce_lcb_2sigma", "nearest_train_tanimoto"], ascending=[False, False])
            .head(top_n)
            .copy()
        )

    return plot_df




def main():
    set_global_seed(RANDOM_SEED)
    ensure_dirs()
    runtime_config = load_config()
    mc_dropout_passes = get_mc_dropout_passes(runtime_config)
    df_path = Path("data/processed/sam_clean.csv")
    manifest_path = Path("data/processed/feature_manifest.json")

    if not df_path.exists():
        raise FileNotFoundError("Cannot find data/processed/sam_clean.csv")
    if not manifest_path.exists():
        raise FileNotFoundError("Cannot find data/processed/feature_manifest.json")

    df = pd.read_csv(df_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_file = json.load(f)

    candidates = build_virtual_sam_library()
    candidates.to_csv(TAB_DIR / "virtual_sam_library_raw.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(TAB_DIR / "virtual_sam_library_deduplicated_valid.csv", index=False, encoding="utf-8-sig")

    model, model_config, manifest, prep, tok, device = load_frontier_model(runtime_config)

    virt_features, fixed_context = build_virtual_feature_frame(candidates, df, manifest)
    virt_features.to_csv(TAB_DIR / "virtual_sam_model_input_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([fixed_context]).to_csv(TAB_DIR / "virtual_screening_high_performance_reference_context.csv", index=False, encoding="utf-8-sig")

    pred, gates = predict_virtual_candidates(
        virt_features,
        model,
        model_config,
        manifest,
        prep,
        tok,
        device,
        mc=mc_dropout_passes,
    )
    uncertainty_scales, uncertainty_calibration = load_uncertainty_scales(manifest["targets"])
    pred = apply_uncertainty_calibration(pred, manifest["targets"], uncertainty_scales)
    pred.to_csv(TAB_DIR / "virtual_sam_raw_predictions.csv", index=False, encoding="utf-8-sig")
    if not uncertainty_calibration.empty:
        uncertainty_calibration.to_csv(TAB_DIR / "virtual_sam_uncertainty_calibration_used.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(TAB_DIR / "virtual_sam_modality_gates.csv", index=False, encoding="utf-8-sig")

    meta_cols = ["row_id", "candidate_id", "name", "SMILES", "core", "spacer", "anchor", "substituent", "design_family"]
    result = virt_features[meta_cols].merge(pred, on="row_id", how="left")
    result = result.merge(gates, on="row_id", how="left")
    result = mark_existing_molecules(result, df)
    result = add_applicability_domain(result, df, bit_cols=manifest["bits"])

    result["pce_lcb_2sigma"] = result["pred_pce"] - 2.0 * result["unc_pce"]
    result["pce_ucb_2sigma"] = result["pred_pce"] + 2.0 * result["unc_pce"]
    result = add_design_rule_score(result)

    result = result.sort_values(
        ["pce_lcb_2sigma", "design_rule_score_norm", "nearest_train_tanimoto"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)

    result.to_csv(TAB_DIR / "virtual_sam_prediction_ranking.csv", index=False, encoding="utf-8-sig")
    new_only = result[~result["is_existing_dataset_sam"]].copy()
    new_only.to_csv(TAB_DIR / "virtual_sam_prediction_ranking_new_only.csv", index=False, encoding="utf-8-sig")
    new_indomain = new_only[new_only["in_applicability_domain"]].copy()
    new_indomain.to_csv(TAB_DIR / "virtual_sam_prediction_ranking_new_indomain_only.csv", index=False, encoding="utf-8-sig")

    selected = select_top_new_indomain_candidates(result, top_n=TOP_N_FIGURE)
    selected.to_csv(TAB_DIR / "virtual_sam_top_selected_for_figures.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
