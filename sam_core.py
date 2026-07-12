import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:
    torch = None
    nn = None
    F = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:
    Chem = None
    AllChem = None


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs():
    for p in ["data/raw", "data/processed", "models", "results/tables", "results/figures"]:
        Path(p).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def device_from_config(config):
    choice = config["training"].get("device", "auto")
    if torch is None:
        return None
    if choice != "auto":
        return torch.device(choice)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clean_name(x):
    y = str(x).replace("\u00a0", " ").strip()
    y = re.sub(r"\s+", " ", y)
    return y


def safe_number(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    t = str(s).strip().replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
    if m:
        return float(m.group(0))
    return np.nan
def canonical_smiles(s):
    if pd.isna(s):
        return "missing_smiles"

    text = str(s).strip()
    if not text:
        return "missing_smiles"

    if Chem is None:
        return clean_name(text)

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return clean_name(text)

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def read_source(config):
    df = pd.read_excel(config["input_file"], sheet_name=config.get("sheet_name", 0))
    df.columns = [clean_name(c) for c in df.columns]
    rename = {}
    for k, v in config["target_columns"].items():
        rename[clean_name(v)] = k
    for c in list(df.columns):
        if c in rename:
            df = df.rename(columns={c: rename[c]})
    return df


def infer_columns(df):
    id_cols = [c for c in ["name", "SMILES", "doi"] if c in df.columns]
    targets = ["pce", "voc", "jsc", "ff"]
    bit_cols = [c for c in df.columns if re.fullmatch(r"Bit_\d+", str(c))]
    start = df.columns.get_loc(bit_cols[0]) if bit_cols else len(df.columns)
    descriptor_cols = []
    for c in df.columns:
        if c in id_cols or c in targets or c in bit_cols:
            continue
        loc = df.columns.get_loc(c)
        if loc < start:
            if pd.api.types.is_numeric_dtype(df[c]):
                descriptor_cols.append(c)
    context_cols = []
    for c in df.columns:
        if c in id_cols or c in targets or c in bit_cols or c in descriptor_cols:
            continue
        context_cols.append(c)
        force_categorical = {
            "Active_Layer",
            "Device_Type",
            "Device_Architecture",
            "Carrier_Role",
            "Substrate_Type",
            "SAM_Solvent",
        }

        force_numeric = {
            "Substrate_Work_Function",
            "SAM_Solution_Concentration(mM)",
            "Soaking",
            "Spin_Coating",
            "Soak_Time",
            "Spin_Coating_Time(s)",
            "Spin_Coating_Speed(rpm)",
            "Annealing",
            "Initial_Annealing_Temperature_for_SAM (°C)",
            "Post-washing_Annealing_Temperature(°C)",
            "Initial_Annealing_Time_for_SAM(min)",
            "Post-washing_Annealing_Time(min)",
            "UV-Ozone_Treatment_Time(min)",
        }

        numeric_context = []
        categorical_context = []

        for c in context_cols:
            if c in force_categorical:
                categorical_context.append(c)
            elif c in force_numeric:
                numeric_context.append(c)
            else:
                vals = df[c].map(safe_number)
                ratio = vals.notna().mean()
                if ratio >= 0.75:
                    numeric_context.append(c)
                else:
                    categorical_context.append(c)


    return {
        "id": id_cols,
        "targets": targets,
        "bits": bit_cols,
        "descriptors": descriptor_cols,
        "numeric_context": numeric_context,
        "categorical_context": categorical_context
    }


def build_clean_frame(config):
    df = read_source(config)
    manifest = infer_columns(df)

    if "SMILES" not in df.columns:
        raise KeyError("SMILES column is required for SAM-level splitting.")

    df["SMILES"] = df["SMILES"].fillna("").astype(str).map(clean_name)
    df["sam_group"] = df["SMILES"].map(canonical_smiles)

    if "sam_group" not in manifest["id"]:
        manifest["id"].append("sam_group")

    for c in manifest["targets"]:
        df[c] = df[c].map(safe_number)

    for c in manifest["descriptors"] + manifest["numeric_context"]:
        df[c] = df[c].map(safe_number)

    for c in manifest["bits"]:
        df[c] = df[c].fillna(0).astype(float).clip(0, 1)

    for c in manifest["categorical_context"]:
        df[c] = df[c].fillna("missing").astype(str).map(clean_name)

    df = df.copy()
    df["row_id"] = np.arange(len(df))
    df = df[df[manifest["targets"]].notna().all(axis=1)].reset_index(drop=True)

    return df, manifest


def split_by_group(df, config):
    seed = config["seed"]
    rng = np.random.default_rng(seed)

    group_col = config["split"]["group_column"]
    if group_col not in df.columns:
        raise KeyError(f"group_column '{group_col}' not found in dataframe.")

    group_values = df[group_col].fillna("missing_group").astype(str)
    groups = np.array(sorted(group_values.unique()))

    n_groups = len(groups)
    test_fraction = float(config["split"]["test_fraction"])
    val_fraction = float(config["split"]["val_fraction"])

    test_n = max(1, int(round(n_groups * test_fraction)))
    val_n = max(1, int(round(n_groups * val_fraction)))

    n_trials = int(config["split"].get("n_trials", 1000))

    desired = {
        "test": test_fraction,
        "val": val_fraction,
        "train": 1.0 - test_fraction - val_fraction,
    }

    target_cols = [c for c in ["pce", "voc", "jsc", "ff"] if c in df.columns]

    if target_cols:
        global_mu = df[target_cols].mean()
        global_sd = df[target_cols].std().replace(0, 1)
    else:
        global_mu = None
        global_sd = None

    def make_one_split():
        order = groups.copy()
        rng.shuffle(order)

        test_groups = set(order[:test_n])
        val_groups = set(order[test_n:test_n + val_n])

        split = []
        for g in group_values:
            if g in test_groups:
                split.append("test")
            elif g in val_groups:
                split.append("val")
            else:
                split.append("train")

        return np.array(split)

    def score_split(split):
        score = 0.0
        n_rows = len(split)

        
        for part in ["train", "val", "test"]:
            frac = float((split == part).sum()) / max(n_rows, 1)
            score += 10.0 * (frac - desired[part]) ** 2

        
        if target_cols:
            for part in ["train", "val", "test"]:
                sub = df.loc[split == part, target_cols]
                if len(sub) == 0:
                    return 1e9

                z = ((sub.mean() - global_mu) / global_sd).fillna(0)
                score += 0.15 * float((z ** 2).mean())

        return score

    best_split = None
    best_score = float("inf")

    for _ in range(n_trials):
        candidate = make_one_split()
        s = score_split(candidate)
        if s < best_score:
            best_score = s
            best_split = candidate

    return best_split


def standardize_fit(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where((sd > 1e-8) & np.isfinite(sd), sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_apply(x, mu, sd):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    y = (x - mu) / sd
    y = np.where(np.isfinite(y), y, 0.0)
    y = np.clip(y, -12.0, 12.0)
    return y.astype(np.float32)


def make_category_maps(df, cols):
    maps = {}
    for c in cols:
        vals = sorted(df[c].fillna("missing").astype(str).unique())
        maps[c] = {v: i + 1 for i, v in enumerate(vals)}
    return maps


def encode_categories(df, cols, maps):
    if not cols:
        return np.zeros((len(df), 0), dtype=np.int64)
    arr = np.zeros((len(df), len(cols)), dtype=np.int64)
    for j, c in enumerate(cols):
        m = maps[c]
        arr[:, j] = df[c].fillna("missing").astype(str).map(lambda z: m.get(z, 0)).to_numpy()
    return arr


class SmilesTokenizer:
    def __init__(self, vocab=None, max_len=220):
        self.max_len = max_len
        if vocab is None:
            vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2, "<mask>": 3}
        self.vocab = vocab

    def fit(self, smiles):
        chars = sorted(set("".join([str(s) for s in smiles])))
        for ch in chars:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        return self

    def encode(self, s):
        ids = [self.vocab["<cls>"]]
        ids += [self.vocab.get(ch, self.vocab["<unk>"]) for ch in str(s)[:self.max_len - 1]]
        mask = [1] * len(ids)
        while len(ids) < self.max_len:
            ids.append(0)
            mask.append(0)
        return np.array(ids, dtype=np.int64), np.array(mask, dtype=np.int64)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "max_len": self.max_len}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["vocab"], d["max_len"])


def atom_id(atom):
    z = atom.GetAtomicNum()
    allowed = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 82]
    if z in allowed:
        return allowed.index(z) + 1
    return min(z, 99)


def graph_from_smiles(smiles, max_atoms):
    atoms = np.zeros(max_atoms, dtype=np.int64)
    degree = np.zeros(max_atoms, dtype=np.int64)
    aromatic = np.zeros(max_atoms, dtype=np.int64)
    charge = np.zeros(max_atoms, dtype=np.float32)
    adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    if Chem is None:
        n = min(len(str(smiles)), max_atoms)
        for i, ch in enumerate(str(smiles)[:n]):
            atoms[i] = min(ord(ch), 99)
            if i + 1 < n:
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
        return atoms, degree, aromatic, charge, adj
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return atoms, degree, aromatic, charge, adj
    n = min(mol.GetNumAtoms(), max_atoms)
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_atoms:
            break
        atoms[i] = atom_id(atom)
        degree[i] = min(atom.GetTotalDegree(), 6)
        aromatic[i] = int(atom.GetIsAromatic())
        charge[i] = float(atom.GetFormalCharge())
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            bt = float(bond.GetBondTypeAsDouble())
            adj[i, j] = bt
            adj[j, i] = bt
    return atoms, degree, aromatic, charge, adj


if torch is not None:
    class SAMDataset(torch.utils.data.Dataset):
        def __init__(self, df, manifest, prep, config, fit_tokenizer=None):
            self.df = df.reset_index(drop=True)
            self.manifest = manifest
            self.config = config
            self.max_atoms = config["model"]["max_atoms"]
            self.targets = self.df[manifest["targets"]].to_numpy(np.float32)
            self.target_scaled = standardize_apply(self.targets, prep["target_mu"], prep["target_sd"])
            desc = self.df[manifest["descriptors"]].to_numpy(np.float64) if manifest["descriptors"] else np.zeros((len(df), 0), dtype=np.float32)
            num = self.df[manifest["numeric_context"]].to_numpy(np.float64) if manifest["numeric_context"] else np.zeros((len(df), 0), dtype=np.float32)
            bits = self.df[manifest["bits"]].to_numpy(np.float32) if manifest["bits"] else np.zeros((len(df), 0), dtype=np.float32)
            self.desc = standardize_apply(desc, prep["desc_mu"], prep["desc_sd"]) if desc.shape[1] else desc
            self.num = standardize_apply(num, prep["num_mu"], prep["num_sd"]) if num.shape[1] else num
            self.bits = bits.astype(np.float32)
            self.cat = encode_categories(self.df, manifest["categorical_context"], prep["cat_maps"])
            self.tokenizer = fit_tokenizer

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            ids, mask = self.tokenizer.encode(row["SMILES"])
            atoms, degree, aromatic, charge, adj = graph_from_smiles(row["SMILES"], self.max_atoms)
            atom_mask = (atoms > 0).astype(np.float32)
            return {
                "smiles_ids": torch.tensor(ids),
                "smiles_mask": torch.tensor(mask, dtype=torch.bool),
                "atoms": torch.tensor(atoms),
                "degree": torch.tensor(degree),
                "aromatic": torch.tensor(aromatic),
                "charge": torch.tensor(charge),
                "adj": torch.tensor(adj),
                "atom_mask": torch.tensor(atom_mask, dtype=torch.bool),
                "desc": torch.tensor(self.desc[idx]),
                "num": torch.tensor(self.num[idx]),
                "bits": torch.tensor(self.bits[idx]),
                "cat": torch.tensor(self.cat[idx]),
                "target": torch.tensor(self.target_scaled[idx]),
                "target_raw": torch.tensor(self.targets[idx]),
                "row_id": torch.tensor(int(row["row_id"]))
            }


    class DropPath(nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p

        def forward(self, x):
            if self.p == 0 or not self.training:
                return x
            keep = 1 - self.p
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = x.new_empty(shape).bernoulli_(keep)
            return x * mask / keep


    class MLP(nn.Module):
        def __init__(self, sizes, dropout=0.0):
            super().__init__()
            layers = []
            for i in range(len(sizes) - 1):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                if i < len(sizes) - 2:
                    layers.append(nn.GELU())
                    layers.append(nn.LayerNorm(sizes[i + 1]))
                    layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


    class GINBlock(nn.Module):
        def __init__(self, dim, dropout, drop_path):
            super().__init__()
            self.mlp = MLP([dim, dim * 2, dim], dropout)
            self.norm = nn.LayerNorm(dim)
            self.dp = DropPath(drop_path)

        def forward(self, x, adj, mask):
            deg = adj.sum(-1, keepdim=True).clamp(min=1)
            msg = torch.bmm(adj, x) / deg
            y = self.mlp(x + msg)
            y = self.dp(y)
            return self.norm(x + y) * mask.unsqueeze(-1)


    class GraphEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            dim = config["model"]["hidden_dim"]
            layers = config["model"]["graph_layers"]
            heads = config["model"]["heads"]
            dropout = config["model"]["dropout"]
            sd = config["model"]["stochastic_depth"]
            self.atom = nn.Embedding(128, dim, padding_idx=0)
            self.degree = nn.Embedding(8, dim)
            self.aromatic = nn.Embedding(2, dim)
            self.charge = nn.Linear(1, dim)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.gin = nn.ModuleList([GINBlock(dim, dropout, sd * i / max(1, layers - 1)) for i in range(layers)])
            enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(enc, num_layers=max(2, layers // 2))
            self.norm = nn.LayerNorm(dim)

        def forward(self, atoms, degree, aromatic, charge, adj, mask):
            x = self.atom(atoms) + self.degree(degree.clamp(0, 7)) + self.aromatic(aromatic.clamp(0, 1)) + self.charge(charge.unsqueeze(-1))
            x = x * mask.unsqueeze(-1)
            for block in self.gin:
                x = block(x, adj, mask)
            cls = self.cls.expand(x.size(0), -1, -1)
            z = torch.cat([cls, x], 1)
            pad = torch.cat([torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device), mask], 1)
            z = self.tr(z, src_key_padding_mask=~pad)
            return self.norm(z[:, 0]), z


    class SmilesEncoder(nn.Module):
        def __init__(self, vocab_size, config):
            super().__init__()
            dim = config["model"]["hidden_dim"]
            layers = config["model"]["smiles_layers"]
            heads = config["model"]["heads"]
            dropout = config["model"]["dropout"]
            max_len = config["model"]["max_smiles_length"]
            self.tok = nn.Embedding(vocab_size, dim, padding_idx=0)
            self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
            enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(enc, num_layers=layers)
            self.norm = nn.LayerNorm(dim)

        def forward(self, ids, mask):
            x = self.tok(ids) + self.pos[:, :ids.size(1)]
            x = self.tr(x, src_key_padding_mask=~mask)
            return self.norm(x[:, 0]), x


    class NumericTokenizer(nn.Module):
        def __init__(self, n, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n, dim) * 0.02)
            self.bias = nn.Parameter(torch.zeros(n, dim))

        def forward(self, x):
            if x.shape[1] == 0:
                return x.new_zeros((x.size(0), 0, self.weight.size(1)))
            return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


    class TabularEncoder(nn.Module):
        def __init__(self, desc_dim, num_dim, bit_dim, cat_cards, config):
            super().__init__()
            dim = config["model"]["hidden_dim"]
            heads = config["model"]["heads"]
            dropout = config["model"]["dropout"]
            layers = config["model"]["tabular_layers"]
            self.desc_tok = NumericTokenizer(desc_dim, dim)
            self.num_tok = NumericTokenizer(num_dim, dim)
            self.bit_proj = MLP([bit_dim, dim * 2, dim], dropout) if bit_dim else None
            self.cat_emb = nn.ModuleList([nn.Embedding(card + 1, dim, padding_idx=0) for card in cat_cards])
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(enc, num_layers=layers)
            self.norm = nn.LayerNorm(dim)

        def forward(self, desc, num, bits, cat):
            tokens = [self.cls.expand(desc.size(0), -1, -1)]
            if desc.shape[1]:
                tokens.append(self.desc_tok(desc))
            if num.shape[1]:
                tokens.append(self.num_tok(num))
            if self.bit_proj is not None:
                tokens.append(self.bit_proj(bits).unsqueeze(1))
            if len(self.cat_emb):
                cats = [emb(cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_emb)]
                tokens.append(torch.cat(cats, 1))
            x = torch.cat(tokens, 1)
            x = self.tr(x)
            return self.norm(x[:, 0]), x


    class FusionEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            dim = config["model"]["hidden_dim"]
            heads = config["model"]["heads"]
            dropout = config["model"]["dropout"]
            layers = config["model"]["fusion_layers"]
            enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(enc, num_layers=layers)
            self.gate = MLP([dim * 3, dim, 3], dropout)
            self.norm = nn.LayerNorm(dim)

        def forward(self, g, s, t):
            base = torch.stack([g, s, t], 1)
            z = self.tr(base)
            gates = torch.softmax(self.gate(torch.cat([g, s, t], -1)), -1)
            fused = (z * gates.unsqueeze(-1)).sum(1)
            return self.norm(fused), gates, z


    class ExpertHead(nn.Module):
        def __init__(self, config, out_dim):
            super().__init__()
            dim = config["model"]["hidden_dim"]
            k = config["model"]["experts"]
            dropout = config["model"]["dropout"]
            self.experts = nn.ModuleList([MLP([dim, dim, dim // 2, out_dim], dropout) for _ in range(k)])
            self.router = MLP([dim, dim // 2, k], dropout)

        def forward(self, x):
            w = torch.softmax(self.router(x), -1)
            ys = torch.stack([e(x) for e in self.experts], -1)
            return (ys * w.unsqueeze(1)).sum(-1), w


    class FrontierSAMNet(nn.Module):
        def __init__(self, config, vocab_size, desc_dim, num_dim, bit_dim, cat_cards):
            super().__init__()
            self.graph = GraphEncoder(config)
            self.smiles = SmilesEncoder(vocab_size, config)
            self.tab = TabularEncoder(desc_dim, num_dim, bit_dim, cat_cards, config)
            self.fusion = FusionEncoder(config)
            self.head = ExpertHead(config, 4)
            self.log_vars = nn.Parameter(torch.zeros(4))

        def forward(self, batch, aux=False):
            g, gt = self.graph(batch["atoms"], batch["degree"], batch["aromatic"], batch["charge"], batch["adj"], batch["atom_mask"])
            s, st = self.smiles(batch["smiles_ids"], batch["smiles_mask"])
            t, tt = self.tab(batch["desc"], batch["num"], batch["bits"], batch["cat"])
            fused, gates, modal = self.fusion(g, s, t)
            pred, route = self.head(fused)
            if aux:
                return pred, {"gates": gates, "route": route, "modal": modal}
            return pred


    class SmilesPretrainer(nn.Module):
        def __init__(self, vocab_size, config):
            super().__init__()
            self.encoder = SmilesEncoder(vocab_size, config)
            dim = config["model"]["hidden_dim"]
            self.mlm = nn.Linear(dim, vocab_size)
            self.proj = MLP([dim, dim, dim], config["model"]["dropout"])

        def forward(self, ids, mask):
            cls, toks = self.encoder(ids, mask)
            return self.mlm(toks), F.normalize(self.proj(cls), dim=-1)


def batch_to_device(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def multitask_loss(pred, target, log_vars):
    mse = (pred - target) ** 2
    loss = 0.0
    for i in range(pred.shape[1]):
        loss = loss + torch.exp(-log_vars[i]) * mse[:, i].mean() + log_vars[i]
    return loss


def metrics_frame(y, p, targets):
    rows = []
    for i, t in enumerate(targets):
        yt = y[:, i]
        pt = p[:, i]
        err = pt - yt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        den = float(np.sum((yt - yt.mean()) ** 2))
        r2 = float(1 - np.sum(err ** 2) / den) if den > 0 else np.nan
        sp = pd.Series(yt).corr(pd.Series(pt), method="spearman")
        rows.append({"target": t, "mae": mae, "rmse": rmse, "r2": r2, "spearman": float(sp) if pd.notna(sp) else np.nan})
    return pd.DataFrame(rows)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
