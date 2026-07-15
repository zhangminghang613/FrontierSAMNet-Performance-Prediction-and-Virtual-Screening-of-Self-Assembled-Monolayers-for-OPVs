from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/processed/sam_clean.csv")

OUT_DIR = Path("results")
TAB_DIR = OUT_DIR / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def canonical_smiles(smiles):
    mol = mol_from_smiles(smiles)

    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def clean_text_label(x, max_len=38):
    x = str(x)
    if len(x) <= max_len:
        return x
    return x[: max_len - 3] + "..."



SMARTS_LIBRARY = {
    
    "phosphonic acid / phosphonate": [
        "P(=O)(O)O",
        "P(=O)([O-])O",
        "P(=O)(O)[O-]",
    ],
    "carboxylic acid / carboxylate": [
        "C(=O)[O;H1]",
        "C(=O)[O-]",
        "C(=O)O",
    ],

    
    "carbazole core": [
        "n1c2ccccc2c2ccccc21",
        "[nX3]1c2ccccc2c2ccccc21",
    ],
    "benzene ring": [
        "c1ccccc1",
    ],
    "thiophene ring": [
        "c1ccsc1",
        "c1cccs1",
    ],

    
    "C2 alkyl spacer": [
        "CCP(=O)(O)O",
        "CCP(=O)([O-])O",
    ],
    "C3 alkyl spacer": [
        "CCCP(=O)(O)O",
    ],
    "C4 alkyl spacer": [
        "CCCCP(=O)(O)O",
    ],
    "C6 alkyl chain": [
        "CCCCCC",
    ],

    
    "carbonyl group": [
        "C=O",
    ],
    "fluorinated carbon": [
        "[CX4](F)",
        "cF",
    ],
    "chlorinated aryl": [
        "cCl",
    ],
    "brominated aryl": [
        "cBr",
    ],
}


def compile_smarts_library():
    compiled = {}

    for name, smarts_list in SMARTS_LIBRARY.items():
        patterns = []

        for smarts in smarts_list:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                patterns.append(patt)

        compiled[name] = patterns

    return compiled


SMARTS_PATTERNS = compile_smarts_library()


def has_smarts(mol, motif_name):
    patterns = SMARTS_PATTERNS.get(motif_name, [])

    for patt in patterns:
        try:
            if mol.HasSubstructMatch(patt):
                return True
        except Exception:
            continue

    return False


def detect_smarts_features(mol):
    out = {}

    for motif in SMARTS_LIBRARY.keys():
        out[motif] = int(has_smarts(mol, motif))

    return out

def calc_rdkit_design_descriptors(mol):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]

    return {
        "Ring count": rdMolDescriptors.CalcNumRings(mol),
        "Aromatic rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "P atoms": atoms.count("P"),
        "N atoms": atoms.count("N"),
        "O atoms": atoms.count("O"),
        "F atoms": atoms.count("F"),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def clean_brics_fragment(fragment):
    s = str(fragment)

    
    s = re.sub(r"\[\d+\*\]", "[*]", s)

    mol = mol_from_smiles(s)

    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_brics_fragments(mol):
    try:
        frags = BRICS.BRICSDecompose(
            mol,
            keepNonLeafNodes=False,
            returnMols=False,
        )
    except Exception:
        return set()

    out = set()

    for f in frags:
        cf = clean_brics_fragment(f)
        if cf is not None:
            out.add(cf)

    return out


def build_unique_sam_table(df):
    required_cols = ["SMILES", "pce"]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    work = df.copy()

    if "name" not in work.columns:
        work["name"] = "SAM"

    work["canonical_smiles"] = work["SMILES"].apply(canonical_smiles)

    invalid = work["canonical_smiles"].isna().sum()

    work = work.dropna(subset=["canonical_smiles", "pce"]).copy()

    unique = (
        work.groupby("canonical_smiles", as_index=False)
        .agg(
            name=("name", "first"),
            original_smiles=("SMILES", "first"),
            pce_mean=("pce", "mean"),
            pce_median=("pce", "median"),
            pce_max=("pce", "max"),
            record_count=("pce", "size"),
        )
    )

    mols = []

    for smi in unique["canonical_smiles"]:
        mols.append(mol_from_smiles(smi))

    unique["mol"] = mols

    return unique, invalid


def add_fragment_and_descriptor_features(unique):
    smarts_rows = []
    desc_rows = []
    brics_rows = []

    for row in unique.itertuples(index=False):
        mol = row.mol

        smarts = detect_smarts_features(mol)
        desc = calc_rdkit_design_descriptors(mol)
        brics = get_brics_fragments(mol)

        smarts_rows.append(smarts)
        desc_rows.append(desc)
        brics_rows.append(brics)

    smarts_df = pd.DataFrame(smarts_rows)
    desc_df = pd.DataFrame(desc_rows)

    out = pd.concat(
        [
            unique.drop(columns=["mol"]).reset_index(drop=True),
            smarts_df.reset_index(drop=True),
            desc_df.reset_index(drop=True),
        ],
        axis=1,
    )

    out["brics_fragments"] = ["|".join(sorted(x)) for x in brics_rows]

    return out, brics_rows


def assign_high_low_groups(unique_features):
    q75 = unique_features["pce_mean"].quantile(0.75)
    q25 = unique_features["pce_mean"].quantile(0.25)

    out = unique_features.copy()
    out["pce_group"] = "middle"
    out.loc[out["pce_mean"] >= q75, "pce_group"] = "high"
    out.loc[out["pce_mean"] <= q25, "pce_group"] = "low"

    return out, q75, q25

def compute_brics_enrichment(unique_features, brics_rows):

    high_mask = unique_features["pce_group"].eq("high").values
    low_mask = unique_features["pce_group"].eq("low").values

    high_total = int(high_mask.sum())
    low_total = int(low_mask.sum())

    all_frags = sorted(set().union(*brics_rows))

    rows = []

    for frag in all_frags:
        present = np.array([frag in s for s in brics_rows])

        high_count = int((present & high_mask).sum())
        low_count = int((present & low_mask).sum())

        total_count = high_count + low_count

        
        if total_count < 2:
            continue

        high_frequency = high_count / max(high_total, 1)
        low_frequency = low_count / max(low_total, 1)

        
        high_rate_pc = (high_count + 0.5) / (high_total + 1.0)
        low_rate_pc = (low_count + 0.5) / (low_total + 1.0)

        log2_enrichment = float(np.log2(high_rate_pc / low_rate_pc))

        rows.append(
            {
                "brics_fragment": frag,
                "high_count": high_count,
                "low_count": low_count,
                "high_total": high_total,
                "low_total": low_total,
                "high_frequency": high_frequency,
                "low_frequency": low_frequency,
                "frequency_difference": high_frequency - low_frequency,
                "log2_enrichment": log2_enrichment,
                "total_count_high_low": total_count,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        out.to_csv(
            TAB_DIR / "sam_brics_fragment_enrichment_unique_smiles.csv",
            index=False,
        )
        return out

    out = out.sort_values(
        ["log2_enrichment", "high_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    out.to_csv(
        TAB_DIR / "sam_brics_fragment_enrichment_unique_smiles.csv",
        index=False,
    )

    return out


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    unique, _ = build_unique_sam_table(df)
    unique_features, brics_rows = add_fragment_and_descriptor_features(unique)
    unique_features, _, _ = assign_high_low_groups(unique_features)

    unique_features.to_csv(
        TAB_DIR / "sam_unique_smiles_fragment_features.csv",
        index=False,
    )

    compute_brics_enrichment(unique_features, brics_rows)


if __name__ == "__main__":
    main()
