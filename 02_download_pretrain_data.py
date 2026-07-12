import argparse
import csv
import gzip
import io
import random
import urllib.request
from pathlib import Path

import pandas as pd

from sam_core import ensure_dirs, load_config


def extract_smiles(text):
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    names = reader.fieldnames or []
    candidates = [c for c in names if c.lower() in {"smiles", "smile", "mol", "molecule"}]
    if not candidates:
        candidates = [c for c in names if "smiles" in c.lower()]
    if not candidates:
        return rows
    col = candidates[0]
    for row in reader:
        s = str(row.get(col, "")).strip()
        if len(s) >= 3 and "." not in s:
            rows.append(s)
    return rows


def local_augmented_smiles(config, limit):
    df = pd.read_csv("data/processed/sam_clean.csv")
    base = sorted(df["SMILES"].dropna().astype(str).unique())
    pool = []
    try:
        from rdkit import Chem
        for s in base:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            pool.append(Chem.MolToSmiles(mol, canonical=True))
            for _ in range(32):
                pool.append(Chem.MolToSmiles(mol, doRandom=True))
    except Exception:
        for s in base:
            pool.extend([s, s[::-1]])
    random.Random(config["seed"]).shuffle(pool)
    return pool[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = load_config()
    ensure_dirs()
    out = Path("data/raw/pretrain_smiles.csv")
    limit = int(config["external_data"]["max_molecules"])
    smiles = []
    if config["external_data"]["enabled"] and not args.offline:
        for url in config["external_data"]["urls"]:
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    raw = r.read()
                if url.endswith(".gz"):
                    raw = gzip.decompress(raw)
                smiles.extend(extract_smiles(raw.decode("utf-8", errors="ignore")))
            except Exception as e:
                print(f"skip {url}: {e}")
            if len(smiles) >= limit:
                break
    if len(smiles) < 2000:
        smiles.extend(local_augmented_smiles(config, limit))
    seen = set()
    clean = []
    for s in smiles:
        if s not in seen:
            seen.add(s)
            clean.append(s)
        if len(clean) >= limit:
            break
    pd.DataFrame({"SMILES": clean}).to_csv(out, index=False)
    print({"pretrain_molecules": len(clean), "file": str(out)})


if __name__ == "__main__":
    main()
