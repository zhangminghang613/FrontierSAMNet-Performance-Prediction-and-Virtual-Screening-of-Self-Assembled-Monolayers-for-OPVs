import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sam_core import SmilesPretrainer, SmilesTokenizer, batch_to_device, ensure_dirs, load_config, set_seed


def smoke_config(config):
    d = json_copy(config)
    d["model"]["hidden_dim"] = 96
    d["model"]["smiles_layers"] = 2
    d["model"]["heads"] = 4
    d["model"]["max_smiles_length"] = 128
    d["training"]["pretrain_batch_size"] = 64
    return d


def json_copy(x):
    import json
    return json.loads(json.dumps(x))


class PretrainSet(Dataset):
    def __init__(self, smiles, tokenizer):
        self.smiles = list(smiles)
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.vocab["<mask>"]

    def __len__(self):
        return len(self.smiles)

    def view(self, s):
        ids, mask = self.tokenizer.encode(s)
        x = torch.tensor(ids)
        m = torch.tensor(mask, dtype=torch.bool)
        labels = x.clone()
        choose = (torch.rand_like(x.float()) < 0.15) & m & (x != self.tokenizer.vocab["<cls>"])
        x[choose] = self.mask_id
        labels[~choose] = -100
        return x, m, labels

    def __getitem__(self, idx):
        a = self.view(self.smiles[idx])
        b = self.view(self.smiles[idx])
        return {"ids1": a[0], "mask1": a[1], "labels1": a[2], "ids2": b[0], "mask2": b[1], "labels2": b[2]}


def contrastive(a, b, temp=0.08):
    z = torch.cat([a, b], 0)
    sim = z @ z.t() / temp
    n = a.size(0)
    labels = torch.arange(n, device=a.device)
    loss1 = F.cross_entropy(sim[:n, n:], labels)
    loss2 = F.cross_entropy(sim[n:, :n], labels)
    return 0.5 * (loss1 + loss2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.smoke:
        config = smoke_config(config)
    ensure_dirs()
    set_seed(config["seed"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    df = pd.read_csv("data/raw/pretrain_smiles.csv")
    smiles = df["SMILES"].dropna().astype(str).drop_duplicates().tolist()
    if args.smoke:
        smiles = smiles[:512]
    tok = SmilesTokenizer(max_len=config["model"]["max_smiles_length"]).fit(smiles)
    tok.save("models/smiles_tokenizer.json")
    ds = PretrainSet(smiles, tok)
    dl = DataLoader(ds, batch_size=config["training"]["pretrain_batch_size"], shuffle=True, num_workers=config["training"]["num_workers"])
    model = SmilesPretrainer(len(tok.vocab), config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["training"]["pretrain_learning_rate"], weight_decay=config["training"]["weight_decay"])
    epochs = config["training"]["smoke_epochs"] if args.smoke else config["training"]["pretrain_epochs"]
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in dl:
            batch = batch_to_device(batch, device)
            logits1, z1 = model(batch["ids1"], batch["mask1"])
            logits2, z2 = model(batch["ids2"], batch["mask2"])
            mlm = 0.5 * (F.cross_entropy(logits1.reshape(-1, logits1.size(-1)), batch["labels1"].reshape(-1), ignore_index=-100) + F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), batch["labels2"].reshape(-1), ignore_index=-100))
            loss = mlm + 0.2 * contrastive(z1, z2)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            opt.step()
            total += float(loss.detach().cpu()) * batch["ids1"].size(0)
            count += batch["ids1"].size(0)
        rows.append({"epoch": epoch, "pretrain_loss": total / max(count, 1)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv("results/tables/pretrain_history.csv", index=False)
    torch.save({"encoder": model.encoder.state_dict(), "vocab_size": len(tok.vocab), "config": config}, "models/smiles_pretrainer.pt")


if __name__ == "__main__":
    main()
