"""Patch-level TCGA survival dataset for CLAM attention MIL experiments.

Provides both MeanPool and CLAM datasets from shared patient splits,
ensuring identical train/val/test partitions across all 6 methods.

Multi-slide patients (e.g., 2 BRCA patients with 2 WSIs each) have all
slides' patches concatenated before aggregation.
"""

import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Embedding loaders
# ---------------------------------------------------------------------------

def _discover_uni_dir(uni_dir: str) -> str:
    """Find directory containing .pt/.h5 embedding files."""
    for sub in ["uni_features/tcga_features", "uni_features", ""]:
        candidate = os.path.join(uni_dir, sub) if sub else uni_dir
        if os.path.isdir(candidate):
            files = [f for f in os.listdir(candidate)
                     if f.endswith((".pt", ".h5"))]
            if files:
                return candidate
    return uni_dir


def _load_single_file(path: str) -> torch.Tensor:
    """Load one .pt or .h5 → [N, D] tensor."""
    if path.endswith(".pt"):
        t = torch.load(path, map_location="cpu", weights_only=True).float()
    elif path.endswith(".h5"):
        import h5py
        with h5py.File(path, "r") as f:
            key = "features" if "features" in f else list(f.keys())[0]
            t = torch.from_numpy(f[key][:]).float()
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if t.dim() == 3:
        t = t.squeeze(0)       # [1, N, D] → [N, D]
    if t.dim() == 1:
        t = t.unsqueeze(0)     # [D] → [1, D]
    return t


def load_patches(paths: List[str], expected_dim: int,
                 max_patches: int, patient_barcode: str = None) -> torch.Tensor:
    """Load patch features from slide files, subsample if needed.

    Multi-slide patients: patches from all slides are concatenated.
    Subsampling is deterministic per patient (seeded by barcode hash).

    Returns: [N, expected_dim] where N <= max_patches
    """
    all_patches = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            all_patches.append(_load_single_file(path))
        except Exception as e:
            print(f"  [WARN] skipping {os.path.basename(path)}: {e}")

    if not all_patches:
        return torch.zeros(1, expected_dim)

    patches = torch.cat(all_patches, dim=0)     # [total, D]

    if patches.shape[0] > max_patches:
        seed = hash(patient_barcode) % (2**32) if patient_barcode else 0
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(patches.shape[0], generator=g)[:max_patches]
        patches = patches[idx]

    return patches


def load_meanpooled(paths: List[str], expected_dim: int) -> torch.Tensor:
    """Load patches from all slides, mean-pool to [D]."""
    all_patches = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            all_patches.append(_load_single_file(path))
        except Exception:
            pass

    if not all_patches:
        return torch.zeros(expected_dim)

    patches = torch.cat(all_patches, dim=0)
    return patches.mean(dim=0)                   # [D]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class CLAMSurvivalDataset(Dataset):
    """Returns variable-length patch bags per patient."""

    def __init__(self, records, gene_matrix, image_dim, max_patches=4096):
        self.records = records
        self.gene_matrix = gene_matrix.astype(np.float32)
        self.image_dim = image_dim
        self.max_patches = max_patches

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        patches = load_patches(
            rec["uni_paths"], self.image_dim,
            self.max_patches, rec["patient_barcode"])
        genes = torch.tensor(self.gene_matrix[idx], dtype=torch.float32)
        return (
            {"patches": patches, "gene": genes},
            {"os_months": rec["os_months"],
             "vital_status": rec["vital_status"],
             "survival": rec["surv_label"]},
        )


class MeanPoolSurvivalDataset(Dataset):
    """Returns mean-pooled slide embeddings per patient."""

    def __init__(self, records, gene_matrix, image_dim):
        self.records = records
        self.gene_matrix = gene_matrix.astype(np.float32)
        self.image_dim = image_dim
        self._cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        key = rec["patient_barcode"]
        if key not in self._cache:
            self._cache[key] = load_meanpooled(
                rec["uni_paths"], self.image_dim)
        genes = torch.tensor(self.gene_matrix[idx], dtype=torch.float32)
        return (
            {"image": self._cache[key], "gene": genes},
            {"os_months": rec["os_months"],
             "vital_status": rec["vital_status"],
             "survival": rec["surv_label"]},
        )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def clam_collate_fn(batch):
    """Pack variable-length patch bags into flat tensor + counts."""
    inputs_list, labels_list = zip(*batch)
    all_patches = torch.cat([inp["patches"] for inp in inputs_list], dim=0)
    patch_counts = torch.tensor(
        [inp["patches"].shape[0] for inp in inputs_list], dtype=torch.long)
    genes = torch.stack([inp["gene"] for inp in inputs_list])
    labels = {}
    for key in labels_list[0]:
        vals = [lab[key] for lab in labels_list]
        if isinstance(vals[0], (int, float, np.integer, np.floating)):
            labels[key] = torch.tensor(vals)
        else:
            labels[key] = torch.stack(vals)
    return (
        {"patches": all_patches, "patch_counts": patch_counts, "gene": genes},
        labels,
    )


# ---------------------------------------------------------------------------
# Shared split builder
# ---------------------------------------------------------------------------

def build_split(uni_dir: str, gene_csv_paths: List[str],
                image_dim: int = 1024, n_top_genes: int = 20000,
                seed: int = 42, fold: int = 0, n_folds: int = 5):
    """Build patient records and gene data with shared train/val/test splits.

    All 6 experiment methods use the same split output.

    Returns dict with keys:
        train_recs, val_recs, test_recs   : lists of record dicts
        train_genes, val_genes, test_genes : np.ndarray [N, n_genes]
        n_genes, image_dim                 : ints
    """
    # Load gene CSVs
    frames = [pd.read_csv(p) for p in gene_csv_paths if os.path.exists(p)]
    if not frames:
        raise FileNotFoundError(f"No gene CSVs: {gene_csv_paths}")
    df = pd.concat(frames, ignore_index=True)
    df["patient_barcode"] = df["sample_id"].str[:12]
    df = df.drop_duplicates(subset="patient_barcode").reset_index(drop=True)
    df = df.dropna(subset=["OS_months", "vital_status"]).reset_index(drop=True)

    # Discover embedding files — group by patient barcode
    emb_files = {}
    try:
        emb_dir = _discover_uni_dir(uni_dir)
        for fname in sorted(os.listdir(emb_dir)):
            if fname.endswith((".pt", ".h5")):
                barcode = fname[:12]
                path = os.path.join(emb_dir, fname)
                emb_files.setdefault(barcode, []).append(path)
    except (FileNotFoundError, OSError):
        pass

    print(f"  Embedding patients: {len(emb_files)}")

    # Non-gene columns
    NON_GENE = {
        "sample_id", "patient_barcode", "label", "survival_label",
        "subtype_label", "uni_path", "image_path", "OS_months",
        "vital_status", "subtype_raw", "patient_id", "magnification",
    }
    gene_cols = [c for c in df.columns
                 if c not in NON_GENE and pd.api.types.is_numeric_dtype(df[c])]

    surv_col = "survival_label" if "survival_label" in df.columns else "label"

    # Match patients
    records, gene_rows = [], []
    for barcode in sorted(emb_files.keys()):
        row = df[df["patient_barcode"] == barcode]
        if row.empty:
            continue
        row = row.iloc[0]
        records.append({
            "uni_paths":        emb_files[barcode],
            "patient_barcode":  barcode,
            "surv_label":       int(row.get(surv_col, 0)),
            "os_months":        float(row["OS_months"]),
            "vital_status":     int(row["vital_status"]),
        })
        gene_rows.append(
            row[gene_cols].infer_objects(copy=False)
            .fillna(0).values.astype(np.float32))

    if not records:
        raise RuntimeError("No patients matched.")

    events = sum(r["vital_status"] for r in records)
    print(f"  Matched patients: {len(records)}  (events={events})")

    # Stratified k-fold split
    labels = [r["vital_status"] for r in records]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    trainval_idx, test_idx = list(skf.split(range(len(records)), labels))[fold]
    trainval_idx = list(trainval_idx)
    test_idx = list(test_idx)

    rng = random.Random(seed)
    rng.shuffle(trainval_idx)
    n_val = int(len(trainval_idx) * 0.15)
    val_idx = trainval_idx[:n_val]
    train_idx = trainval_idx[n_val:]

    print(f"  Fold {fold}/{n_folds}: {len(train_idx)} train / "
          f"{len(val_idx)} val / {len(test_idx)} test")

    # Gene preprocessing
    gene_matrix = np.stack(gene_rows, axis=0)
    k = min(n_top_genes, gene_matrix.shape[1])
    var_idx = np.argsort(gene_matrix[train_idx].var(axis=0))[::-1][:k]
    scaler = StandardScaler()
    train_genes = scaler.fit_transform(
        gene_matrix[train_idx][:, var_idx]).astype(np.float32)
    val_genes = scaler.transform(
        gene_matrix[val_idx][:, var_idx]).astype(np.float32)
    test_genes = scaler.transform(
        gene_matrix[test_idx][:, var_idx]).astype(np.float32)

    n_genes = train_genes.shape[1]
    print(f"  n_genes={n_genes}")

    return {
        "train_recs":  [records[i] for i in train_idx],
        "val_recs":    [records[i] for i in val_idx],
        "test_recs":   [records[i] for i in test_idx],
        "train_genes": train_genes,
        "val_genes":   val_genes,
        "test_genes":  test_genes,
        "n_genes":     n_genes,
        "image_dim":   image_dim,
    }


# ---------------------------------------------------------------------------
# Dataloader factories
# ---------------------------------------------------------------------------

def build_clam_loaders(split: dict, batch_size: int = 8,
                       max_patches: int = 4096, num_workers: int = 0):
    """Build CLAM dataloaders from a split dict."""
    def _loader(recs, genes, shuffle):
        ds = CLAMSurvivalDataset(recs, genes, split["image_dim"], max_patches)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=clam_collate_fn,
                          pin_memory=False)
    return {
        "train": _loader(split["train_recs"], split["train_genes"], True),
        "val":   _loader(split["val_recs"],   split["val_genes"],   False),
        "test":  _loader(split["test_recs"],  split["test_genes"],  False),
    }


def build_meanpool_loaders(split: dict, batch_size: int = 32,
                           num_workers: int = 0):
    """Build MeanPool dataloaders from a split dict."""
    def _loader(recs, genes, shuffle):
        ds = MeanPoolSurvivalDataset(recs, genes, split["image_dim"])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=False)
    return {
        "train": _loader(split["train_recs"], split["train_genes"], True),
        "val":   _loader(split["val_recs"],   split["val_genes"],   False),
        "test":  _loader(split["test_recs"],  split["test_genes"],  False),
    }
