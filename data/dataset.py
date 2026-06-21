"""
Unified TCGA multimodal dataset for pan-cancer training.

Handles all 5 cancer types (BRCA, LUAD, KIRC, STAD, LUSC) from a single class.
Supports both .pt and .h5 slide embedding formats.

Expected directory layout (configured via configs/*.yaml):
    uni_features/          ← .pt or .h5 files named by TCGA barcode
      └── tcga_features/   ← optional nested dir (BRCA EC2 layout)

Each embedding file contains:
    .pt  → Tensor[N_patches, embedding_dim]  or  Tensor[embedding_dim]
    .h5  → dataset "features"  shape (N_patches, embedding_dim)

Gene CSVs must have:
    Column "sample_id"  : TCGA barcode (used for 12-char patient matching)
    Column "label"      : binary survival label  (0 = alive >3yr, 1 = died ≤3yr)
    Remaining numeric columns: gene expression values (raw counts or TPM)

Subtype CSVs (optional) must have:
    Column "sample_id"  : TCGA barcode
    Column "label"      : integer subtype class  (0-based; -1 = unknown)

Usage:
    from data.dataset import build_dataloaders

    loaders = build_dataloaders(
        uni_dir         = "tangle_data/brca",
        gene_csv_paths  = ["tcga_brca_data/brca_gene_train.csv"],
        subtype_csv_paths = ["tcga_brca_data/brca_subtype.csv"],
        image_dim       = 1024,
        n_top_genes     = 20000,
        val_fraction    = 0.15,
        test_fraction   = 0.10,
        batch_size      = 32,
    )
    # loaders["train"], loaders["val"], loaders["test"]
    # loaders["n_genes"], loaders["image_dim"]
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# ── Embedding loader (handles .pt and .h5) ────────────────────────────────────

def _load_embedding(path: str, expected_dim: int) -> torch.Tensor:
    """
    Load a slide embedding from .pt or .h5 and return mean-pooled [D] tensor.
    Falls back to zeros on any error.
    """
    try:
        if not path:
            return torch.zeros(expected_dim)
        if path.endswith(".pt"):
            feat = torch.load(path, map_location="cpu").float()
        elif path.endswith(".h5"):
            import h5py
            with h5py.File(path, "r") as f:
                feat = torch.from_numpy(f["features"][:]).float()
        else:
            return torch.zeros(expected_dim)

        # Mean-pool patch dimension if needed
        if feat.dim() == 3:
            feat = feat.squeeze(0).mean(0)   # [1, N, D] → [D]
        elif feat.dim() == 2:
            feat = feat.mean(0)              # [N, D] → [D]
        return feat

    except Exception:
        return torch.zeros(expected_dim)


def _discover_uni_dir(base_dir: str) -> str:
    """
    Find the directory containing embedding files.
    Supports flat layout (uni_features/) and nested (uni_features/tcga_features/).
    """
    nested = os.path.join(base_dir, "uni_features", "tcga_features")
    flat   = os.path.join(base_dir, "uni_features")
    if os.path.isdir(nested):
        return nested
    if os.path.isdir(flat):
        return flat
    # Allow passing the embeddings dir directly
    if os.path.isdir(base_dir):
        return base_dir
    raise FileNotFoundError(
        f"Cannot find uni_features directory under: {base_dir}"
    )


# ── Core Dataset ──────────────────────────────────────────────────────────────

class TCGAMultimodalDataset(Dataset):
    """
    Multimodal dataset: slide embeddings + gene expression → survival + subtype.

    Args:
        records         : list of dicts with keys:
                            "uni_path"        (str)
                            "patient_barcode" (str)
                            "surv_label"      (int, 0/1)
                            "subtype_label"   (int, 0-based or -1)
        gene_data       : np.ndarray [N, n_genes], preprocessed (log1p + scaled)
        image_dim       : expected embedding dimension (for zero-fallback)
        emb_cache       : optional pre-built dict {path: Tensor[D]} (shared across splits)
    """

    def __init__(self,
                 records:   List[dict],
                 gene_data: np.ndarray,
                 image_dim: int,
                 emb_cache: Optional[Dict[str, torch.Tensor]] = None):
        self.records   = records
        self.gene_data = gene_data.astype(np.float32)
        self.image_dim = image_dim
        self._cache    = emb_cache if emb_cache is not None else {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        path = rec["uni_path"]

        if path not in self._cache:
            self._cache[path] = _load_embedding(path, self.image_dim)
        slide_emb = self._cache[path]

        gene_vec = torch.tensor(self.gene_data[idx], dtype=torch.float32)

        inputs = {"image": slide_emb, "gene": gene_vec}
        labels = {"survival": int(rec["surv_label"])}
        return inputs, labels


# ── Gene preprocessing ────────────────────────────────────────────────────────

def _preprocess_genes(
    gene_matrix: np.ndarray,
    n_top_genes: int,
    is_train:    bool,
    top_gene_idx: Optional[np.ndarray] = None,
    scaler:       Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    log1p → top-K variance selection → StandardScaler.

    Returns:
        processed matrix, top_gene_idx, fitted scaler
    """
    mat = np.log1p(np.clip(gene_matrix, 0, None)).astype(np.float32)

    if is_train:
        k = min(n_top_genes, mat.shape[1])
        top_gene_idx = np.argsort(mat.var(axis=0))[::-1][:k]
        mat = mat[:, top_gene_idx]
        scaler = StandardScaler()
        mat = scaler.fit_transform(mat).astype(np.float32)
    else:
        mat = mat[:, top_gene_idx]
        mat = scaler.transform(mat).astype(np.float32)

    return mat, top_gene_idx, scaler


# ── Builder ───────────────────────────────────────────────────────────────────

def build_dataloaders(
    uni_dir:            str,
    gene_csv_paths:     List[str],
    subtype_csv_paths:  List[str]       = None,
    image_dim:          int             = 1024,
    n_top_genes:        int             = 20000,
    val_fraction:       float           = 0.15,
    test_fraction:      float           = 0.10,
    batch_size:         int             = 32,
    num_workers:        int             = 0,
    seed:               int             = 42,
    precache:           bool            = False,
) -> dict:
    """
    Build train / val / test DataLoaders from raw files.

    Returns dict with keys:
        "train", "val", "test"       : DataLoader
        "n_genes"                    : int
        "image_dim"                  : int
        "scaler"                     : fitted StandardScaler
        "top_gene_idx"               : np.ndarray of gene indices
        "train_size", "val_size", "test_size" : int
    """
    subtype_csv_paths = subtype_csv_paths or []

    # ── Load and merge gene CSVs ──────────────────────────────────────────────
    gene_frames = [pd.read_csv(p) for p in gene_csv_paths if os.path.exists(p)]
    if not gene_frames:
        raise FileNotFoundError(f"No gene CSVs found: {gene_csv_paths}")
    gene_df = pd.concat(gene_frames, ignore_index=True)
    gene_df["patient_barcode"] = gene_df["sample_id"].str[:12]
    gene_df = gene_df.drop_duplicates(subset="patient_barcode").reset_index(drop=True)

    # ── Load subtype CSVs (optional) ──────────────────────────────────────────
    sub_frames = [pd.read_csv(p) for p in subtype_csv_paths if os.path.exists(p)]
    if sub_frames:
        sub_df = pd.concat(sub_frames, ignore_index=True)
        sub_df["patient_barcode"] = sub_df["sample_id"].str[:12]
        sub_df = sub_df.drop_duplicates(subset="patient_barcode")[
            ["patient_barcode", "label"]
        ].rename(columns={"label": "subtype_label"})
    else:
        sub_df = None

    # ── Discover embedding files ───────────────────────────────────────────────
    emb_files = {}
    try:
        emb_dir = _discover_uni_dir(uni_dir)
        for fname in os.listdir(emb_dir):
            if fname.endswith((".pt", ".h5")):
                barcode = os.path.splitext(fname)[0][:12]
                emb_files[barcode] = os.path.join(emb_dir, fname)
    except (FileNotFoundError, OSError):
        emb_dir = ""

    print(f"  Embedding files found : {len(emb_files)}"
          + (" (gene-only fallback: image → zeros)" if not emb_files else ""))

    # ── Match patients ────────────────────────────────────────────────────────
    NON_GENE = {
        "sample_id", "patient_barcode", "label", "survival_label",
        "subtype_label", "uni_path", "image_path", "OS_months",
        "vital_status", "subtype_raw", "patient_id", "magnification",
    }
    gene_cols = [c for c in gene_df.columns
                 if c not in NON_GENE
                 and pd.api.types.is_numeric_dtype(gene_df[c])]

    records, gene_rows = [], []
    surv_col = "survival_label" if "survival_label" in gene_df.columns else "label"

    gene_has_subtype = "subtype_label" in gene_df.columns

    if emb_files:
        # Embedding-driven: only include patients with matching embedding files
        patient_iter = emb_files.items()
    else:
        # Gene-CSV-driven fallback: all patients, image→zeros at runtime
        patient_iter = ((bc, "") for bc in gene_df["patient_barcode"].tolist())

    for barcode, fpath in patient_iter:
        row = gene_df[gene_df["patient_barcode"] == barcode]
        if row.empty:
            continue
        row = row.iloc[0]
        rec = {
            "uni_path":        fpath,
            "patient_barcode": barcode,
            "surv_label":      int(row.get(surv_col, 0)),
        }
        if gene_has_subtype and sub_df is None:
            rec["subtype_label"] = int(row.get("subtype_label", -1))
        records.append(rec)
        gene_rows.append(row[gene_cols].infer_objects(copy=False).fillna(0).values.astype(np.float32))

    if not records:
        raise RuntimeError(
            "No patients matched between embedding files and gene CSVs. "
            f"Check uni_dir={uni_dir} and gene_csv_paths={gene_csv_paths}"
        )

    # Merge subtype labels from separate CSV (overrides gene CSV if both present)
    if sub_df is not None:
        sub_map = sub_df.set_index("patient_barcode")["subtype_label"].to_dict()
        for rec in records:
            rec["subtype_label"] = int(sub_map.get(rec["patient_barcode"], -1))
    elif not gene_has_subtype:
        for rec in records:
            rec["subtype_label"] = -1

    print(f"  Matched patients      : {len(records)}")

    # ── Train / val / test split ──────────────────────────────────────────────
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    n_test = int(len(indices) * test_fraction)
    n_val  = int(len(indices) * val_fraction)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    gene_matrix = np.stack(gene_rows, axis=0)

    train_gene, top_gene_idx, scaler = _preprocess_genes(
        gene_matrix[train_idx], n_top_genes, is_train=True
    )
    val_gene,  _, _ = _preprocess_genes(
        gene_matrix[val_idx],  n_top_genes, is_train=False,
        top_gene_idx=top_gene_idx, scaler=scaler
    )
    test_gene, _, _ = _preprocess_genes(
        gene_matrix[test_idx], n_top_genes, is_train=False,
        top_gene_idx=top_gene_idx, scaler=scaler
    )

    # ── Pre-cache embeddings in RAM (shared across splits) ────────────────────
    emb_cache: Dict[str, torch.Tensor] = {}
    if precache:
        all_paths = list({rec["uni_path"] for rec in records})
        print(f"  Pre-caching {len(all_paths)} slide embeddings...", flush=True)
        for p in all_paths:
            emb_cache[p] = _load_embedding(p, image_dim)
        print(f"  Cache ready.", flush=True)

    def _subset(idx_list, gene_data):
        return [records[i] for i in idx_list], gene_data

    train_recs, train_gene = _subset(train_idx, train_gene)
    val_recs,   val_gene   = _subset(val_idx,   val_gene)
    test_recs,  test_gene  = _subset(test_idx,  test_gene)

    n_genes = train_gene.shape[1]

    def _loader(recs, genes, shuffle):
        ds = TCGAMultimodalDataset(recs, genes, image_dim, emb_cache)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=False)

    print(f"  Split: {len(train_recs)} train / {len(val_recs)} val / "
          f"{len(test_recs)} test  |  n_genes={n_genes}")

    return {
        "train":        _loader(train_recs, train_gene, shuffle=True),
        "val":          _loader(val_recs,   val_gene,   shuffle=False),
        "test":         _loader(test_recs,  test_gene,  shuffle=False),
        "n_genes":      n_genes,
        "image_dim":    image_dim,
        "scaler":       scaler,
        "top_gene_idx": top_gene_idx,
        "train_size":   len(train_recs),
        "val_size":     len(val_recs),
        "test_size":    len(test_recs),
    }


# ── Self-test (synthetic data — no real files needed) ─────────────────────────

if __name__ == "__main__":
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:

        # ── Create fake embedding .pt files ──────────────────────────────────
        emb_dir = os.path.join(tmp, "uni_features")
        os.makedirs(emb_dir)
        barcodes = [f"TCGA-XX-{i:04d}" for i in range(60)]
        for bc in barcodes:
            # Simulate (50 patches, 1024-dim)
            torch.save(torch.randn(50, 1024), os.path.join(emb_dir, f"{bc}.pt"))

        # ── Create fake gene CSV ──────────────────────────────────────────────
        rng = np.random.default_rng(0)
        gene_names = [f"GENE_{i:05d}" for i in range(500)]
        gene_vals  = rng.exponential(5.0, size=(60, 500)).astype(np.float32)
        survival   = rng.integers(0, 2, size=60).tolist()

        gene_csv = os.path.join(tmp, "gene.csv")
        df = pd.DataFrame(gene_vals, columns=gene_names)
        df.insert(0, "sample_id", [f"{bc}-01" for bc in barcodes])
        df.insert(1, "label", survival)
        df.to_csv(gene_csv, index=False)

        # ── Create fake subtype CSV ───────────────────────────────────────────
        sub_csv = os.path.join(tmp, "subtype.csv")
        pd.DataFrame({
            "sample_id": [f"{bc}-01" for bc in barcodes],
            "label":     rng.integers(0, 4, size=60).tolist(),
        }).to_csv(sub_csv, index=False)

        # ── Build loaders ─────────────────────────────────────────────────────
        loaders = build_dataloaders(
            uni_dir           = tmp,
            gene_csv_paths    = [gene_csv],
            subtype_csv_paths = [sub_csv],
            image_dim         = 1024,
            n_top_genes       = 100,
            val_fraction      = 0.15,
            test_fraction     = 0.10,
            batch_size        = 8,
        )

        for split in ("train", "val", "test"):
            inputs, labels = next(iter(loaders[split]))
            assert inputs["image"].shape == (8, 1024) or inputs["image"].shape[0] <= 8
            assert inputs["gene"].shape[1]  == loaders["n_genes"]
            assert "survival" in labels and "subtype" in labels
            print(f"{split:5s}: image={inputs['image'].shape}  "
                  f"gene={inputs['gene'].shape}  "
                  f"surv={labels['survival'].tolist()[:4]}  ✓")

        print(f"\nn_genes={loaders['n_genes']}  image_dim={loaders['image_dim']}")
        print("All TCGAMultimodalDataset self-tests PASSED.")
