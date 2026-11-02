"""HAM10000 dataset: dermoscopic images + tabular metadata for 7-class lesion classification.

Splits grouped by lesion_id to prevent data leakage (same lesion may have
multiple images). Stratified at the group level by majority label.
"""

import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Localization categories present in HAM10000
LOCALIZATION_CATS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp", "trunk",
    "upper extremity", "unknown",
]
LOC_TO_IDX = {c: i for i, c in enumerate(LOCALIZATION_CATS)}

SEX_CATS = ["male", "female", "unknown"]
SEX_TO_IDX = {c: i for i, c in enumerate(SEX_CATS)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _eval_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class HAM10000Dataset(Dataset):
    """HAM10000 dataset returning ({"image": [3,224,224], "meta": [D]}, {"label": int}).

    Args:
        records: list of dicts with keys image_id, dx, age, sex, localization
        image_dirs: list of Path directories to search for ISIC images
        transform: torchvision transform for images
        age_mean: mean age from training set (for normalization)
        age_std: std age from training set (for normalization)
    """

    def __init__(self, records, image_dirs, transform, age_mean, age_std):
        self.records = records
        self.image_dirs = [Path(d) for d in image_dirs]
        self.transform = transform
        self.age_mean = age_mean
        self.age_std = max(age_std, 1e-6)
        # Precompute metadata dimension
        # 1 (age) + len(SEX_CATS) + len(LOCALIZATION_CATS) + 1 (age_missing flag)
        self.meta_dim = 1 + len(SEX_CATS) + len(LOCALIZATION_CATS) + 1
        # Build image_id -> path lookup for fast access
        self._path_cache = {}
        for rec in records:
            iid = rec["image_id"]
            for d in self.image_dirs:
                p = d / f"{iid}.jpg"
                if p.exists():
                    self._path_cache[iid] = p
                    break

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # --- Image ---
        img_path = self._path_cache.get(rec["image_id"])
        if img_path is None:
            raise FileNotFoundError(
                f"Image {rec['image_id']}.jpg not found in {self.image_dirs}")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # --- Metadata ---
        meta = torch.zeros(self.meta_dim)
        # Age (scaled)
        age = rec.get("age")
        if age is not None and np.isfinite(age):
            meta[0] = (age - self.age_mean) / self.age_std
        else:
            meta[-1] = 1.0  # age_missing flag

        # Sex one-hot
        sex = str(rec.get("sex", "unknown")).lower().strip()
        sex_idx = SEX_TO_IDX.get(sex, SEX_TO_IDX["unknown"])
        meta[1 + sex_idx] = 1.0

        # Localization one-hot
        loc = str(rec.get("localization", "unknown")).lower().strip()
        loc_idx = LOC_TO_IDX.get(loc, LOC_TO_IDX["unknown"])
        meta[1 + len(SEX_CATS) + loc_idx] = 1.0

        label = CLASS_TO_IDX[rec["dx"]]
        return {"image": img, "meta": meta}, {"label": label}


def build_ham10000_splits(data_dir, seed=42, val_frac=0.15, test_frac=0.15):
    """Build train/val/test splits grouped by lesion_id, stratified by label.

    Args:
        data_dir: path containing HAM10000_metadata.csv and image directories
        seed: random seed for reproducibility
        val_frac: fraction for validation
        test_frac: fraction for test

    Returns:
        dict with keys: train, val, test (Dataset objects),
                        meta_dim, class_names, class_weights, metadata
    """
    data_dir = Path(data_dir)
    meta_csv = data_dir / "HAM10000_metadata.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(
            f"HAM10000_metadata.csv not found at {meta_csv}. "
            f"Run download_instructions() for setup help."
        )

    df = pd.read_csv(meta_csv)

    # Collect all directories containing ISIC images
    image_dirs = []
    for candidate in [
        data_dir,
        data_dir / "images",
        data_dir / "HAM10000_images",
        data_dir / "HAM10000_images_part_1",
        data_dir / "HAM10000_images_part_2",
        data_dir / "ham10000_images_part_1",
        data_dir / "ham10000_images_part_2",
    ]:
        if candidate.is_dir() and any(candidate.glob("ISIC_*.jpg")):
            image_dirs.append(candidate)
    if not image_dirs:
        raise FileNotFoundError(
            f"No image directory found under {data_dir}. "
            f"Expected ISIC_*.jpg files."
        )

    # Group by lesion_id, assign majority label per group
    records = []
    for _, row in df.iterrows():
        records.append({
            "image_id": row["image_id"],
            "lesion_id": row["lesion_id"],
            "dx": row["dx"],
            "age": row.get("age", np.nan),
            "sex": row.get("sex", "unknown"),
            "localization": row.get("localization", "unknown"),
        })

    # Build group-level split
    group_labels = {}
    group_records = {}
    for rec in records:
        lid = rec["lesion_id"]
        if lid not in group_records:
            group_records[lid] = []
        group_records[lid].append(rec)
        if lid not in group_labels:
            group_labels[lid] = []
        group_labels[lid].append(rec["dx"])

    # Majority label per group
    group_majority = {}
    for lid, labels in group_labels.items():
        counts = Counter(labels)
        group_majority[lid] = counts.most_common(1)[0][0]

    # Stratified group-level split
    rng = np.random.RandomState(seed)
    lesion_ids = sorted(group_majority.keys())
    majority_labels = [group_majority[lid] for lid in lesion_ids]

    # Group by class for stratification
    class_groups = {}
    for lid, label in zip(lesion_ids, majority_labels):
        if label not in class_groups:
            class_groups[label] = []
        class_groups[label].append(lid)

    train_ids, val_ids, test_ids = [], [], []
    for cls_name, lids in class_groups.items():
        lids = list(lids)
        rng.shuffle(lids)
        n = len(lids)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        test_ids.extend(lids[:n_test])
        val_ids.extend(lids[n_test:n_test + n_val])
        train_ids.extend(lids[n_test + n_val:])

    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    test_ids_set = set(test_ids)

    # Verify no leakage
    assert not (train_ids_set & val_ids_set), "Train/val lesion overlap!"
    assert not (train_ids_set & test_ids_set), "Train/test lesion overlap!"
    assert not (val_ids_set & test_ids_set), "Val/test lesion overlap!"

    train_recs = [r for r in records if r["lesion_id"] in train_ids_set]
    val_recs = [r for r in records if r["lesion_id"] in val_ids_set]
    test_recs = [r for r in records if r["lesion_id"] in test_ids_set]

    # Fit age statistics on training set only
    train_ages = [r["age"] for r in train_recs
                  if r["age"] is not None and np.isfinite(r["age"])]
    age_mean = float(np.mean(train_ages)) if train_ages else 0.0
    age_std = float(np.std(train_ages)) if train_ages else 1.0

    # Compute class weights (inverse frequency) for training
    train_labels = [CLASS_TO_IDX[r["dx"]] for r in train_recs]
    label_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * max(label_counts.get(i, 1), 1))
         for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    )

    train_ds = HAM10000Dataset(
        train_recs, image_dirs, _train_transforms(), age_mean, age_std)
    val_ds = HAM10000Dataset(
        val_recs, image_dirs, _eval_transforms(), age_mean, age_std)
    test_ds = HAM10000Dataset(
        test_recs, image_dirs, _eval_transforms(), age_mean, age_std)

    metadata = {
        "age_mean": age_mean,
        "age_std": age_std,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "n_train_groups": len(train_ids),
        "n_val_groups": len(val_ids),
        "n_test_groups": len(test_ids),
        "meta_dim": train_ds.meta_dim,
        "class_weights": class_weights.tolist(),
    }

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "meta_dim": train_ds.meta_dim,
        "class_names": CLASSES,
        "class_weights": class_weights,
        "metadata": metadata,
    }


def download_instructions():
    """Print instructions for obtaining the HAM10000 dataset."""
    print("""
HAM10000 Dataset Download Instructions
=======================================

Option 1: Kaggle (recommended)
  pip install kaggle
  kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
  unzip skin-cancer-mnist-ham10000.zip -d data/ham10000/

Option 2: Harvard Dataverse
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Expected layout after download:
  data/ham10000/
    HAM10000_metadata.csv
    HAM10000_images_part_1/
      ISIC_0024306.jpg
      ...
    HAM10000_images_part_2/
      ISIC_0024307.jpg
      ...

Note: Never commit credentials or API keys. Use `kaggle.json` from
~/.kaggle/ for authenticated downloads.
""")
