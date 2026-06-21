"""
Prepare TCGA-GBMLGG data for the chaotic-multimodal pipeline.

Combines GBM (Glioblastoma, Grade IV) + LGG (Lower Grade Glioma, Grade II/III).
This is the standard benchmark used in MCAT, SurvPath and other comp-path papers.

Downloads from UCSC Xena:
  - RNA-seq HiSeqV2 for both GBM and LGG (~700 patients combined)
  - Clinical/survival data

Survival label (3-year threshold — captures both GBM and LGG prognosis range):
  1 = died within 3 years (1095 days)
  0 = alive OR survived > 3 years

Subtype label (tumor grade):
  0 = LGG (grade II/III — lower grade glioma)
  1 = GBM (grade IV — glioblastoma)

Output:
  tcga_gbm_data/histology_gene_train.csv

Usage:
  python data/prepare_gbm.py
  python data/prepare_gbm.py --out_dir tcga_gbm_data --survival_threshold 1095
"""

import argparse
import os
import io
import gzip
import urllib.request

import numpy as np
import pandas as pd

XENA_BASE   = "https://tcga.xenahubs.net/download"
GBM_RNA_URL = f"{XENA_BASE}/TCGA.GBM.sampleMap/HiSeqV2.gz"
LGG_RNA_URL = f"{XENA_BASE}/TCGA.LGG.sampleMap/HiSeqV2.gz"
GBM_CLI_URL = f"{XENA_BASE}/TCGA.GBM.sampleMap/GBM_clinicalMatrix"
LGG_CLI_URL = f"{XENA_BASE}/TCGA.LGG.sampleMap/LGG_clinicalMatrix"

N_TOP_GENES = 20000


def download(url, compressed=False):
    print(f"  Downloading: {url.split('/')[-1]}")
    with urllib.request.urlopen(url) as r:
        raw = r.read()
    if compressed:
        raw = gzip.decompress(raw)
    return raw


def shorten(s):
    parts = str(s).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else s


def load_rna(url):
    raw  = download(url, compressed=True)
    df   = pd.read_csv(io.BytesIO(raw), sep="\t", low_memory=False)
    gene = df.columns[0]
    df   = df.set_index(gene).T
    df.index = [shorten(i) for i in df.index]
    df.index.name = "patient_id"
    df   = df[~df.index.duplicated(keep="first")].astype(float)
    return df


def load_clinical(url, subtype_label, survival_threshold):
    raw  = download(url, compressed=False)
    clin = pd.read_csv(io.BytesIO(raw), sep="\t", low_memory=False)
    clin.columns = [c.strip().lower() for c in clin.columns]
    clin["patient_id"] = clin[clin.columns[0]].apply(shorten)

    os_col   = next((c for c in clin.columns if c in ("days_to_death", "os_days",
                     "overall_survival")), None)
    stat_col = next((c for c in clin.columns if "vital_status" in c or "os_status" in c), None)
    print(f"    OS col={os_col}  Status col={stat_col}")

    followup_col = next((c for c in clin.columns if "days_to_last_followup" in c), None)

    def surv_label(row):
        try:
            stat = str(row[stat_col]).lower() if stat_col else ""
            dead = "dead" in stat or "deceas" in stat or stat == "1"
            if dead:
                days = float(row[os_col]) if os_col else np.nan
                if np.isnan(days): return -1
                return 1 if days <= survival_threshold else 0
            else:
                # Alive — use followup to confirm survived past threshold
                fu = float(row[followup_col]) if followup_col else np.nan
                if np.isnan(fu): return -1
                return 0 if fu > survival_threshold else -1  # censored before threshold
        except Exception:
            return -1

    clin["surv_label"]    = clin.apply(surv_label, axis=1)
    clin["subtype_label"] = subtype_label   # 0=LGG, 1=GBM
    return clin.set_index("patient_id")[["surv_label", "subtype_label"]]


def prepare(out_dir: str, survival_threshold: int = 1095):
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/4] Loading GBM clinical...")
    gbm_clin = load_clinical(GBM_CLI_URL, subtype_label=1, survival_threshold=survival_threshold)
    print(f"  GBM survival: {gbm_clin['surv_label'].value_counts().to_dict()}")

    print("\n[2/4] Loading LGG clinical...")
    lgg_clin = load_clinical(LGG_CLI_URL, subtype_label=0, survival_threshold=survival_threshold)
    print(f"  LGG survival: {lgg_clin['surv_label'].value_counts().to_dict()}")

    clin_all = pd.concat([gbm_clin, lgg_clin])
    clin_all = clin_all[~clin_all.index.duplicated(keep="first")]

    print("\n[3/4] Loading RNA-seq (GBM + LGG)...")
    gbm_rna = load_rna(GBM_RNA_URL)
    lgg_rna = load_rna(LGG_RNA_URL)
    print(f"  GBM RNA: {gbm_rna.shape}  LGG RNA: {lgg_rna.shape}")

    # Align genes
    common_genes = gbm_rna.columns.intersection(lgg_rna.columns)
    rna_all = pd.concat([gbm_rna[common_genes], lgg_rna[common_genes]])
    rna_all = rna_all[~rna_all.index.duplicated(keep="first")]
    print(f"  Combined RNA: {rna_all.shape}")

    # Top-variance gene selection
    top_genes = rna_all.var(axis=0).nlargest(N_TOP_GENES).index
    rna_all   = rna_all[top_genes]

    print("\n[4/4] Merging...")
    merged = rna_all.join(clin_all, how="inner")
    merged = merged[merged["surv_label"] != -1]
    print(f"  Patients with valid survival: {len(merged)}")
    print(f"  Survival 0/1:  {merged['surv_label'].value_counts().to_dict()}")
    print(f"  Subtype 0/1:   {merged['subtype_label'].value_counts().to_dict()}")

    merged = merged.reset_index().rename(columns={
        "patient_id":   "sample_id",
        "surv_label":   "label",
    })

    out_path = os.path.join(out_dir, "histology_gene_train.csv")
    merged.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}  ({len(merged)} patients, {N_TOP_GENES} genes)")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",            default="tcga_gbm_data")
    parser.add_argument("--survival_threshold", type=int, default=1095,
                        help="Days for binary survival (default 1095 = 3yr)")
    args = parser.parse_args()
    prepare(args.out_dir, args.survival_threshold)
