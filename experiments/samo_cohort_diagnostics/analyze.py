"""Analyze SAMO cohort characteristics and validation-only objective selection.

Run from repository root. Does not retrain or alter existing experiments.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import yaml
from train_survival import build_survival_dataloaders
from experiments.samo.objectives import comparable_pairs

METHODS = ("baseline", "aux", "rank", "aux_rank", "adaptive_aux", "samo")


def labels_from_loader(loader):
    dataset = loader.dataset
    times, events = [], []
    # Sequential indexing avoids depending on loader shuffle/drop_last.
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        times.append(float(y["os_months"]))
        events.append(float(y["vital_status"]))
    return np.asarray(times), np.asarray(events)


def characteristics(times, events):
    n = len(times)
    e = events.astype(bool)
    if n == 0:
        raise ValueError("Empty dataset")
    # Distinct individuals, strictly ordered durations, observed earlier event.
    pairs = int(np.sum(e[:, None] & (times[:, None] < times[None, :])))
    return dict(n_patients=n, n_events=int(e.sum()), n_censored=int((~e).sum()),
                censoring_fraction=float((~e).mean()),
                n_comparable_pairs=pairs,
                comparable_pair_fraction=float(pairs / (n * (n - 1))) if n > 1 else 0.0,
                median_followup_months=float(np.median(times)),
                median_event_months=float(np.median(times[e])) if e.any() else None)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def analyze(args):
    rows, missing = [], []
    for cancer in args.cancers:
        cfg = yaml.safe_load((Path("configs") / (cancer + ".yaml")).read_text())
        for fold in range(args.folds):
            data = build_survival_dataloaders(
                uni_dir=cfg["uni_dir"], gene_csv_paths=cfg["gene_csv_paths"],
                image_dim=cfg["image_dim"], n_top_genes=cfg.get("n_top_genes", 20000),
                batch_size=cfg.get("batch_size", 32), seed=args.seed,
                fold=fold, n_folds=args.folds)
            stats = {}
            for split in ("train", "val", "test"):
                t, e = labels_from_loader(data[split])
                stats.update({split + "_" + k: v for k, v in characteristics(t, e).items()})
            # Objective selection uses ONLY validation trajectory. Do not read test
            # performance until the objective is fixed.
            candidates = []
            for method in METHODS:
                root = Path(args.samo_results) / cancer / f"fold_{fold}" / method
                epoch_file = root / "epochs.csv"
                summary_file = root / "summary.json"
                if not epoch_file.exists() or not summary_file.exists():
                    missing.append(str(root))
                    continue
                epoch_rows = read_rows(epoch_file)
                eligible = [r for r in epoch_rows if int(r["epoch"]) <= args.selection_epoch]
                if not eligible:
                    raise ValueError(f"No eligible validation epochs in {epoch_file}")
                chosen_epoch = max(eligible, key=lambda r: float(r["val_ci"]))
                candidates.append((method, float(chosen_epoch["val_ci"]),
                                   int(chosen_epoch["epoch"]), summary_file))
            if len(candidates) != len(METHODS):
                continue
            selected = max(candidates, key=lambda x: (x[1], -METHODS.index(x[0])))
            # IMPORTANT: Existing summaries store test performance at each method's
            # own best-validation checkpoint across all 50 epochs, not necessarily
            # the selection-epoch checkpoint. Never label it selected test performance.
            baseline = next(c for c in candidates if c[0] == "baseline")
            rows.append(dict(cancer=cancer, fold=fold, **stats,
                             selected_method=selected[0],
                             selected_val_ci=selected[1],
                             selected_val_epoch=selected[2],
                             baseline_val_ci=baseline[1],
                             selected_val_gain=selected[1]-baseline[1],
                             **{f"val_ci_{m}": next(c[1] for c in candidates if c[0] == m)
                                for m in METHODS}))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output / "cohort_objective_features.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {"folds_analyzed": len(rows), "missing": missing,
              "selection_epoch": args.selection_epoch,
              "selection_counts": dict(Counter(r["selected_method"] for r in rows)),
              "mean_validation_gain_vs_cox": float(np.mean([r["selected_val_gain"] for r in rows]))
                  if rows else None,
              "warning": ("Exploratory in-sample analysis only. Selecting an objective on "
                          "validation and assessing the same validation set is optimistic. "
                          "Existing test summaries use checkpoints chosen over all epochs "
                          "and cannot establish unbiased selected-model test performance. "
                          "For confirmatory evaluation, rerun selected objectives using "
                          "nested validation and evaluate test exactly once.")}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cancers", nargs="+", default=["brca", "kirc", "stad"])
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selection-epoch", type=int, default=50)
    p.add_argument("--samo-results", default="results/samo_survival")
    p.add_argument("--out", default="results/samo_cohort_diagnostics")
    args = p.parse_args()
    if args.selection_epoch < 1:
        p.error("--selection-epoch must be >= 1")
    analyze(args)


if __name__ == "__main__":
    main()
