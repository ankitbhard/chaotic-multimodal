"""Audit patient-level intersection of CPTAC-CCRCC pathology, RNA and survival.

Uses user-exported manifests; no raw protected data or huge slide downloads.
Standard library only. Run: python -m experiments.cptac_ccrcc.audit --help
"""
import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


def read_table(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t" if str(path).lower().endswith(".tsv") else ",")
        if not reader.fieldnames:
            raise ValueError(f"No headers in {path}")
        return list(reader), reader.fieldnames


def norm_id(value):
    # Preserve full CPTAC patient ID; never truncate to an aliquot or TCGA-like prefix.
    return re.sub(r"\s+", "", str(value or "")).upper()


def collect_ids(path, id_col):
    rows, columns = read_table(path)
    if id_col not in columns:
        raise ValueError(f"{path}: missing {id_col!r}; columns={columns}")
    ids = [norm_id(row[id_col]) for row in rows if norm_id(row[id_col])]
    return set(ids), Counter(ids), len(rows)


def valid_survival(row, time_col, event_col, event_positive, event_negative):
    raw_time = str(row.get(time_col, "")).strip()
    raw_event = str(row.get(event_col, "")).strip().lower()
    try:
        t = float(raw_time)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(t) or t <= 0:
        return None
    if raw_event in event_positive:
        event = 1
    elif raw_event in event_negative:
        event = 0
    else:
        return None
    return t, event


def audit(args):
    slides, slide_counts, n_slides = collect_ids(args.slides, args.slides_id)
    genes, gene_counts, n_genes = collect_ids(args.genes, args.genes_id)
    clinical_rows, columns = read_table(args.clinical)
    required = {args.clinical_id, args.time_col, args.event_col}
    if not required.issubset(columns):
        raise ValueError(f"Clinical file missing columns {sorted(required - set(columns))}; available={columns}")
    positive = {x.strip().lower() for x in args.event_positive.split(",")}
    negative = {x.strip().lower() for x in args.event_negative.split(",")}
    if positive & negative:
        raise ValueError("Event-positive and event-negative values overlap")
    clinical_valid = {}
    conflicting = set()
    clinical_ids = set()
    invalid = Counter()
    for row in clinical_rows:
        pid = norm_id(row[args.clinical_id])
        if not pid:
            invalid["missing_patient_id"] += 1
            continue
        clinical_ids.add(pid)
        outcome = valid_survival(row, args.time_col, args.event_col, positive, negative)
        if outcome is None:
            invalid["missing_or_invalid_survival"] += 1
            continue
        if pid in clinical_valid and clinical_valid[pid] != outcome:
            conflicting.add(pid)
        clinical_valid[pid] = outcome
    for pid in conflicting:
        clinical_valid.pop(pid, None)
    matched = sorted(slides & genes & set(clinical_valid))
    overlap = slides & genes & clinical_ids
    events = sum(clinical_valid[pid][1] for pid in matched)
    report = {
        "cohort": "CPTAC-CCRCC",
        "source_rows": {"slides": n_slides, "genes": n_genes, "clinical": len(clinical_rows)},
        "unique_patients": {"slides": len(slides), "genes": len(genes), "clinical": len(clinical_ids),
                            "valid_survival": len(clinical_valid)},
        "overlap": {"slides_and_genes": len(slides & genes),
                    "slides_and_clinical": len(slides & clinical_ids),
                    "genes_and_clinical": len(genes & clinical_ids),
                    "all_three_before_survival_filter": len(overlap),
                    "all_three_with_valid_survival": len(matched)},
        "eligible_events": events,
        "eligible_censored": len(matched) - events,
        "eligible_event_fraction": events / len(matched) if matched else None,
        "duplicate_patient_rows": {
            "slides": sum(v-1 for v in slide_counts.values()),
            "genes": sum(v-1 for v in gene_counts.values())},
        "conflicting_clinical_patient_count": len(conflicting),
        "invalid_clinical_rows": dict(invalid),
        "survival_time_unit": args.time_unit,
        "notes": [
            "Manifest presence is not proof that the WSI and RNA measurements are downloadable, compatible or tumor-derived.",
            "Survival must have a consistent endpoint, origin and time unit across patients.",
            "This audit does not train or externally validate SAMO.",
            "Do not tune model selection or preprocessing on external test outcomes."
        ]
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eligibility_report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "eligible_patients.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "survival_time", "event"])
        for pid in matched:
            writer.writerow([pid, clinical_valid[pid][0], clinical_valid[pid][1]])
    print(json.dumps(report, indent=2))
    if len(matched) < args.min_patients or events < args.min_events:
        print("STATUS: INSUFFICIENT under supplied feasibility thresholds")
        return 2
    print("STATUS: FEASIBILITY THRESHOLDS MET (not proof of external-validation readiness)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides", required=True, help="CSV/TSV with one row per slide or patient")
    parser.add_argument("--genes", required=True, help="CSV/TSV with one row per RNA sample or patient")
    parser.add_argument("--clinical", required=True, help="CSV/TSV with patient-level survival data")
    parser.add_argument("--slides-id", default="patient_id")
    parser.add_argument("--genes-id", default="patient_id")
    parser.add_argument("--clinical-id", default="patient_id")
    parser.add_argument("--time-col", default="os_months")
    parser.add_argument("--event-col", default="vital_status")
    parser.add_argument("--event-positive", default="1,dead,deceased")
    parser.add_argument("--event-negative", default="0,alive,living")
    parser.add_argument("--time-unit", default="months", choices=["days", "months", "years"])
    parser.add_argument("--min-patients", type=int, default=100,
                        help="Exploratory feasibility threshold, not statistical power calculation")
    parser.add_argument("--min-events", type=int, default=30,
                        help="Exploratory feasibility threshold, not statistical power calculation")
    parser.add_argument("--out", default="results/cptac_ccrcc_audit")
    args = parser.parse_args()
    if args.min_patients < 1 or args.min_events < 1:
        parser.error("Thresholds must be positive")
    raise SystemExit(audit(args))


if __name__ == "__main__":
    main()
