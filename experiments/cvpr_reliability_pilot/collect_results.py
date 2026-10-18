"""Quick script to collect and print all pilot results."""
import json, os, sys
import numpy as np

results_base = sys.argv[1] if len(sys.argv) > 1 else "results/cvpr_reliability_pilot"

for cancer in ["brca", "kirc", "luad"]:
    d = os.path.join(results_base, cancer)
    methods = {}
    for fold in range(3):
        fd = os.path.join(d, f"fold_{fold}")
        if not os.path.isdir(fd):
            continue
        for f in sorted(os.listdir(fd)):
            if f.endswith(".json"):
                m = f.replace(".json", "")
                with open(os.path.join(fd, f)) as fh:
                    r = json.load(fh)
                if m not in methods:
                    methods[m] = {"test": [], "val": [], "ep": [], "rel": []}
                methods[m]["test"].append(r.get("test_cindex", 0))
                methods[m]["val"].append(r.get("best_val_cindex", 0))
                methods[m]["ep"].append(r.get("best_epoch", 0))
                rel = r.get("test_reliability") or {}
                if rel:
                    methods[m]["rel"].append(rel.get("alpha_img_mean", 0.5))

    if methods:
        print(f"\n{cancer.upper()} ({len(list(methods.values())[0]['test'])} folds):")
        for m in sorted(methods, key=lambda x: -np.mean(methods[x]["test"])):
            tc = methods[m]["test"]
            vc = methods[m]["val"]
            gap = np.mean(vc) - np.mean(tc)
            ep = np.mean(methods[m]["ep"])
            rel_str = ""
            if methods[m]["rel"]:
                rel_str = f"  img_wt={np.mean(methods[m]['rel']):.3f}"
            folds = " ".join(f"{c:.4f}" for c in tc)
            print(f"  {m:18s}: {np.mean(tc):.4f}+/-{np.std(tc):.4f}  "
                  f"val={np.mean(vc):.4f}  gap={gap:.3f}  ep={ep:.0f}  "
                  f"[{folds}]{rel_str}")

# Robustness summary
print("\n\nROBUSTNESS SUMMARY:")
for cancer in ["brca", "kirc", "luad"]:
    rob_path = os.path.join(results_base, cancer, "robustness.json")
    if not os.path.exists(rob_path):
        print(f"\n{cancer.upper()}: No robustness data")
        continue
    with open(rob_path) as f:
        rob = json.load(f)

    # Aggregate by method
    method_rob = {}
    for key, results in rob.items():
        parts = key.rsplit("_fold", 1)
        if len(parts) != 2:
            continue
        method = parts[0]
        if method not in method_rob:
            method_rob[method] = {}
        for cond, vals in results.items():
            if isinstance(vals, dict) and "cindex" in vals:
                if cond not in method_rob[method]:
                    method_rob[method][cond] = []
                method_rob[method][cond].append(vals["cindex"])

    if method_rob:
        print(f"\n{cancer.upper()}:")
        header = f"  {'Method':18s} | {'Clean':7s} | {'Img0.5':7s} | {'Gene0.5':7s} | {'NoImg':7s} | {'NoGene':7s}"
        print(header)
        print("  " + "-" * 72)
        for m in sorted(method_rob):
            clean = np.mean(method_rob[m].get("clean", [0.5]))
            img05 = np.mean(method_rob[m].get("noise_image_sigma0.5", [0.5]))
            gene05 = np.mean(method_rob[m].get("noise_gene_sigma0.5", [0.5]))
            no_img = np.mean(method_rob[m].get("missing_image", [0.5]))
            no_gene = np.mean(method_rob[m].get("missing_gene", [0.5]))
            print(f"  {m:18s} | {clean:.4f}  | {img05:.4f}  | {gene05:.4f}  | {no_img:.4f}  | {no_gene:.4f}")
