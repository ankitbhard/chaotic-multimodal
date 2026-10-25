"""Aggregate completed SAMO runs without touching training results."""
import argparse,csv,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",default="results/samo_survival")
    p.add_argument("--out",default="experiments/samo/RESULTS_SUMMARY.csv")
    args=p.parse_args()
    rows=[]
    for path in sorted(Path(args.results).glob("*/fold_*/*/summary.json")):
        item=json.loads(path.read_text())
        if item.get("completed"):rows.append(item)
    if not rows:raise RuntimeError("No completed results")
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader();writer.writerows(rows)
    print("Summarized",len(rows),"completed runs into",out)
if __name__=="__main__":main()
