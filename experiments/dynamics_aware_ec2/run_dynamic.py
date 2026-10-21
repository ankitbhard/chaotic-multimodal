#!/usr/bin/env python3
"""Run dynamic AdamW with the original data, model, split, and scoring pipeline."""
import argparse
import json
from pathlib import Path
import torch
from experiments.dynamics_aware_ec2.dynamics_optimizer import DynamicsAwareAdamW

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=["both","survival","classification"],default="both")
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--folds",type=int,default=3)
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    p.add_argument("--cancers",nargs="+",default=["brca","kirc","luad","lusc","stad"])
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--weight-decay",type=float,default=.0001)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--log-every",type=int,default=50)
    p.add_argument("--data-dir",default="data/cifar10")
    p.add_argument("--out",default="results/dynamics_aware_adamw")
    p.add_argument("--beta2-min",type=float,default=.95)
    p.add_argument("--sensitivity",type=float,default=3.)
    p.add_argument("--change-ema",type=float,default=.9)
    p.add_argument("--skip-existing",action="store_true")
    args=p.parse_args()
    args.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device.type!="cuda": raise RuntimeError("CUDA GPU required for EC2 experiment")
    from experiments.optimizer_dynamics_overnight import run as base
    old_run_one=base.run_one
    old_diagnostics=base.diagnostics
    old_adamw=torch.optim.AdamW
    class ConfiguredDynamicAdamW(DynamicsAwareAdamW):
        def __init__(self,params,**kwargs):
            super().__init__(params,beta2_min=args.beta2_min,sensitivity=args.sensitivity,
                             change_ema=args.change_ema,**kwargs)
    def dynamic_diagnostics(model,opt,before,raw_grads,lr,prev_update):
        # Original diagnostics assume constant beta2; remove invalid columns.
        rows=old_diagnostics(model,opt,before,raw_grads,lr,prev_update)
        for row in rows:
            for key in ("v_mean","effective_lr_p10","effective_lr_p50","effective_lr_p90"):
                row[key]=float("nan")
        return rows
    def dynamic_run_one(run_args,task,label,fold,optimizer_name,*rest):
        if optimizer_name!="adamw": return
        folder=Path(run_args.out)/task/label/f"fold_{fold}"
        target=folder/"dynamics_aware_adamw"
        if run_args.skip_existing and (target/"summary.json").exists():
            print("SKIP",target,flush=True);return
        source=folder/"adamw"
        if source.exists(): raise RuntimeError(f"Stale temporary output {source}; remove only after inspection")
        old_run_one(run_args,task,label,fold,optimizer_name,*rest)
        summary_path=source/"summary.json"
        if not summary_path.exists(): raise RuntimeError(f"Missing summary: {summary_path}")
        summary=json.loads(summary_path.read_text())
        summary.update(optimizer="dynamics_aware_adamw",beta2_min=run_args.beta2_min,
                       sensitivity=run_args.sensitivity,change_ema=run_args.change_ema)
        summary_path.write_text(json.dumps(summary,indent=2))
        # Preserve result labeling for CSV consumers.
        for name in ("steps.csv",):
            f=source/name
            if f.exists():
                f.write_text(f.read_text().replace(",adamw,",",dynamics_aware_adamw,"))
        if target.exists(): raise RuntimeError(f"Existing target: {target}")
        source.rename(target)
    base.run_one=dynamic_run_one
    base.diagnostics=dynamic_diagnostics
    torch.optim.AdamW=ConfiguredDynamicAdamW
    try:
        for task in (["survival","classification"] if args.task=="both" else [args.task]):
            print("TASK",task,"DEVICE",args.device,flush=True)
            (base.run_survival if task=="survival" else base.run_classification)(args)
    finally:
        base.run_one=old_run_one
        base.diagnostics=old_diagnostics
        torch.optim.AdamW=old_adamw
    print("Finished:",args.out,flush=True)

if __name__=="__main__": main()
