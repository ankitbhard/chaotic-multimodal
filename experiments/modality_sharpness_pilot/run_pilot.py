#!/usr/bin/env python3
"""Pilot on existing TCGA loader; all outputs confined to --output."""
import argparse, copy, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train_survival import build_survival_dataloaders, cox_loss, evaluate
from train_msam_survival import build_paper_optimizer
from models.joint_model import build_model
from experiments.modality_sharpness_pilot.block_sam import BlockSAM
from experiments.modality_sharpness_pilot.chaotic_schedule import SAMSchedule, ScheduledSAM
from experiments.modality_sharpness_pilot.modality_schedule import ModalitySchedule, ModalityScheduledSAM
from experiments.modality_sharpness_pilot.new_optimizer_prototypes import OptimizerExperiment, install_shared_gradient_hooks

NAMES=("sam","msam","block_fixed","block_adaptive","schedule_periodic","schedule_random","schedule_chaotic","schedule_sgd","schedule_full","modality_periodic","modality_random","modality_chaotic","new_sgd","new_adamw","new_adaptive_step","new_gradient_diagnostic")

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def one_epoch(model,loader,opt,name,device):
    model.train(); losses=[]; step_times=[]
    for inp,lab in loader:
        x={k:v.to(device) for k,v in inp.items()}
        os_m=lab["os_months"].to(device).float()
        events=lab["vital_status"].to(device).float()
        def closure():
            out=model(x)
            return cox_loss(out["survival"][:,1],os_m,events)
        if device.type=="cuda": torch.cuda.synchronize()
        step_start=time.perf_counter()
        if name in ("sam","msam"):
            labels={"survival":lab["survival"].to(device),
                    "os_months":os_m,"vital_status":events}
            def legacy_closure():
                out=model(x)
                return cox_loss(out["survival"][:,1],os_m,events),out
            loss,_=opt.step(legacy_closure,x,labels)
        else:
            loss=opt.step(closure)
            opt.scheduler.step()
        if device.type=="cuda": torch.cuda.synchronize()
        step_times.append(time.perf_counter()-step_start)
        losses.append(float(loss.detach()))
    return float(np.mean(losses)),float(np.mean(step_times))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True,help="Existing cohort YAML config")
    p.add_argument("--data-root",default=None,help="Optional EC2 local root to remap original absolute paths")
    p.add_argument("--uni-dir",default=None)
    p.add_argument("--gene-csv",action="append",default=None,help="Repeat for multiple CSV files")
    p.add_argument("--output",required=True)
    p.add_argument("--fold",type=int,default=0)
    p.add_argument("--n-folds",type=int,default=5)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--seed",type=int,default=None)
    p.add_argument("--optimizers",nargs="+",choices=NAMES,default=list(NAMES))
    p.add_argument("--rho-image",type=float,default=0.05)
    p.add_argument("--rho-gene",type=float,default=0.05)
    p.add_argument("--rho",type=float,default=0.05,help="Legacy SAM/M-SAM radius")
    p.add_argument("--sam-fraction",type=float,default=0.5,help="Fraction of steps using SAM in scheduled variants")
    p.add_argument("--image-sam-fraction",type=float,default=0.5)
    p.add_argument("--gene-sam-fraction",type=float,default=0.5)
    p.add_argument("--probe-every",type=int,default=20)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--hvp-diagnostics",action="store_true",help="Record autograd HVP block curvature at probe steps")
    p.add_argument("--curvature-diagnostics",action="store_true",help="Log symmetric block curvature probes; does not alter optimizer updates")
    p.add_argument("--no-budget-match",action="store_true",help="Disable matched block perturbation budget")
    args=p.parse_args()
    if not 0<=args.fold<args.n_folds: p.error("fold must be in [0,n_folds)")
    with open(args.config) as f: cfg=yaml.safe_load(f)
    if args.data_root:
        # Map the known old absolute prefix into a new EC2 data root.
        old="/home/ubuntu/"
        for key in ("uni_dir",):
            if str(cfg[key]).startswith(old): cfg[key]=str(Path(args.data_root)/str(cfg[key])[len(old):])
        cfg["gene_csv_paths"]=[str(Path(args.data_root)/str(v)[len(old):]) if str(v).startswith(old) else v for v in cfg["gene_csv_paths"]]
    if args.uni_dir: cfg["uni_dir"]=args.uni_dir
    if args.gene_csv: cfg["gene_csv_paths"]=args.gene_csv
    if args.seed is not None: cfg["seed"]=args.seed
    cfg["epochs"]=args.epochs
    outdir=Path(args.output).resolve()
    if ROOT==outdir or ROOT in outdir.parents and outdir==ROOT: raise ValueError("Invalid output directory")
    print(json.dumps({"uni_dir":cfg["uni_dir"],"gene_csv_paths":cfg["gene_csv_paths"],
                      "image_dim":cfg["image_dim"],"fold":args.fold,"output":str(outdir)},indent=2))
    if args.dry_run:
        assert Path(cfg["uni_dir"]).is_dir(),"UNI directory missing"
        assert all(Path(x).is_file() for x in cfg["gene_csv_paths"]),"Gene CSV missing"
        print("Paths exist. No training performed.");return
    seed=int(cfg.get("seed",42))+args.fold
    seed_all(seed)
    loaders=build_survival_dataloaders(
        uni_dir=cfg["uni_dir"],gene_csv_paths=cfg["gene_csv_paths"],
        image_dim=cfg["image_dim"],n_top_genes=cfg.get("n_top_genes",20000),
        batch_size=cfg.get("batch_size",32),seed=int(cfg.get("seed",42)),
        fold=args.fold,n_folds=args.n_folds)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": print("WARNING: CPU training will be slow.")
    seed_all(seed)
    prototype=build_model(n_genes=loaders["n_genes"],image_dim=loaders["image_dim"],
                          fusion_type=cfg.get("fusion_type","attention"),dropout=cfg.get("dropout",0.3))
    initial=copy.deepcopy(prototype.state_dict())
    digest=hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in initial.values())).hexdigest()
    outdir.mkdir(parents=True,exist_ok=True)
    manifest={"config":cfg,"args":vars(args),"initial_weights_sha256":digest,
              "device":str(device),"methods":{}}
    for name in args.optimizers:
        seed_all(seed)
        model=build_model(n_genes=loaders["n_genes"],image_dim=loaders["image_dim"],
                          fusion_type=cfg.get("fusion_type","attention"),dropout=cfg.get("dropout",0.3)).to(device)
        model.load_state_dict(initial)
        cfg_run=dict(cfg);cfg_run["lr"]=float(cfg["lr"]);cfg_run["weight_decay"]=float(cfg.get("weight_decay",0))
        cfg_run["epochs"]=args.epochs
        if name in ("sam","msam"):
            opt,_,_=build_paper_optimizer(name,model,cfg_run,len(loaders["train"]),rho=args.rho)
        elif name.startswith("new_"):
            mode="conflict_projection" if name=="new_gradient_diagnostic" else name.removeprefix("new_")
            if name=="new_gradient_diagnostic":
                install_shared_gradient_hooks(model)
            opt=OptimizerExperiment(model,mode,lr=cfg_run["lr"],weight_decay=cfg_run["weight_decay"])
            from optimizers.chaotic_lr_scheduler import CosineAnnealingScheduler
            opt.scheduler=CosineAnnealingScheduler(opt.base,T_max=max(1,args.epochs*len(loaders["train"])))
        elif name.startswith("modality_"):
            mode=name.removeprefix("modality_")
            schedule=ModalitySchedule(mode,args.epochs*len(loaders["train"]),
                image_fraction=args.image_sam_fraction,
                gene_fraction=args.gene_sam_fraction,seed=seed)
            opt=ModalityScheduledSAM(model,schedule,lr=cfg_run["lr"],
                                    weight_decay=cfg_run["weight_decay"],rho=args.rho)
            from optimizers.chaotic_lr_scheduler import CosineAnnealingScheduler
            opt.scheduler=CosineAnnealingScheduler(opt.base,T_max=max(1,args.epochs*len(loaders["train"])))
        elif name.startswith("schedule_"):
            mode=name.removeprefix("schedule_")
            fraction=0.0 if mode=="sgd" else 1.0 if mode=="full" else args.sam_fraction
            schedule=SAMSchedule("periodic" if mode in ("sgd","full") else mode,
                                 args.epochs*len(loaders["train"]),fraction= fraction,seed=seed)
            opt=ScheduledSAM(model,schedule,lr=cfg_run["lr"],
                             weight_decay=cfg_run["weight_decay"],rho=args.rho)
            from optimizers.chaotic_lr_scheduler import CosineAnnealingScheduler
            opt.scheduler=CosineAnnealingScheduler(opt.base,T_max=max(1,args.epochs*len(loaders["train"])))
        else:
            opt=BlockSAM(model,lr=cfg_run["lr"],weight_decay=cfg_run["weight_decay"],
                         rho_image=args.rho_image,rho_gene=args.rho_gene,
                         adaptive=name=="block_adaptive",probe_every=args.probe_every,
                         match_global_budget=not args.no_budget_match,
                         curvature_diagnostics=args.curvature_diagnostics,
                         hvp_diagnostics=args.hvp_diagnostics)
            from optimizers.chaotic_lr_scheduler import CosineAnnealingScheduler
            opt.scheduler=CosineAnnealingScheduler(opt.base,T_max=max(1,args.epochs*len(loaders["train"])))
        # Same shuffle seed for each optimizer; separate model RNG streams may diverge
        # because legacy M-SAM uses extra stochastic operations.
        train_loader=loaders["train"]
        if hasattr(train_loader,"generator"): train_loader.generator=torch.Generator().manual_seed(seed)
        best=-float("inf");best_state=None;history=[];start=time.perf_counter()
        for ep in range(1,args.epochs+1):
            train_loss,mean_step_seconds=one_epoch(model,train_loader,opt,name,device)
            val_loss,val_ci,*_=evaluate(model,loaders["val"],device)
            history.append({"epoch":ep,"train_loss":train_loss,"val_loss":float(val_loss),
                            "val_cindex":float(val_ci),
                            "mean_step_seconds":mean_step_seconds,
                            "lr":float(opt.base.param_groups[0]["lr"]) if name.startswith(("block_","schedule_","modality_","new_")) else None})
            if np.isfinite(val_ci) and val_ci>best:
                best=float(val_ci);best_state=copy.deepcopy(model.state_dict())
            print(f"{name} fold={args.fold} epoch={ep} train={train_loss:.4f} val_ci={val_ci:.4f}",flush=True)
        if best_state is None: raise RuntimeError("No valid validation C-index")
        model.load_state_dict(best_state)
        test_loss,test_ci,*_=evaluate(model,loaders["test"],device)
        result={"optimizer":name,"fold":args.fold,"best_val_cindex":best,
                "test_cindex":float(test_ci),"test_loss":float(test_loss),
                "seconds":time.perf_counter()-start,"initial_weights_sha256":digest,
                "history":history}
        if name.startswith("new_"):
            result["optimizer_history"]=opt.history
            result["note"]="new_gradient_diagnostic logs shared-space gradient cosine only; no projection is applied" if name=="new_gradient_diagnostic" else "Exploratory optimizer baseline"
        if name.startswith(("schedule_","modality_")):
            result["schedule_history"]=opt.history
            result["sam_steps"]=opt.sam_steps
            result["total_steps"]=opt.steps
            if name.startswith("modality_"):
                result["modality_counts"]=opt.schedule.counts
                result["sam_budget"]=opt.schedule.sam_budget
        if name.startswith("block_"):
            result["radius_history"]=opt.history
            if args.curvature_diagnostics: result["curvature_history"]=opt.curvature_history
            if args.hvp_diagnostics: result["hvp_history"]=opt.hvp_history
        with open(outdir/f"{name}_fold{args.fold}.json","w") as f: json.dump(result,f,indent=2)
        manifest["methods"][name]={"test_cindex":float(test_ci)}
    with open(outdir/"manifest.json","w") as f: json.dump(manifest,f,indent=2)
    print("Completed:",outdir)

if __name__=="__main__": main()
