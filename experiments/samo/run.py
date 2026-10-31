"""Matched SAMO survival benchmark. Run from repository root."""
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from train_survival import build_survival_dataloaders, concordance_index
from models.joint_model import build_model
from experiments.cvpr_reliability_pilot.models import JointModelWithAux
from experiments.optimizer_dynamics_overnight.run import seed_all
from experiments.samo.objectives import cox_efron_free, ranking_loss, LossWeightController
from experiments.samo.representation_gradients import prepare_branch_gradients, apply_branch_gradients
from experiments.samo.progress_controller import ProgressController
from experiments.samo.history_controller import HistoryController, load_history
from experiments.samo.censor_aware_controller import CensorAwareController, survival_information

METHODS=("baseline","aux","rank","aux_rank","adaptive_aux","samo","ra_samo","ra_samo_no_proj","ra_samo_v1","ra_samo_v2","ra_samo_v3_0","ra_samo_v3_1","ra_samo_v3_2","ra_samo_v3_3")

def loss_parts(outputs, t, e, method, temperature):
    risk=outputs["survival"][:,1]
    fusion=cox_efron_free(risk,t,e)
    zero=fusion*0.
    img=cox_efron_free(outputs["aux_img_risk"],t,e) if "aux" in method or method.startswith("ra_samo_v3_") or method in ("samo","ra_samo","ra_samo_no_proj","ra_samo_v1","ra_samo_v2") else zero
    gene=cox_efron_free(outputs["aux_gene_risk"],t,e) if "aux" in method or method.startswith("ra_samo_v3_") or method in ("samo","ra_samo","ra_samo_no_proj","ra_samo_v1","ra_samo_v2") else zero
    rank=ranking_loss(risk,t,e,temperature) if method.startswith("ra_samo_v3_") or method in ("rank","aux_rank","samo","ra_samo","ra_samo_no_proj","ra_samo_v1","ra_samo_v2") else zero
    return fusion,img,gene,rank

def evaluate(model, loader, device):
    model.eval()
    risks=[]; times=[]; events=[]; losses=[]
    with torch.no_grad():
        for inputs,labels in loader:
            output=model({k:v.to(device) for k,v in inputs.items()})
            risk=output["survival"][:,1]
            t=labels["os_months"].float().to(device)
            e=labels["vital_status"].float().to(device)
            losses.append(float(cox_efron_free(risk,t,e)))
            risks.extend(risk.cpu().tolist());times.extend(t.cpu().tolist());events.extend(e.cpu().tolist())
    return float(np.mean(losses)),float(concordance_index(times,risks,events))

def evaluate_modality_diagnostics(model, loader, device):
    """Validation-only diagnostics; no training updates or checkpoint selection."""
    if not hasattr(model, "aux_img"):
        return {}
    collected = {key: [] for key in ("fused", "image_aux", "gene_aux",
                                      "image_ablated", "gene_ablated")}
    times, events = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            x = {k: v.to(device) for k, v in inputs.items()}
            full = model(x)
            collected["fused"].extend(full["survival"][:, 1].cpu().tolist())
            collected["image_aux"].extend(full["aux_img_risk"].cpu().tolist())
            collected["gene_aux"].extend(full["aux_gene_risk"].cpu().tolist())
            # Zero the encoded modality via the model's modality_mask interface.
            # This is an out-of-distribution intervention, not causal attribution.
            for name, mask in (("image_ablated", (0., 1.)),
                               ("gene_ablated", (1., 0.))):
                m = x["image"].new_tensor(mask).view(1, 2).expand(x["image"].shape[0], -1)
                out = model(x, modality_mask=m)
                collected[name].extend(out["survival"][:, 1].cpu().tolist())
            times.extend(labels["os_months"].float().tolist())
            events.extend(labels["vital_status"].float().tolist())
    scores = {f"val_ci_{key}": float(concordance_index(times, risks, events))
              for key, risks in collected.items()}
    scores["val_image_ablation_delta"] = scores["val_ci_fused"] - scores["val_ci_image_ablated"]
    scores["val_gene_ablation_delta"] = scores["val_ci_fused"] - scores["val_ci_gene_ablated"]
    return scores


def run_one(args,cancer,fold,method):
    out=Path(args.out)/cancer/f"fold_{fold}"/method
    summary=out/"summary.json"
    if args.skip_existing and summary.exists():
        data=json.loads(summary.read_text())
        if data.get("completed") and data.get("epochs")==args.epochs:
            print("SKIP",out,flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed+fold)
    cfg=yaml.safe_load(Path("configs",cancer+".yaml").read_text())
    data=build_survival_dataloaders(
        uni_dir=cfg["uni_dir"],gene_csv_paths=cfg["gene_csv_paths"],
        image_dim=cfg["image_dim"],n_top_genes=cfg.get("n_top_genes",20000),
        batch_size=cfg.get("batch_size",32),seed=args.seed,fold=fold,n_folds=args.folds)
    generator=torch.Generator()
    train=DataLoader(data["train"].dataset,batch_size=cfg.get("batch_size",32),
                     shuffle=True,drop_last=True,num_workers=0,generator=generator)
    val,test=data["val"],data["test"]
    # Baseline retains exact original model; auxiliary methods add two heads.
    # For matched shared-parameter initialization, initialize original model first
    # and copy its matching tensors into auxiliary model.
    seed_all(args.seed+fold)
    baseline=build_model(data["n_genes"],data["image_dim"],"attention",0.3,verbose=False)
    if method in ("baseline","rank"):
        model=baseline
    else:
        model=JointModelWithAux(data["n_genes"],data["image_dim"],0.3)
        src=baseline.state_dict()
        target=model.state_dict()
        matched={k:v for k,v in src.items() if k in target and target[k].shape==v.shape}
        model.load_state_dict(matched,strict=False)
        if len(matched)==0:raise RuntimeError("No baseline tensors matched auxiliary model")
    model=model.to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    adaptive=method in ("adaptive_aux","samo","ra_samo","ra_samo_no_proj","ra_samo_v1")
    controller=LossWeightController(initial=(args.aux_weight,args.aux_weight,args.rank_weight),
                                   min_weight=args.min_weight,max_weight=args.max_weight,
                                   warmup=args.warmup)
    weights=[args.aux_weight,args.aux_weight,args.rank_weight]
    progress=ProgressController(ema=args.progress_ema,gain=args.progress_gain,
                                min_scale=args.progress_min_scale,
                                max_scale=args.progress_max_scale,warmup=args.warmup)
    progress_scales=(1.0,1.0)
    history_controller=HistoryController(load_history(args.history_files, horizon=args.history_horizon),
        min_weight=args.min_weight,max_weight=args.max_weight,warmup=args.warmup)
    if method=="ra_samo_v2" and not args.history_files:
        raise ValueError("ra_samo_v2 requires --history-files from patient-disjoint development data")
    v3=CensorAwareController(method,aux_weight=args.aux_weight,rank_weight=args.rank_weight,warmup=args.warmup) if method.startswith("ra_samo_v3_") else None
    train_labels=data["train"].dataset
    # Extract only training labels, never validation/test, for survival information.
    train_times=[];train_events=[]
    if v3 is not None:
        for _, label in DataLoader(train_labels,batch_size=128,shuffle=False,num_workers=0):
            train_times.extend(label["os_months"].float().tolist())
            train_events.extend(label["vital_status"].float().tolist())
    info=survival_information(train_times,train_events) if v3 is not None else {}
    history=[]; best=-float("inf");best_state=None;best_epoch=0
    start=time.perf_counter()
    for epoch in range(args.epochs):
        lr=args.lr*0.5*(1+math.cos(math.pi*epoch/args.epochs))
        for group in optimizer.param_groups:group["lr"]=lr
        model.train()
        generator.manual_seed(args.seed+fold*10000+epoch)
        sums=np.zeros(4);n=0
        diag_sums={key:0.0 for key in (
            "image_cosine_aux","image_cosine_rank","image_projected_aux",
            "image_projected_rank","image_relative_update_change",
            "gene_cosine_aux","gene_cosine_rank","gene_projected_aux",
            "gene_projected_rank","gene_relative_update_change")}
        diag_n=0
        primary_norm_sums={"image":0.0,"gene":0.0}
        for inputs,labels in train:
            x={k:v.to(args.device) for k,v in inputs.items()}
            t=labels["os_months"].float().to(args.device)
            e=labels["vital_status"].float().to(args.device)
            optimizer.zero_grad(set_to_none=True)
            parts=loss_parts(model(x),t,e,method,args.temperature)
            effective_weights=(weights[0]*progress_scales[0],weights[1]*progress_scales[1],weights[2]) if method=="ra_samo_v1" else weights
            loss=parts[0]+effective_weights[0]*parts[1]+effective_weights[1]*parts[2]+effective_weights[2]*parts[3]
            if not torch.isfinite(loss):raise RuntimeError(f"Nonfinite loss {cancer}/{fold}/{method}")
            prepared = None
            if method in ("ra_samo", "ra_samo_no_proj","ra_samo_v1"):
                prepared = prepare_branch_gradients(
                    model, *parts, effective_weights,
                    project=(method == "ra_samo"))
            loss.backward()
            if prepared is not None:
                diagnostics=apply_branch_gradients(prepared)
                diag_n+=1
                for branch, diag in diagnostics.items():
                    primary_norm_sums[branch]+=diag.primary_norm
                    for attr in ("cosine_aux","cosine_rank","projected_aux",
                                 "projected_rank","relative_update_change"):
                        diag_sums[f"{branch}_{attr}"]+=float(getattr(diag,attr))
            optimizer.step()
            sums+=np.array([float(z.detach()) for z in parts]);n+=1
        if n==0:raise RuntimeError("Empty training epoch")
        means=(sums/n).tolist()
        # Update controller only using objectives active in this method.
        if adaptive:
            new=controller.update(epoch,means)
            weights[:2]=new[:2]
            if method in ("samo","ra_samo","ra_samo_no_proj","ra_samo_v1"):weights[2]=new[2]
        if method=="ra_samo_v1":
            progress_scales=progress.update(epoch,means[1],means[2])
        if method=="ra_samo_v2":
            weights=history_controller.update(epoch,means,weights)
        if v3 is not None:
            # v3.3 rollout selection is implemented by the next EC2 agent; no validation
            # scores are passed here until matched, isolated rollouts are available.
            weights=list(v3.update(epoch,means,info).weights)
        val_loss,val_ci=evaluate(model,val,args.device)
        diag_means={k:v/diag_n for k,v in diag_sums.items()} if diag_n else {}
        if diag_n:
            diag_means.update({f"{branch}_primary_gradient_norm": value/diag_n
                               for branch,value in primary_norm_sums.items()})
        if args.modality_diagnostics:
            diag_means.update(evaluate_modality_diagnostics(model,val,args.device))
        history.append(dict(**diag_means,v3_mode=(v3.mode if v3 else ""),v3_reason=(v3.state.reason if v3 else ""),v3_event_rate=info.get("event_rate", ""),v3_pair_density=info.get("pair_density", ""),progress_scale_img=progress_scales[0],
                            progress_scale_gene=progress_scales[1],epoch=epoch+1,train_fusion_loss=means[0],
                            train_img_loss=means[1],train_gene_loss=means[2],
                            train_rank_loss=means[3],val_loss=val_loss,val_ci=val_ci,
                            weight_img=weights[0],weight_gene=weights[1],
                            weight_rank=weights[2],lr=lr,elapsed_seconds=time.perf_counter()-start))
        if np.isfinite(val_ci) and val_ci>best:
            best=val_ci;best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"{cancer} fold={fold} {method} epoch={epoch+1}/{args.epochs} val={val_ci:.4f}",flush=True)
    if best_state is None:raise RuntimeError("No finite validation C-index")
    model.load_state_dict(best_state)
    test_loss,test_ci=evaluate(model,test,args.device)
    with (out/"epochs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]))
        writer.writeheader();writer.writerows(history)
    summary.write_text(json.dumps(dict(completed=True,dataset=cancer,fold=fold,method=method,
         epochs=args.epochs,best_epoch=best_epoch,best_val_ci=best,test_ci=test_ci,
         test_loss=test_loss,wall_seconds=time.perf_counter()-start,
         parameter_count=sum(p.numel() for p in model.parameters()),
         lr=args.lr,weight_decay=args.weight_decay,seed=args.seed+fold),indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cancers",nargs="+",default=["brca","kirc","stad"])
    p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    p.add_argument("--folds",type=int,default=3)
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=0.001)
    p.add_argument("--weight-decay",type=float,default=0.0001)
    p.add_argument("--aux-weight",type=float,default=0.1)
    p.add_argument("--rank-weight",type=float,default=0.1)
    p.add_argument("--min-weight",type=float,default=0.02)
    p.add_argument("--max-weight",type=float,default=0.3)
    p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--progress-ema",type=float,default=0.9)
    p.add_argument("--progress-gain",type=float,default=1.0)
    p.add_argument("--progress-min-scale",type=float,default=0.5)
    p.add_argument("--progress-max-scale",type=float,default=1.5)
    p.add_argument("--history-files",nargs="*",default=[],help="Patient-disjoint development histories for v2")
    p.add_argument("--history-horizon",type=int,default=3)
    p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--out",default="results/samo_survival")
    p.add_argument("--skip-existing",action="store_true")
    p.add_argument("--modality-diagnostics",action="store_true",
                   help="Log unimodal and zero-feature-ablation validation C-indices; diagnostic only")
    a=p.parse_args()
    a.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for cancer in a.cancers:
        for fold in range(a.folds):
            for method in a.methods:
                run_one(a,cancer,fold,method)
if __name__=="__main__":main()
