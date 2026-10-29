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

METHODS=("baseline","aux","rank","aux_rank","adaptive_aux","samo","ra_samo","ra_samo_no_proj")

def loss_parts(outputs, t, e, method, temperature):
    risk=outputs["survival"][:,1]
    fusion=cox_efron_free(risk,t,e)
    zero=fusion*0.
    img=cox_efron_free(outputs["aux_img_risk"],t,e) if "aux" in method or method in ("samo","ra_samo","ra_samo_no_proj") else zero
    gene=cox_efron_free(outputs["aux_gene_risk"],t,e) if "aux" in method or method=="samo" else zero
    rank=ranking_loss(risk,t,e,temperature) if method in ("rank","aux_rank","samo","ra_samo","ra_samo_no_proj") else zero
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
    adaptive=method in ("adaptive_aux","samo","ra_samo","ra_samo_no_proj")
    controller=LossWeightController(initial=(args.aux_weight,args.aux_weight,args.rank_weight),
                                   min_weight=args.min_weight,max_weight=args.max_weight,
                                   warmup=args.warmup)
    weights=[args.aux_weight,args.aux_weight,args.rank_weight]
    history=[]; best=-float("inf");best_state=None;best_epoch=0
    start=time.perf_counter()
    for epoch in range(args.epochs):
        lr=args.lr*0.5*(1+math.cos(math.pi*epoch/args.epochs))
        for group in optimizer.param_groups:group["lr"]=lr
        model.train()
        generator.manual_seed(args.seed+fold*10000+epoch)
        sums=np.zeros(4);n=0
        for inputs,labels in train:
            x={k:v.to(args.device) for k,v in inputs.items()}
            t=labels["os_months"].float().to(args.device)
            e=labels["vital_status"].float().to(args.device)
            optimizer.zero_grad(set_to_none=True)
            parts=loss_parts(model(x),t,e,method,args.temperature)
            loss=parts[0]+weights[0]*parts[1]+weights[1]*parts[2]+weights[2]*parts[3]
            if not torch.isfinite(loss):raise RuntimeError(f"Nonfinite loss {cancer}/{fold}/{method}")
            prepared = None
            if method in ("ra_samo", "ra_samo_no_proj"):
                prepared = prepare_branch_gradients(
                    model, *parts, weights,
                    project=(method == "ra_samo"))
            loss.backward()
            if prepared is not None:
                apply_branch_gradients(prepared)
            optimizer.step()
            sums+=np.array([float(z.detach()) for z in parts]);n+=1
        if n==0:raise RuntimeError("Empty training epoch")
        means=(sums/n).tolist()
        # Update controller only using objectives active in this method.
        if adaptive:
            new=controller.update(epoch,means)
            weights[:2]=new[:2]
            if method in ("samo","ra_samo","ra_samo_no_proj"):weights[2]=new[2]
        val_loss,val_ci=evaluate(model,val,args.device)
        history.append(dict(epoch=epoch+1,train_fusion_loss=means[0],
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
    p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--out",default="results/samo_survival")
    p.add_argument("--skip-existing",action="store_true")
    a=p.parse_args()
    a.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for cancer in a.cancers:
        for fold in range(a.folds):
            for method in a.methods:
                run_one(a,cancer,fold,method)
if __name__=="__main__":main()
