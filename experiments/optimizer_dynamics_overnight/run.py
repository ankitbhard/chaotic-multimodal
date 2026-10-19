"""Adam vs AdamW dynamics experiments: cancer survival and CIFAR-10."""
import argparse, csv, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def evaluate_classification(model, loader, device):
    model.eval(); total=correct=n=0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device); logits=model(x)
            total+=torch.nn.functional.cross_entropy(logits,y,reduction="sum").item()
            correct+=(logits.argmax(1)==y).sum().item(); n+=len(y)
    return total/max(n,1), correct/max(n,1)

def diagnostics(model,opt,before,raw_grads,lr,prev_update):
    import torch.nn.functional as F
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad or name not in before: continue
        group=name.split(".")[0]
        d=groups.setdefault(group,{"g":[],"m":[],"v":[],"scale":[],"u":[],"w":[],"oldg":[]})
        state=opt.state.get(p,{})
        if not state or "exp_avg" not in state: continue
        t=state["step"]; t=float(t.item()) if torch.is_tensor(t) else float(t)
        m=state["exp_avg"].detach()/(1-opt.defaults["betas"][0]**t)
        v=state["exp_avg_sq"].detach()/(1-opt.defaults["betas"][1]**t)
        d["g"].append(raw_grads[name].reshape(-1).float().cpu())
        d["m"].append(m.reshape(-1).float().cpu())
        d["v"].append(v.reshape(-1).float().cpu())
        d["scale"].append((lr/(v.sqrt()+opt.defaults["eps"])).reshape(-1).float().cpu())
        d["u"].append((p.detach()-before[name]).reshape(-1).float().cpu())
        d["w"].append(before[name].reshape(-1).float().cpu())
    rows=[]
    for group,d in groups.items():
        if not d["g"]: continue
        z={k:torch.cat(v) for k,v in d.items() if v}
        g,u,w=z["g"],z["u"],z["w"]; scale=z["scale"]
        norm=lambda x:float(torch.linalg.vector_norm(x))
        cos=lambda a,b:float(F.cosine_similarity(a,b,dim=0)) if norm(a)>0 and norm(b)>0 else float("nan")
        row={"group":group,"grad_norm":norm(g),"grad_weight_ratio":norm(g)/(norm(w)+1e-12),
             "m_norm":norm(z["m"]),"v_mean":float(z["v"].mean()),
             "effective_lr_p10":float(torch.quantile(scale,.1)),"effective_lr_p50":float(torch.quantile(scale,.5)),
             "effective_lr_p90":float(torch.quantile(scale,.9)),
             "update_norm":norm(u),"update_weight_ratio":norm(u)/(norm(w)+1e-12),
             "grad_update_cos":cos(g,u),"predicted_delta":float(torch.dot(g,u)),
             "update_prev_cos":cos(u,prev_update[group]) if group in prev_update and prev_update[group].numel()==u.numel() else float("nan")}
        prev_update[group]=u
        rows.append(row)
    return rows

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def run_one(args,task,label,fold,optimizer_name,make_loaders,make_model,loss_fn,score_fn):
    out=Path(args.out)/task/label/f"fold_{fold}"/optimizer_name
    if args.skip_existing and (out/"summary.json").exists():
        print("SKIP",out,flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed+fold)
    train,val,test=make_loaders()
    model=make_model().to(args.device)
    optcls=torch.optim.Adam if optimizer_name=="adam" else torch.optim.AdamW
    opt=optcls(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    steps=[]; epochs=[]; prev={}
    best=-float("inf");best_state=None;best_epoch=None;step=0
    for epoch in range(args.epochs):
        lr=args.lr*.5*(1+math.cos(math.pi*epoch/args.epochs))
        for pg in opt.param_groups:pg["lr"]=lr
        model.train();total=0;n=0
        # Fixed epoch-specific shuffle for matched optimizer comparisons
        if hasattr(train.sampler,"generator") and train.sampler.generator is not None:
            train.sampler.generator.manual_seed(args.seed+fold*10000+epoch)
        for batch in train:
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model,batch,args.device)
            if not torch.isfinite(loss):raise RuntimeError(f"nonfinite loss {out} ep {epoch}")
            loss.backward()
            capture=(step%args.log_every==0)
            if capture:
                before={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad and p.grad is not None}
                grads={name:p.grad.detach().clone() for name,p in model.named_parameters() if name in before}
            opt.step()
            if capture:
                for row in diagnostics(model,opt,before,grads,lr,prev):
                    row.update(epoch=epoch+1,step=step,optimizer=optimizer_name,task=task,dataset=label,fold=fold)
                    steps.append(row)
            total+=float(loss.detach());n+=1;step+=1
        val_loss,val_score=score_fn(model,val,args.device)
        epochs.append({"epoch":epoch+1,"train_loss":total/max(n,1),"val_loss":val_loss,"val_score":val_score,"lr":lr})
        if np.isfinite(val_score) and val_score>best:
            best=val_score;best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"{label} fold={fold} {optimizer_name} epoch={epoch+1}/{args.epochs} val={val_score:.4f}",flush=True)
    if best_state is None:raise RuntimeError("No valid validation score")
    model.load_state_dict(best_state)
    test_loss,test_score=score_fn(model,test,args.device)
    write_csv(out/"steps.csv",steps);write_csv(out/"epochs.csv",epochs)
    (out/"summary.json").write_text(json.dumps({"task":task,"dataset":label,"fold":fold,"optimizer":optimizer_name,
        "epochs":args.epochs,"best_epoch":best_epoch,"best_val_score":best,
        "test_score":test_score,"test_loss":test_loss,"lr":args.lr,"weight_decay":args.weight_decay},indent=2))

def run_survival(args):
    import yaml
    from train_survival import build_survival_dataloaders,concordance_index
    from models.joint_model import build_model
    from experiments.cvpr_reliability_pilot.losses import cox_loss
    for cancer in args.cancers:
        cfg=yaml.safe_load(Path("configs",cancer+".yaml").read_text())
        for fold in range(args.folds):
            def loaders():
                data=build_survival_dataloaders(uni_dir=cfg["uni_dir"],gene_csv_paths=cfg["gene_csv_paths"],
                    image_dim=cfg["image_dim"],n_top_genes=cfg.get("n_top_genes",20000),
                    batch_size=cfg.get("batch_size",32),seed=args.seed,fold=fold,n_folds=args.folds)
                args.n_genes=data["n_genes"];args.image_dim=data["image_dim"]
                ds=data["train"].dataset
                generator=torch.Generator()
                tr=DataLoader(ds,batch_size=cfg.get("batch_size",32),shuffle=True,drop_last=True,
                              num_workers=0,generator=generator)
                return tr,data["val"],data["test"]
            def model():
                return build_model(args.n_genes,args.image_dim,"attention",0.3,verbose=False)
            def loss_fn(m,batch,device):
                inputs,labels=batch
                outputs=m({k:v.to(device) for k,v in inputs.items()})
                return cox_loss(outputs["survival"][:,1],labels["os_months"].float().to(device),
                                labels["vital_status"].float().to(device))
            @torch.no_grad()
            def score_fn(m,loader,device):
                m.eval();risks=[];times=[];events=[];losses=[]
                for inputs,labels in loader:
                    outputs=m({k:v.to(device) for k,v in inputs.items()})
                    risk=outputs["survival"][:,1]
                    t=labels["os_months"].float().to(device);e=labels["vital_status"].float().to(device)
                    losses.append(float(cox_loss(risk,t,e)))
                    risks.extend(risk.cpu().tolist());times.extend(t.cpu().tolist());events.extend(e.cpu().tolist())
                return float(np.mean(losses)),float(concordance_index(times,risks,events))
            for optimizer in ("adam","adamw"):
                run_one(args,"survival",cancer,fold,optimizer,loaders,model,loss_fn,score_fn)

def run_classification(args):
    import torchvision
    from torchvision import transforms
    from torch.utils.data import Subset
    norm=transforms.Normalize((.4914,.4822,.4465),(.247,.243,.261))
    train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),norm])
    test_tf=transforms.Compose([transforms.ToTensor(),norm])
    train_ds=torchvision.datasets.CIFAR10(root=args.data_dir,train=True,download=True,transform=train_tf)
    eval_ds=torchvision.datasets.CIFAR10(root=args.data_dir,train=True,download=True,transform=test_tf)
    test_ds=torchvision.datasets.CIFAR10(root=args.data_dir,train=False,download=True,transform=test_tf)
    idx=np.random.RandomState(2026).permutation(len(train_ds))
    train_idx=idx[:45000];val_idx=idx[45000:]
    for seed in args.seeds:
        def loaders():
            generator=torch.Generator()
            return (DataLoader(Subset(train_ds,train_idx),batch_size=args.batch_size,shuffle=True,num_workers=2,
                               pin_memory=True,generator=generator),
                    DataLoader(Subset(eval_ds,val_idx),batch_size=256,num_workers=2),
                    DataLoader(test_ds,batch_size=256,num_workers=2))
        def model():
            net=torchvision.models.resnet18(weights=None,num_classes=10)
            net.conv1=torch.nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
            net.maxpool=torch.nn.Identity()
            return net
        def loss_fn(m,batch,device):
            x,y=batch
            return torch.nn.functional.cross_entropy(m(x.to(device)),y.to(device))
        for optimizer in ("adam","adamw"):
            run_one(args,"classification","cifar10",seed,optimizer,loaders,model,loss_fn,evaluate_classification)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=["survival","classification"],required=True)
    p.add_argument("--cancers",nargs="+",default=["brca","kirc","luad","lusc","stad"])
    p.add_argument("--folds",type=int,default=3);p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--lr",type=float,default=.001);p.add_argument("--weight-decay",type=float,default=.0001)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--log-every",type=int,default=50)
    p.add_argument("--out",default="results/optimizer_dynamics_overnight")
    p.add_argument("--data-dir",default="data/cifar10")
    p.add_argument("--skip-existing",action="store_true")
    args=p.parse_args()
    if args.log_every<1:raise ValueError("--log-every must be >=1")
    args.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device",args.device,flush=True)
    if args.task=="survival":run_survival(args)
    else:run_classification(args)

if __name__=="__main__":main()
