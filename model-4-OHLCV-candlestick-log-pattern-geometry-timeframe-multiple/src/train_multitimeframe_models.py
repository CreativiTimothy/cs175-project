import argparse, copy, json, random
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch, yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from src.data.multitimeframe import prepare_multitimeframe_data
from src.metrics import classification_metrics
from src.models.multitimeframe import MultiTimeframeModel

DIRECT = ["open_prev_close_log","high_prev_close_log","low_prev_close_log","close_return_1","body_log",
          "open_to_high_log","open_to_low_log","range_log","upper_wick_fraction","lower_wick_fraction",
          "body_fraction","close_location","open_location","bullish_bar","volume_change_1"]

class DS(Dataset):
    def __init__(self,t,c,y): self.t=torch.tensor(t,dtype=torch.float32); self.c=torch.tensor(c,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.t[i],self.c[i],self.y[i]

def predict(model,loader,device):
    model.eval(); values=[]
    with torch.no_grad():
        for t,c,_ in loader: values.append(model(t.to(device),c.to(device)).cpu().numpy())
    return np.concatenate(values)

def save(meta,y,scores,calibrator,path):
    probability=calibrator.predict_proba(scores.reshape(-1,1))[:,1]
    frame=meta.copy(); frame["probability_up"]=probability; frame["predicted_direction"]=(probability>=.5).astype(int); frame.to_csv(path,index=False)
    return classification_metrics(y,probability)

def logistic(config,data,output):
    Xtr=data.context_train.reshape(len(data.context_train),-1); Xv=data.context_validation.reshape(len(data.context_validation),-1)
    Xte=data.context_test.reshape(len(data.context_test),-1); scaler=StandardScaler().fit(Xtr)
    Xtr,Xv,Xte=scaler.transform(Xtr),scaler.transform(Xv),scaler.transform(Xte)
    rows=[]
    for C in config["model"].get("c_values",[.0001,.001,.01,.1,1.0]):
        m=LogisticRegression(C=C,solver="liblinear",max_iter=4000).fit(Xtr,data.y_train.astype(int)); p=m.predict_proba(Xv)[:,1]
        rows.append({"C":C,"roc_auc":roc_auc_score(data.y_validation,p),"log_loss":log_loss(data.y_validation,p)})
    table=pd.DataFrame(rows).sort_values(["roc_auc","log_loss"],ascending=[False,True]); table.to_csv(output/"c_tuning_results.csv",index=False)
    C=float(table.iloc[0].C); model=LogisticRegression(C=C,solver="liblinear",max_iter=4000).fit(np.r_[Xtr,Xv],np.r_[data.y_train,data.y_validation].astype(int))
    probability=model.predict_proba(Xte)[:,1]; frame=data.meta_test.copy(); frame["probability_up"]=probability; frame.to_csv(output/"test_predictions.csv",index=False)
    metrics=classification_metrics(data.y_test,probability); metrics["best_C"]=C; json.dump(metrics,open(output/"metrics.json","w"),indent=2); joblib.dump({"scaler":scaler,"model":model},output/"model.joblib"); print(json.dumps(metrics,indent=2))

def neural(config,data,output):
    seed=int(config["project"].get("seed",42)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); indices=[data.feature_names.index(n) for n in DIRECT]
    mc=config["model"]; model=MultiTimeframeModel(mc["type"],len(data.feature_names),len(data.timeframe_names),indices,mc.get("projection_size",16),mc.get("hidden_size",32),mc.get("context_hidden",128),mc.get("fusion_hidden",32),mc.get("dropout",.3)).to(device)
    print(f"Model={mc['type']}; parameters={sum(p.numel() for p in model.parameters()):,}; device={device}")
    bs=config["training"].get("batch_size",128); tr=DataLoader(DS(data.target_train,data.context_train,data.y_train),bs,shuffle=True); va=DataLoader(DS(data.target_validation,data.context_validation,data.y_validation),bs); te=DataLoader(DS(data.target_test,data.context_test,data.y_test),bs)
    lossfn=nn.BCEWithLogitsLoss(); opt=torch.optim.AdamW(model.parameters(),lr=config["training"].get("learning_rate",1e-4),weight_decay=config["training"].get("weight_decay",1e-3)); best=None; bestloss=1e9; stale=0
    for epoch in range(1,config["training"].get("epochs",100)+1):
        model.train(); tl=[]
        for t,c,y in tr: opt.zero_grad(); loss=lossfn(model(t.to(device),c.to(device)),y.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); tl.append(loss.item())
        model.eval(); vl=[]
        with torch.no_grad():
            for t,c,y in va: vl.append(lossfn(model(t.to(device),c.to(device)),y.to(device)).item())
        v=float(np.mean(vl)); print(f"Epoch {epoch:03d} train={np.mean(tl):.6f} validation={v:.6f}")
        if v<bestloss-1e-6: bestloss=v; best=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=config["training"].get("patience",15): print("Early stopping"); break
    model.load_state_dict(best); vr=predict(model,va,device); cal=LogisticRegression(C=1e6,max_iter=1000).fit(vr.reshape(-1,1),data.y_validation.astype(int)); scores=predict(model,te,device); metrics=save(data.meta_test,data.y_test,scores,cal,output/"test_predictions.csv"); json.dump(metrics,open(output/"metrics.json","w"),indent=2); torch.save(model.state_dict(),output/"best_model.pt"); print(json.dumps(metrics,indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args(); config=yaml.safe_load(Path(args.config).read_text()); output=Path(config["project"]["output_dir"]); output.mkdir(parents=True,exist_ok=True); data=prepare_multitimeframe_data(config,output)
    logistic(config,data,output) if config["model"]["type"]=="logistic" else neural(config,data,output)
if __name__=="__main__": main()
