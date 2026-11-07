"""A 32D representation taught to retain fault information, with a light decoder."""
import copy
import time
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from models import Autoencoder, packed, masked_loss, neural_output


def fit_task_encoder(x,mask,train,validation,train_y,validation_y,seed=42,max_epochs=30):
    train=np.asarray(train); validation=np.asarray(validation)
    if np.intersect1d(train,validation).size: raise ValueError('Training and validation overlap')
    if len(train)!=len(train_y) or len(validation)!=len(validation_y): raise ValueError('Targets must align with the supplied rows')
    if not np.isin(train_y,[0,1]).all() or not np.isin(validation_y,[0,1]).all(): raise ValueError('Only eligible binary targets accepted')
    started=time.monotonic(); torch.set_num_threads(4); torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    # Bound neural inputs only for this candidate; train scaler and missingness remain unchanged.
    x=np.clip(x,-10,10).astype(np.float32)
    net=Autoencoder(x.shape[1],32); head=nn.Linear(32,train_y.shape[1])
    optimizer=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-3,weight_decay=1e-4)
    positive=train_y.sum(axis=0); weight=np.clip(np.sqrt((len(train_y)-positive)/np.maximum(positive,1)),1,20)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,dtype=torch.float32))
    best=-np.inf; best_state=None; history=[]; stale=0
    for epoch in range(1,max_epochs+1):
        net.train(); head.train(); order=rng.permutation(len(train)); losses=[]
        for begin in range(0,len(order),256):
            chosen=order[begin:begin+256]; idx=train[chosen]
            values=torch.from_numpy(x[idx]); seen=torch.from_numpy(mask[idx])
            keep=(torch.rand_like(values)>.05)&seen
            corrupted=(values+.02*torch.randn_like(values))*keep
            recon,z=net(torch.cat([corrupted,keep.float()*.25],dim=-1))
            classify=criterion(head(z),torch.tensor(train_y[chosen],dtype=torch.float32))
            loss=classify+.05*masked_loss(recon,values,seen)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),5)
            optimizer.step(); losses.append(float(loss.detach()))
        if epoch==1 or epoch%5==0 or epoch==max_epochs:
            z,_=neural_output(net,x,mask,validation)
            with torch.no_grad(): scores=torch.sigmoid(head(torch.from_numpy(z))).numpy()
            aps=[average_precision_score(validation_y[:,j],scores[:,j]) for j in range(train_y.shape[1]) if len(np.unique(validation_y[:,j]))==2]
            # ANY_FAULT is the primary task; other heads share the encoder during fitting.
            score=float(average_precision_score(validation_y[:,0],scores[:,0])) if len(np.unique(validation_y[:,0]))==2 else float(np.mean(aps)) if aps else -1.
            history.append(dict(epoch=epoch,train_objective=float(np.mean(losses)),development_any_ap=score))
            if score>best:
                best=score;best_state=copy.deepcopy(net.state_dict());best_head=copy.deepcopy(head.state_dict());best_epoch=epoch;stale=0
            else: stale+=1
            if stale>=4:break
    net.load_state_dict(best_state);raw,_=neural_output(net,x,mask,train)
    scaler=StandardScaler().fit(raw);scaler.scale_=np.maximum(scaler.scale_,1e-6)
    return dict(kind='task_encoder',type='task_encoder',dimensions=32,input_dim=x.shape[1],state_dict=net.state_dict(),
                auxiliary_head=best_head,latent_scaler=scaler,history=history,best_epoch=best_epoch,
                fit_seconds=time.monotonic()-started,uses_labels=True,input_clip=10.)


def encode_task(bundle,x,mask):
    net=Autoencoder(bundle['input_dim'],32);net.load_state_dict(bundle['state_dict'])
    raw,_=neural_output(net,np.clip(x,-10,10).astype(np.float32),mask)
    return bundle['latent_scaler'].transform(raw).astype(np.float32)
