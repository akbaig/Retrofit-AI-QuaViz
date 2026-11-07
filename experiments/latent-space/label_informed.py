"""Training-only event supervision for a reconstructable 32D encoder."""
import copy,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from models import Autoencoder,packed,masked_loss,neural_output
from evaluate import reconstruction_metrics
LABEL_KIND='label_informed'
LABEL_FILES=['events_train.csv','events_validation.csv']


def annotate_labels(meta,events,available):
    """Annotation overlay only; never an encoder feature or selection metric."""
    result=meta[['asset_id','available_utc']].reset_index(drop=True).copy()
    if not available:
        result['fault_label']='UNLABELED';result['label_eligible']=False
        return result
    times=pd.to_datetime(result.available_utc,utc=True)
    visible=[set() for _ in range(len(result))];ambiguous=np.zeros(len(result),bool)
    for event in events.itertuples():
        live=result.asset_id.eq(event.asset_id)&times.ge(pd.Timestamp(event.onset_utc))
        end=pd.to_datetime(event.end_utc,utc=True)
        if pd.notna(end):live &= times.lt(end)
        detectable=pd.to_datetime(event.detectable_utc,utc=True)
        if str(event.observable).lower()=='true' and pd.notna(detectable):
            hit=live&times.ge(detectable);ambiguous|=(live&~hit).to_numpy()
            for idx in np.flatnonzero(hit):visible[idx].add(str(event.fault_type))
        else:ambiguous|=live.to_numpy()
    result['fault_label']=['AMBIGUOUS_EVENT' if ambiguous[i] else '|'.join(sorted(v)) if v else 'NO_LABELED_EVENT' for i,v in enumerate(visible)]
    result['label_eligible']=~ambiguous
    return result


def training_supervision(meta,events,available):
    train=np.flatnonzero(meta.split.eq('train'))
    annotations=annotate_labels(meta.iloc[train],events,available)
    use=annotations.label_eligible.to_numpy()
    return train[use],annotations.fault_label.to_numpy()[use]


def contrastive_loss(z,codes,temperature=.2):
    unit=nn.functional.normalize(z,dim=1)
    logits=(unit@unit.T)/temperature;diagonal=torch.eye(len(z),dtype=torch.bool,device=z.device)
    logits=logits.masked_fill(diagonal,-1e9)
    positives=(codes[:,None]==codes[None,:])&~diagonal
    counts=positives.sum(dim=1);valid=counts>0
    if not valid.any():return z.sum()*0
    log_prob=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    return (-(log_prob*positives).sum(dim=1)[valid]/counts[valid]).mean()


def fit_label_informed(x,mask,train,validation,supervised_indices,states,latent=32,seed=42,max_epochs=50,contrastive_weight=.1):
    """No labels for validation/test are accepted by this interface."""
    train=np.asarray(train);validation=np.asarray(validation);idx=np.asarray(supervised_indices)
    states=np.asarray(states)
    if len(idx)!=len(states) or not np.isin(idx,train).all():raise ValueError('Supervision must index training rows only')
    groups,codes=np.unique(states,return_inverse=True)
    if len(groups)<2:raise ValueError('At least two known training states are required')
    if np.intersect1d(train,validation).size:raise ValueError('Validation cannot overlap training')
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    members=[idx[codes==j] for j in range(len(groups))]
    net=Autoencoder(x.shape[1],latent);optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    best=float('inf');best_state=None;bad_checks=0;history=[]
    for epoch in range(1,max_epochs+1):
        net.train();order=rng.permutation(train);losses=[];reconstruction_losses=[];contrastive_losses=[]
        for start in range(0,len(order),256):
            batch=order[start:start+256];target=torch.from_numpy(x[batch]);observed=torch.from_numpy(mask[batch])
            predicted,_=net(torch.from_numpy(packed(x[batch],mask[batch])))
            reconstruction=masked_loss(predicted,target,observed)
            chosen=rng.choice(len(groups),32,replace=True)
            labeled=np.concatenate([rng.choice(members[g],4,replace=True) for g in chosen])
            labels=torch.from_numpy(np.tile(np.repeat(chosen,4),2))
            values=torch.from_numpy(x[labeled]);seen=torch.from_numpy(mask[labeled]);views=[]
            for _ in range(2):
                keep=(torch.rand_like(values)>.1)&seen
                views.append(torch.cat([(values+.03*torch.randn_like(values))*keep,keep.float()*.25],dim=-1))
            z=net.encode(torch.cat(views));contrast=contrastive_loss(z,labels)
            loss=reconstruction+contrastive_weight*contrast
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()));reconstruction_losses.append(float(reconstruction.detach()));contrastive_losses.append(float(contrast.detach()))
        if epoch==1 or epoch%5==0 or epoch==max_epochs:
            _,recon=neural_output(net,x,mask,validation)
            score=reconstruction_metrics(x[validation],recon,mask[validation])['nmse']
            history.append({'epoch':epoch,'train_objective':float(np.mean(losses)),'reconstruction_loss':float(np.mean(reconstruction_losses)),
                            'contrastive_loss':float(np.mean(contrastive_losses)),'validation_nmse':score})
            if score<best:best=score;best_state=copy.deepcopy(net.state_dict());best_epoch=epoch;bad_checks=0
            else:bad_checks+=1
            if bad_checks>=5:break
    net.load_state_dict(best_state);net.eval();raw,_=neural_output(net,x,mask,train)
    scaler=StandardScaler().fit(raw);floor=max(float(np.max(scaler.scale_))*1e-6,1e-6);scaler.scale_=np.maximum(scaler.scale_,floor)
    return {'kind':LABEL_KIND,'type':'neural','temporal':False,'dimensions':latent,'seed':seed,'input_dim':x.shape[1],
            'state_dict':net.state_dict(),'latent_scaler':scaler,'best_epoch':best_epoch,'history':history,
            'effective_dimensions':int(np.sum(np.std(raw,axis=0)>floor)),'fit_seconds':time.monotonic()-started,'uses_labels':True,
            'supervised_rows':len(idx),'supervised_indices':idx,'training_states':dict(zip(groups.tolist(),np.bincount(codes).tolist())),
            'contrastive_weight':contrastive_weight,'contrastive_temperature':.2,'training_label_scope':'own pump, chronological training interval only'}
