"""Five unsupervised reconstruction encoders; no event-label interface."""
import copy,time
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from evaluate import reconstruction_metrics
KINDS=['pca','kernel_pca','autoencoder','denoising_ae','temporal_ae']
def packed(x,mask):return np.concatenate([x,mask.astype(np.float32)*.25],axis=-1).astype(np.float32)
def sequence_indices(ids,length=12,step=5):
    ids=np.asarray(ids);i=np.arange(len(ids));start=np.maximum.accumulate(np.where(np.r_[True,ids[1:]!=ids[:-1]],i,0))
    return np.maximum(i[:,None]-np.arange(length-1,-1,-1)[None,:]*step,start[:,None])
def masked_loss(pred,target,mask):
    counts=mask.sum(dim=0);valid=counts>0
    return (((pred-target).square()*mask).sum(dim=0)[valid]/counts[valid]).mean()
class Autoencoder(nn.Module):
    def __init__(self,d,latent=32,temporal=False):
        super().__init__();self.temporal=temporal
        if temporal:
            self.input=nn.Sequential(nn.Linear(2*d,96),nn.GELU())
            self.gru=nn.GRU(96,96,batch_first=True);self.bottleneck=nn.Linear(96,latent)
        else:
            self.encoder=nn.Sequential(nn.Linear(2*d,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,latent))
        self.decoder=nn.Sequential(nn.Linear(latent,64),nn.GELU(),nn.Linear(64,128),nn.GELU(),nn.Linear(128,d))
    def encode(self,x):
        if self.temporal:
            _,h=self.gru(self.input(x));return self.bottleneck(h[-1])
        return self.encoder(x)
    def forward(self,x):
        z=self.encode(x);return self.decoder(z),z

def neural_output(net,x,mask,indices=None,sequence=None):
    indices=np.arange(len(x)) if indices is None else np.asarray(indices)
    zs=[];recons=[];net.eval()
    with torch.no_grad():
        for begin in range(0,len(indices),512):
            chosen=indices[begin:begin+512];inputs=chosen if sequence is None else sequence[chosen]
            recon,z=net(torch.from_numpy(packed(x[inputs],mask[inputs])))
            zs.append(z.numpy());recons.append(recon.numpy())
    return np.concatenate(zs),np.concatenate(recons)

def fit_encoder(kind,x,mask,train,validation,latent=32,sequence=None,seed=42,max_epochs=50):
    if kind not in KINDS:raise ValueError(kind)
    start=time.monotonic();train=np.asarray(train);validation=np.asarray(validation)
    torch.set_num_threads(4);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    bundle={'kind':kind,'dimensions':latent,'seed':seed,'input_dim':x.shape[1],'history':[]}
    v=packed(x[train],mask[train])
    if kind=='pca':
        pca=PCA(n_components=latent,random_state=seed).fit(v)
        bundle.update(pca=pca,type='pca');ztrain=pca.transform(v)
    elif kind=='kernel_pca':
        subset=v[::max(1,len(v)//800)];dist=pairwise_distances(subset,squared=True);positive=dist[dist>0]
        gamma=1/max(float(np.median(positive)),1e-6)
        kernel=Nystroem(kernel='rbf',gamma=gamma,n_components=min(256,len(train)),random_state=seed).fit(v)
        kp=kernel.transform(v);pca=PCA(n_components=latent,random_state=seed).fit(kp);ztrain=pca.transform(kp)
        decoders=[]
        for j in range(x.shape[1]):
            valid=mask[train,j];decoders.append(Ridge(alpha=1.).fit(ztrain[valid],x[train[valid],j]))
        bundle.update(type='kernel',kernel=kernel,pca=pca,decoders=decoders,gamma=gamma)
    else:
        temporal=kind=='temporal_ae';seq=sequence if temporal else None
        if temporal and seq is None:raise ValueError('Temporal encoder requires causal sequence indices')
        net=Autoencoder(x.shape[1],latent,temporal);optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
        best=float('inf');best_state=None;bad_checks=0
        for epoch in range(1,max_epochs+1):
            net.train();order=rng.permutation(train);losses=[]
            for begin in range(0,len(order),256):
                idx=order[begin:begin+256];source=idx if seq is None else seq[idx]
                inputs=torch.from_numpy(x[source]);observed=torch.from_numpy(mask[source]);target=torch.from_numpy(x[idx]);target_mask=torch.from_numpy(mask[idx])
                if kind in ['denoising_ae','temporal_ae']:
                    keep=(torch.rand_like(inputs)>.15)&observed;inputs=(inputs+.03*torch.randn_like(inputs))*keep
                else:keep=observed
                reconstruction,z=net(torch.cat([inputs,keep.float()*.25],dim=-1));loss=masked_loss(reconstruction,target,target_mask)
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
            if epoch==1 or epoch%5==0 or epoch==max_epochs:
                _,pred=neural_output(net,x,mask,validation,seq);metric=reconstruction_metrics(x[validation],pred,mask[validation])['nmse']
                bundle['history'].append({'epoch':epoch,'train_objective':float(np.mean(losses)),'validation_nmse':metric})
                if metric<best:
                    best=metric;best_state=copy.deepcopy(net.state_dict());bundle['best_epoch']=epoch;bad_checks=0
                else:bad_checks+=1
                if bad_checks>=5:break
        net.load_state_dict(best_state);net.eval();ztrain,_=neural_output(net,x,mask,train,seq)
        bundle.update(type='neural',state_dict=net.state_dict(),temporal=temporal)
    scaler=StandardScaler().fit(ztrain)
    floor=max(float(np.max(scaler.scale_))*1e-6,1e-6);scaler.scale_=np.maximum(scaler.scale_,floor)
    bundle.update(latent_scaler=scaler,fit_seconds=time.monotonic()-start,effective_dimensions=int(np.sum(np.std(ztrain,axis=0)>floor)))
    return bundle

def encode_reconstruct(bundle,x,mask,sequence=None):
    if bundle['type']=='pca':
        raw=bundle['pca'].transform(packed(x,mask));recon=bundle['pca'].inverse_transform(raw)[:,:bundle['input_dim']]
    elif bundle['type']=='kernel':
        raw=bundle['pca'].transform(bundle['kernel'].transform(packed(x,mask)));recon=np.column_stack([decoder.predict(raw) for decoder in bundle['decoders']])
    else:
        net=Autoencoder(bundle['input_dim'],bundle['dimensions'],bundle['temporal']);net.load_state_dict(bundle['state_dict']);net.eval()
        raw,recon=neural_output(net,x,mask,sequence=sequence if bundle['temporal'] else None)
    return bundle['latent_scaler'].transform(raw).astype(np.float32),np.asarray(recon,dtype=np.float32)
