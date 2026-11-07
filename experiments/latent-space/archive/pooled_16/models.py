"""Unsupervised, fixed-size representation models."""
import time
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


def masked_loss(prediction,target,mask):
    # No gradient encourages reconstructing an invented missing measurement.
    return ((prediction-target).square()*mask).sum()/mask.sum().clamp(min=1)


def packed(x,mask):
    return np.concatenate([x,mask.astype(np.float32)*.25],axis=-1).astype(np.float32)


class Autoencoder(nn.Module):
    def __init__(self,d,latent=16,temporal=False,length=12):
        super().__init__()
        self.temporal=temporal;self.length=length;self.d=d
        if temporal:
            self.input=nn.Sequential(nn.Linear(2*d,64),nn.GELU())
            self.gru=nn.GRU(64,64,batch_first=True)
            self.bottleneck=nn.Linear(64,latent)
        else:
            self.encoder=nn.Sequential(nn.Linear(2*d,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,latent))
        self.decoder=nn.Sequential(nn.Linear(latent,64),nn.GELU(),nn.Linear(64,128),nn.GELU(),nn.Linear(128,d*(length if temporal else 1)))

    def encode(self,x):
        if self.temporal:
            _,h=self.gru(self.input(x))
            return self.bottleneck(h[-1])
        return self.encoder(x)

    def forward(self,x):
        z=self.encode(x);recon=self.decoder(z)
        if self.temporal:recon=recon.reshape(-1,self.length,self.d)
        return recon,z


def fit_neural(x,mask,train_indices,sequence=None,epochs=30,seed=42):
    torch.set_num_threads(4);torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    net=Autoencoder(x.shape[1],temporal=sequence is not None)
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    xp=torch.from_numpy(x);mp=torch.from_numpy(mask);history=[]
    for epoch in range(epochs):
        net.train();order=rng.permutation(train_indices);losses=[]
        for start in range(0,len(order),256):
            idx=order[start:start+256]
            if sequence is not None:idx=sequence[idx]
            target=xp[idx];observed=mp[idx]
            keep=(torch.rand_like(target)>.15)&observed
            corrupted=(target+.03*torch.randn_like(target))*keep
            inputs=torch.cat([corrupted,keep.float()*.25],dim=-1)
            recon,z=net(inputs)
            loss=masked_loss(recon,target,observed)+1e-4*z.square().mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()))
        history.append({'epoch':epoch+1,'train_masked_mse':float(np.mean(losses))})
        if epoch%5==0 or epoch==epochs-1:print('epoch',epoch+1,'loss',history[-1]['train_masked_mse'],flush=True)
    return net.eval(),history


def encode_neural(net,x,mask,sequence=None):
    torch.set_num_threads(4);parts=[]
    with torch.no_grad():
        for start in range(0,len(x),512):
            idx=np.arange(start,min(start+512,len(x)))
            if sequence is not None:idx=sequence[idx]
            parts.append(net.encode(torch.from_numpy(packed(x[idx],mask[idx]))).numpy())
    return np.concatenate(parts)


def fit_contrastive(x,mask,train,y,seed=42,epochs=30):
    """Label-informed metric learning, using training events only.

    Two masked views share a positive; other positives have the exact same
    multilabel state. Inverse-frequency state sampling limits normal dominance.
    This is a supervised-contrastive experiment, not an implementation of TS2Vec.
    """
    torch.set_num_threads(4);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    net=Autoencoder(x.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-4)
    codes=np.sum(y.astype(np.int64)*(2**np.arange(y.shape[1])),axis=1)
    groups=np.unique(codes[train]);members=[train[codes[train]==g] for g in groups]
    history=[];xp=torch.from_numpy(x);mp=torch.from_numpy(mask)
    for epoch in range(epochs):
        losses=[]
        for _ in range(100):
            # Paired group draws ensure several same-state examples per batch.
            chosen=rng.choice(len(groups),32,replace=True)
            idx=np.concatenate([rng.choice(members[g],4,replace=True) for g in chosen])
            target=xp[idx];observed=mp[idx]
            inputs=[]
            for _view in range(2):
                keep=(torch.rand_like(target)>.1)&observed
                inputs.append(torch.cat([(target+.03*torch.randn_like(target))*keep,keep.float()*.25],dim=-1))
            recon,z=net(torch.cat(inputs));unit=nn.functional.normalize(z,dim=1)
            logits=unit@unit.T/.2
            diagonal=torch.eye(len(z),dtype=torch.bool)
            logits=logits.masked_fill(diagonal,-1e9)
            c=torch.from_numpy(np.tile(codes[idx],2))
            positives=(c[:,None]==c[None,:])&~diagonal
            log_probability=logits-torch.logsumexp(logits,dim=1,keepdim=True)
            contrast=-(log_probability*positives).sum(dim=1)/positives.sum(dim=1).clamp(min=1)
            loss=contrast.mean()+.05*masked_loss(recon,target.repeat(2,1),observed.repeat(2,1))
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            losses.append(float(loss.detach()))
        history.append({'epoch':epoch+1,'contrastive_plus_reconstruction_loss':float(np.mean(losses))})
        if epoch%5==0 or epoch==epochs-1:print('contrastive epoch',epoch+1,'loss',history[-1]['contrastive_plus_reconstruction_loss'],flush=True)
    return net.eval(),history


def fit_model(kind,x,mask,train,sequence=None,seed=42):
    start=time.monotonic();history=[]
    sample=np.asarray(train)[::5]
    if kind in ['raw_pca','physical_pca']:
        model=PCA(n_components=16,random_state=seed).fit(packed(x[train],mask[train]))
        z=model.transform(packed(x,mask));bundle={'model':model,'type':'pca'}
    elif kind=='rbf_pca':
        v=packed(x[sample],mask[sample]);dist=pairwise_distances(v[::max(1,len(v)//1000)],squared=True)
        gamma=1/max(float(np.median(dist[dist>0])),1e-6)
        kernel=Nystroem(kernel='rbf',gamma=gamma,n_components=256,random_state=seed).fit(v)
        pca=PCA(n_components=16,random_state=seed).fit(kernel.transform(v))
        z=pca.transform(kernel.transform(packed(x,mask)))
        bundle={'model':pca,'kernel':kernel,'gamma':gamma,'type':'rbf'}
    else:
        seq=sequence if kind=='temporal_ae' else None
        model,history=fit_neural(x,mask,sample,sequence=seq,seed=seed)
        z=encode_neural(model,x,mask,seq)
        bundle={'type':'neural','temporal':seq is not None,'state_dict':model.state_dict(),'input_dim':x.shape[1]}
    scaler=StandardScaler().fit(z[train])
    bundle.update({'latent_scaler':scaler,'kind':kind,'seed':seed,'seconds':time.monotonic()-start,'history':history})
    return scaler.transform(z).astype(np.float32),bundle


def transform_model(bundle,x,mask,sequence=None):
    if bundle['type']=='pca':z=bundle['model'].transform(packed(x,mask))
    elif bundle['type']=='rbf':z=bundle['model'].transform(bundle['kernel'].transform(packed(x,mask)))
    else:
        model=Autoencoder(bundle['input_dim'],temporal=bundle['temporal'])
        model.load_state_dict(bundle['state_dict']);model.eval()
        z=encode_neural(model,x,mask,sequence if bundle['temporal'] else None)
    return bundle['latent_scaler'].transform(z).astype(np.float32)
