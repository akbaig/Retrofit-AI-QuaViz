"""Label-free observations and chronological partitions for per-pump encoders."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parents[1]/'BusinessTurku_SinceAI2026_FINAL_BUILDER_PUBLISH/RETROFIT_AI_PARTICIPANTS_2026/data/full'
sys.path.insert(0,str(ROOT/'vendor'))
from features import build_features
INPUT_FILES=['assets.csv','features_1min.parquet','operating_context.csv','vibration_snapshots.parquet']
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def fingerprint(paths,settings=None):
    return hashlib.sha256(json.dumps({'files':{str(p):sha256(p) for p in paths},'settings':settings or {}},sort_keys=True).encode()).hexdigest()
def source_fingerprint():
    return fingerprint([ROOT/'data.py',*sorted((ROOT/'vendor').glob('*.py')),*[DATA/f for f in INPUT_FILES]])
def chronological_split(n,warmup=120):
    a=int(n*.6);b=int(n*.8)
    split=np.full(n,'test',dtype=object);split[:a]='train';split[a:b]='validation'
    eligible=np.ones(n,bool);eligible[a:min(a+warmup,b)]=False;eligible[b:min(b+warmup,n)]=False
    return split,eligible
class ObservedScaler:
    def fit(self,frame):
        self.columns=[];self.excluded={};centers=[];scales=[]
        for name in frame:
            v=frame[name].to_numpy(dtype=np.float64);v=v[np.isfinite(v)]
            if len(v)<2:self.excluded[name]='unobserved_or_insufficient_training_samples';continue
            std=v.std()
            if std<1e-6:self.excluded[name]='constant_in_training';continue
            self.columns.append(name);centers.append(v.mean());scales.append(std)
        if not self.columns:raise ValueError('No variable observed training features')
        self.center=np.asarray(centers,dtype=np.float32);self.scale=np.asarray(scales,dtype=np.float32)
        return self
    def transform(self,frame):
        a=frame[self.columns].to_numpy(dtype=np.float32);mask=np.isfinite(a)
        x=np.where(mask,(a-self.center)/self.scale,0).astype(np.float32)
        return x,mask
def candidate_features(frame):
    prefixes=('phys__','snapshot__','level__','std__','range__','delta10__','delta60__','mean10__','mean60__','span10__','span60__','max__','obs__')
    return [c for c in frame if c.startswith(prefixes)]
def prepare(out):
    signature=source_fingerprint();marker=out/'preparation.json'
    if (out/'features.parquet').exists():
        if not marker.exists() or json.loads(marker.read_text())['fingerprint']!=signature:
            raise RuntimeError('Preparation sources changed. Archive results and run again in a fresh results directory.')
        return pd.read_parquet(out/'metadata.parquet'),pd.read_parquet(out/'features.parquet')
    assets=pd.read_csv(DATA/'assets.csv');minutes=pd.read_parquet(DATA/'features_1min.parquet')
    context=pd.read_csv(DATA/'operating_context.csv');shots=pd.read_parquet(DATA/'vibration_snapshots.parquet')
    metadata=[];frames=[]
    for _,asset in assets.iterrows():
        print('Preparing',asset.asset_id,flush=True)
        meta,frame=build_features(minutes,asset,context,shots)
        meta['minute_index']=np.arange(len(meta));meta['split'],meta['score_eligible']=chronological_split(len(meta))
        meta['original_benchmark_split']=asset.split
        for col in ['instrumentation_tier','profile','control']:meta[col]=asset[col]
        modes=minutes[minutes.asset_id.eq(asset.asset_id)].set_index('window_start_utc').operating_mode
        meta['operating_mode']=meta.window_start_utc.map(modes).fillna('NO_TELEMETRY')
        metadata.append(meta);frames.append(frame[candidate_features(frame)])
    meta=pd.concat(metadata,ignore_index=True);frame=pd.concat(frames,ignore_index=True)
    meta.to_parquet(out/'metadata.parquet',index=False);frame.to_parquet(out/'features.parquet',index=False)
    payload={'fingerprint':signature,'source_hashes':{name:sha256(DATA/name) for name in INPUT_FILES},
             'vendor_hashes':{p.name:sha256(p) for p in (ROOT/'vendor').glob('*.py')},
             'event_labels_read':False,'dimensions':32,'split_fractions':[.6,.2,.2],'warmup_minutes':120}
    marker.write_text(json.dumps(payload,indent=2));assets.to_csv(out/'assets.csv',index=False)
    return meta,frame
