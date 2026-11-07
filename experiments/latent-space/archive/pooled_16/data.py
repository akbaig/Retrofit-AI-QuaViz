"""Data preparation and evaluation metadata, independent of fitted encoders."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / 'BusinessTurku_SinceAI2026_FINAL_BUILDER_PUBLISH/RETROFIT_AI_PARTICIPANTS_2026/data/full'
sys.path.insert(0, str(ROOT / 'vendor'))
from features import build_features, CHANNELS


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def fingerprint(paths,settings=None):
    payload={'files':{str(p):sha256(p) for p in paths},'settings':settings or {}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def preparation_fingerprint():
    files=[ROOT/'data.py',*sorted((ROOT/'vendor').glob('*.py'))]
    files += [DATA/name for name in ['assets.csv','features_1min.parquet','operating_context.csv','vibration_snapshots.parquet','events_train.csv','events_validation.csv','benign_windows_train_validation.csv']]
    return fingerprint(files,{'completed_minutes':True,'protocol':2})


def check_fingerprint(actual,expected):
    if actual!=expected:
        raise RuntimeError('Cached artifacts do not match current sources/settings. Preserve results under a new directory name and rerun from an empty results directory.')


class Preprocessor:
    """Training-only robust scaling; unknown measurements retain a separate mask."""
    def fit(self, frame):
        self.columns=[c for c in frame if frame[c].notna().sum()>=2 and frame[c].std()>1e-8]
        a=frame[self.columns].to_numpy(dtype=np.float32)
        self.center=np.nanmedian(a,axis=0)
        iqr=np.nanquantile(a,.75,axis=0)-np.nanquantile(a,.25,axis=0)
        self.scale=np.where(iqr>1e-5,iqr,np.nanstd(a,axis=0)).clip(1e-5).astype(np.float32)
        return self

    def transform(self,frame):
        a=frame[self.columns].to_numpy(dtype=np.float32)
        mask=np.isfinite(a)
        x=np.nan_to_num(np.clip((a-self.center)/self.scale,-8,8),nan=0,posinf=8,neginf=-8)
        return x.astype(np.float32),mask


def sequence_indices(asset_ids,length=12,step=5):
    asset_ids=np.asarray(asset_ids)
    starts=np.maximum.accumulate(np.where(np.r_[True,asset_ids[1:]!=asset_ids[:-1]],np.arange(len(asset_ids)),0))
    return np.maximum(np.arange(len(asset_ids))[:,None]-np.arange(length-1,-1,-1)[None,:]*step,starts[:,None])


def annotate(meta,events,benign=None):
    out=meta.copy()
    names=sorted(events.fault_type.unique())
    y=np.zeros((len(out),len(names)),dtype=np.uint8)
    active=np.zeros(len(out),int);unobservable=np.zeros(len(out),bool);predetectable=np.zeros(len(out),bool)
    event_ids=[[] for _ in range(len(out))]
    active_ids=[[] for _ in range(len(out))]
    t=pd.to_datetime(out.available_utc,utc=True)
    for _,e in events.iterrows():
        same=out.asset_id.eq(e.asset_id)&out.split.ne('test')
        onset=pd.Timestamp(e.onset_utc)
        end=pd.to_datetime(e.end_utc,utc=True)
        live=same&t.ge(onset)
        if pd.notna(end): live&=t.lt(end)
        active+=live.to_numpy()
        for i in np.flatnonzero(live):active_ids[i].append(str(e.event_id))
        observable=str(e.observable).lower()=='true'
        detectable=pd.to_datetime(e.detectable_utc,utc=True)
        if observable and pd.notna(detectable):
            visible=live&t.ge(detectable)
            predetectable|=(live&~visible).to_numpy()
            y[visible,names.index(e.fault_type)]=1
            for i in np.flatnonzero(visible):event_ids[i].append(str(e.event_id))
        else: unobservable|=live.to_numpy()
    labels=[]
    for i in range(len(out)):
        if out.iloc[i]['split']=='test':label='UNLABELED_TEST'
        elif y[i].sum()>1:label='MULTIPLE'
        elif y[i].sum()==1:label=names[np.flatnonzero(y[i])[0]]
        elif unobservable[i]:label='UNOBSERVABLE_EVENT'
        elif active[i]>0:label='PRE_DETECTABLE'
        else:label='NO_LABELED_EVENT'
        labels.append(label)
    out['label']=labels
    out['fault_types']=['|'.join(np.asarray(names)[row.astype(bool)]) for row in y]
    out['event_ids']=['|'.join(x) for x in event_ids]
    out['active_event_ids']=['|'.join(x) for x in active_ids]
    out['has_predetectable_event']=predetectable
    out['has_unobservable_event']=unobservable
    out['active_event_count']=active
    out['detectable_event_count']=y.sum(axis=1)
    out['benign_type']=''
    if benign is not None:
        for _,e in benign.iterrows():
            hit=out.asset_id.eq(e.asset_id)&t.ge(pd.Timestamp(e.start_utc))&t.lt(pd.Timestamp(e.end_utc))
            out.loc[hit,'benign_type']=out.loc[hit,'benign_type'].apply(lambda s:'|'.join(filter(None,[s,str(e.benign_type)])))
    out['broad_label']=[
        'MACHINE+SENSOR' if any(n.startswith(('SENSOR_','COMMUNICATION')) for n in f.split('|')) and any(n and not n.startswith(('SENSOR_','COMMUNICATION')) for n in f.split('|'))
        else 'SENSOR' if f.startswith(('SENSOR_','COMMUNICATION')) else 'MACHINE' if f else label
        for f,label in zip(out.fault_types,out.label)]
    return out,y,names


def evaluation_eligible(meta):
    # Display color is not an eligibility mask: visible and unknown states overlap.
    return meta.split.ne('test')&~meta.has_predetectable_event&~meta.has_unobservable_event


def prepare(splits=('train','validation')):
    a=pd.read_csv(DATA/'assets.csv')
    f=pd.read_parquet(DATA/'features_1min.parquet')
    ctx=pd.read_csv(DATA/'operating_context.csv')
    shots=pd.read_parquet(DATA/'vibration_snapshots.parquet')
    events=pd.concat([pd.read_csv(DATA/f'events_{s}.csv') for s in ['train','validation']],ignore_index=True)
    benign=pd.read_csv(DATA/'benign_windows_train_validation.csv')
    frames=[];metas=[]
    for _,asset in a[a.split.isin(splits)].iterrows():
        print('Preparing',asset.asset_id,flush=True)
        meta,x=build_features(f,asset,ctx,shots)
        for col in ['split','instrumentation_tier','profile','control']:meta[col]=asset[col]
        modes=f[f.asset_id==asset.asset_id].set_index('window_start_utc').operating_mode
        meta['operating_mode']=meta.window_start_utc.map(modes).fillna('NO_TELEMETRY')
        meta['minute_index']=np.arange(len(meta))
        frames.append(x);metas.append(meta)
    meta=pd.concat(metas,ignore_index=True);x=pd.concat(frames,ignore_index=True)
    meta,y,names=annotate(meta,events,benign)
    return meta,x,y,names


def feature_sets(frame):
    raw=[c for c in frame if c.startswith(('value__','std__','range__')) or c in ['obs__row_fraction','obs__running']]
    physical=[c for c in frame if c.startswith(('phys__','snapshot__','level__','std__','range__','delta10__','delta60__','mean10__','mean60__','span10__','span60__','max__','obs__'))]
    return {'raw':raw,'physical':physical}


def audit(out):
    import pyarrow.parquet as pq
    a=pd.read_csv(DATA/'assets.csv')
    f=pd.read_parquet(DATA/'features_1min.parquet')
    e=pd.concat([pd.read_csv(DATA/f'events_{s}.csv').assign(split=s) for s in ['train','validation']],ignore_index=True)
    manifest=json.loads((DATA/'manifest.json').read_text())
    result={'synthetic':True,'raw_rows':sum(manifest['telemetry_rows'].values()),'minute_rows_supplied':len(f),
            'complete_minute_grid':12*7*24*60,'absent_complete_minutes':12*7*24*60-len(f),
            'snapshots':pq.ParquetFile(DATA/'vibration_snapshots.parquet').metadata.num_rows,
            'events_by_split':e.groupby('split').size().to_dict(),
            'observable_events_by_split':e[e.observable.astype(str).str.lower().eq('true')].groupby('split').size().to_dict(),
            'duplicate_minutes':int(f.duplicated(['asset_id','window_start_utc']).sum()),
            'source_hashes':{str(p.relative_to(DATA)):sha256(p) for p in DATA.iterdir() if p.is_file()},
            'vendor_hashes':{p.name:sha256(p) for p in (ROOT/'vendor').glob('*.py')}}
    counts=f.groupby('asset_id').size().rename('minute_rows').reset_index()
    a.merge(counts).to_csv(out/'assets_audit.csv',index=False)
    e.to_csv(out/'events_audit.csv',index=False)
    f.groupby('asset_id')[[c+'__count' for c in CHANNELS]].sum().to_csv(out/'observed_sample_counts.csv')
    (out/'data_audit.json').write_text(json.dumps(result,indent=2))
    return result
