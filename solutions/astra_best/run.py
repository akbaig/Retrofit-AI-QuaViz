"""Reproducible supervised experiment: prepare, train, select and evaluate."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd

from paths import HERE,SOURCE
sys.path.insert(0,str(SOURCE/'src'))
from retrofit_sim.scoring import score
from features import build_features
from learning import LearnedModel,make_targets
from alerts import AlertPolicy,make_alerts


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_data(name):
    original=name.startswith('original_')
    path=SOURCE/'data/full' if original else HERE/'synthetic_data'/name
    assets=pd.read_csv(path/'assets.csv')
    if original:
        split=name.removeprefix('original_')
        assert split in ('train','validation')
        assets=assets[assets.split.eq(split)].copy()
    ids=assets.asset_id.tolist()
    features=pd.read_parquet(path/'features_1min.parquet',filters=[('asset_id','in',ids)])
    context=pd.read_csv(path/'operating_context.csv');context=context[context.asset_id.isin(ids)]
    sp=path/'vibration_snapshots.parquet'
    snapshots=pd.read_parquet(sp,filters=[('asset_id','in',ids)]) if sp.exists() else None
    events=pd.read_csv(path/(f'events_{split}.csv' if original else 'events.csv'))
    benign=pd.read_csv(path/'benign_windows_train_validation.csv') if original else None
    if benign is not None:benign=benign[benign.asset_id.isin(ids)]
    return assets,features,context,snapshots,events,benign


def prepared(name):
    fingerprint=sha(HERE/'features.py')[:10]+sha(HERE/'physics_features.py')[:10]
    cache=HERE/'cache'/f'{name}_{fingerprint}.joblib'
    if cache.exists():return joblib.load(cache)
    assets,features,context,snapshots,events,benign=load_data(name)
    metas=[];xs=[]
    for _,asset in assets.iterrows():
        m,x=build_features(features,asset,context,snapshots)
        metas.append(m);xs.append(x)
    result=(pd.concat(metas,ignore_index=True),pd.concat(xs,ignore_index=True),assets,events,benign)
    cache.parent.mkdir(exist_ok=True)
    joblib.dump(result,cache,compress=3)
    print('prepared',name,len(result[0]),'minutes',len(result[1].columns),'features',flush=True)
    return result


def train(args):
    out=HERE/'runs'/args.name
    if (out/'model.joblib').exists():raise RuntimeError('Run already exists; use a new run name')
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    pieces=[prepared(n) for n in args.datasets]
    meta=pd.concat([p[0] for p in pieces],ignore_index=True)
    x=pd.concat([p[1] for p in pieces],ignore_index=True)
    events=pd.concat([p[3] for p in pieces],ignore_index=True)
    y,w,healthy=make_targets(meta,events)
    model=LearnedModel(kind=args.kind,include_waveforms=not args.no_waveforms,
                       include_residuals=not args.no_residuals,routed=args.routed)
    if args.reference_from:
        old=joblib.load(HERE/'runs'/args.reference_from/'model.joblib')
        model.regressors=old.regressors
    elif model.include_residuals:
        print('fitting healthy reference',flush=True)
        model.fit_reference(x,meta,healthy)
    tx=model.transform(x,meta)
    model.fit_classifier(tx,meta,y,w)
    joblib.dump(model,out/'model.joblib',compress=3)
    write_json(out/'training.json',dict(datasets=args.datasets,kind=args.kind,
        include_waveforms=model.include_waveforms,include_residuals=model.include_residuals,routed=model.routed,
        reference_from=args.reference_from,rows=len(x),assets=int(meta.asset_id.nunique()),
        healthy_rows=int(healthy.sum()),model_features=len(model.columns),
        fault_sample_counts=model.training_info,elapsed_seconds=time.perf_counter()-started,
        source_hashes={p.name:sha(p) for p in [HERE/'features.py',HERE/'physics_features.py',HERE/'learning.py',HERE/'alerts.py',HERE/'run.py']}))
    print('saved',out/'model.joblib','seconds',round(time.perf_counter()-started,1),flush=True)


def predict(args):
    out=HERE/'runs'/args.name
    model=joblib.load(out/'model.joblib')
    for name in args.evaluate:
        meta,x,assets,events,benign=prepared(name)
        transformed=model.transform(x,meta)
        probs=(model.probabilities_from_transformed(transformed,meta) if getattr(model,'is_composite',False)
            else model.probabilities_from_transformed(transformed))
        probs.to_parquet(out/f'{name}_probabilities.parquet',index=False)
        print('predicted',name,flush=True)


def select(args):
    out=HERE/'runs'/args.name
    datasets={name:prepared(name) for name in args.evaluate}
    probs={name:pd.read_parquet(out/f'{name}_probabilities.parquet') for name in args.evaluate}
    trials=[];reports={}
    for threshold,persistence,recovery in itertools.product([.55,.7,.8,.9,.96],[5,15],[60,240]):
        policy=AlertPolicy(threshold=threshold,persistence=persistence,recovery=recovery)
        pid=f't{threshold}_p{persistence}_r{recovery}'
        group={}
        for name,(meta,x,assets,events,benign) in datasets.items():
            pred=make_alerts(meta,probs[name],policy)
            r=score(pred,events,assets,benign)
            group[name]=r
            trials.append(dict(policy_id=pid,dataset=name,**asdict(policy),
                **{k:v for k,v in r.items() if isinstance(v,(int,float)) or v is None}))
        reports[pid]=group
    g=pd.DataFrame(trials);g.to_csv(out/'policy_grid.csv',index=False)
    # Equal weight for familiar public validation and independently generated
    # synthetic development. The final synthetic holdout is not opened here.
    ranking=g.groupby('policy_id').composite_score.mean().sort_values(ascending=False)
    pid=ranking.index[0]
    row=g[g.policy_id==pid].iloc[0]
    policy=AlertPolicy(threshold=float(row.threshold),persistence=int(row.persistence),recovery=int(row.recovery))
    write_json(out/'policy.json',asdict(policy))
    write_json(out/'selection.json',dict(policy_id=pid,selection_mean=float(ranking.iloc[0]),
        development_datasets=args.evaluate,reports=reports[pid]))
    for name,(meta,x,assets,events,benign) in datasets.items():
        pred=make_alerts(meta,probs[name],policy)
        pred.to_csv(out/f'{name}_alerts.csv',index=False)
        r=reports[pid][name]
        print(args.name,pid,name,{k:r[k] for k in ['composite_score','true_positive_events','events_scored','fault_type_recall','false_alert_episodes','excess_repeat_episodes','alerts_scored']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','train','predict','select'])
    p.add_argument('--name',default='learned_01')
    p.add_argument('--kind',choices=['hist','extra'],default='hist')
    p.add_argument('--datasets',nargs='+',default=['original_train','train','train_balanced'])
    p.add_argument('--evaluate',nargs='+',default=['original_validation','development'])
    p.add_argument('--no-waveforms',action='store_true')
    p.add_argument('--no-residuals',action='store_true')
    p.add_argument('--reference-from')
    p.add_argument('--routed',action='store_true')
    a=p.parse_args()
    if a.phase=='prepare':
        for n in a.datasets:prepared(n)
    else:{'train':train,'predict':predict,'select':select}[a.phase](a)
