"""Reproducible expanded encoding experiment. Commands: prepare, train, freeze, test."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse
import datetime
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from protocol import (ROOT,LEGACY,FAMILIES,NEW_SEEDS,TRAIN_ROLES,prepare_sources,load_family,
                      fit_indices,make_targets,rank_common,sha256,dump_json)
from data import ObservedScaler
from models import KINDS,fit_encoder,encode_reconstruct,sequence_indices,packed
from label_informed import fit_label_informed
from fault_detection import binary_metrics,choose_threshold
from task_encoder import fit_task_encoder,encode_task

KINDS_ALL = KINDS+['label_informed','task_encoder']


def representation(bundle,frame,meta):
    x,mask=bundle['preprocessor'].transform(frame)
    if bundle['kind']=='raw':return packed(x,mask)
    if bundle['kind']=='task_encoder':return encode_task(bundle,x,mask)
    seq=sequence_indices(meta.recording.to_numpy()) if bundle['kind']=='temporal_ae' else None
    return encode_reconstruct(bundle,x,mask,seq)[0]


def fit_readouts(z,y,train,validation,raw=False):
    """All targets share the same eligible rows; one KNN neighbor search per encoder."""
    valid=[j for j in range(y.shape[1]) if len(np.unique(y[train,j]))==2]
    probes={};thresholds={};validation_scores={};warning_text=[]
    methods=['logistic','histogram'] if raw else ['logistic','knn']
    for method in methods:
        matrix=np.full((len(validation),y.shape[1]),np.nan,dtype=np.float32)
        objects={}
        if method=='knn':
            probe=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=min(15,len(train)),n_jobs=1))
            probe.fit(z[train],y[train][:,valid]);ps=probe.predict_proba(z[validation])
            # Multioutput returns a list, including when only one target exists.
            if not isinstance(ps,list):ps=[ps]
            for j,p in zip(valid,ps):matrix[:,j]=p[:,1]
            objects={'multioutput':probe,'valid_targets':valid}
        else:
            for j in valid:
                if method=='logistic':
                    probe=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,tol=1e-6,random_state=42))
                else:
                    probe=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.08,
                        l2_regularization=3.,min_samples_leaf=30,class_weight='balanced',early_stopping=False,random_state=42)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');probe.fit(z[train],y[train,j])
                warning_text += [str(w.message) for w in caught]
                matrix[:,j]=probe.predict_proba(z[validation])[:,1];objects[j]=probe
        cuts={}
        for j in valid:
            cut=choose_threshold(y[validation,j],matrix[:,j])
            cuts[j]=dict(value=.5 if cut is None else cut,
                         origin='fixed_0.5_single_class_development' if cut is None else 'development_max_f1')
        probes[method]=objects;thresholds[method]=cuts;validation_scores[method]=matrix
    return probes,thresholds,validation_scores,warning_text


def predict_readout(z,objects,n_targets):
    scores=np.full((len(z),n_targets),np.nan,dtype=np.float32)
    if 'multioutput' in objects:
        # Bound distance matrix memory without changing voting or sample density.
        for start in range(0,len(z),2000):
            ps=objects['multioutput'].predict_proba(z[start:start+2000])
            if not isinstance(ps,list):ps=[ps]
            for j,p in zip(objects['valid_targets'],ps):scores[start:start+2000,j]=p[:,1]
    else:
        for j,probe in objects.items():scores[:,j]=probe.predict_proba(z)[:,1]
    return scores


def score_rows(meta,y,scores,targets,thresholds,identity,split):
    rows=[]
    for recording,g in meta.groupby('recording',sort=True):
        idx=g.index.to_numpy()
        for j,cut in thresholds.items():
            valid=idx[(y[idx,j]>=0)&np.isfinite(scores[idx,j])]
            metrics=binary_metrics(y[valid,j],scores[valid,j],cut['value'])
            rows.append(dict(**identity,recording=recording,family=g.family.iloc[0],
                seed=None if pd.isna(g.seed.iloc[0]) else int(g.seed.iloc[0]),split=split,target=targets[j],
                threshold_origin=cut['origin'],**metrics))
    return rows


def save_predictions(path,meta,y,scores,targets):
    table=meta[['recording','family','seed','available_utc']].reset_index(drop=True).copy()
    for j,target in enumerate(targets):
        table[f'y__{target}']=y[:,j];table[f'score__{target}']=scores[:,j]
    table.to_parquet(path,index=False)


def train(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'frozen.json').exists():raise ValueError('This run is frozen; start a new output directory')
    source_files=[*sorted(ROOT.glob('*.py')),*sorted(LEGACY.glob('*.py')),*sorted((LEGACY/'vendor').glob('*.py'))]
    config=dict(epochs=args.epochs,preparation_sha256=sha256(args.cache/'preparation.json'),
                source_hashes={str(p.resolve()):sha256(p) for p in source_files})
    config_path=out/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Training code, data or settings changed; choose a fresh output directory')
    dump_json(config_path,config)
    for family in (args.families or FAMILIES):
        meta,frame=load_family(args.cache,family)
        train_all=fit_indices(meta);validation_all=np.flatnonzero(meta.role.eq('development') & meta.minute_index.ge(120) & meta.minute_index.mod(5).eq(0))
        eligible=meta.label_eligible.to_numpy()
        train_idx=train_all[eligible[train_all]]
        val_idx=np.flatnonzero(meta.role.eq('development') & meta.minute_index.ge(120) & meta.label_eligible)
        _,targets=make_targets(meta.iloc[train_idx]);y,_=make_targets(meta,targets)
        pre=ObservedScaler().fit(frame.iloc[train_all]);x,mask=pre.transform(frame)
        seq=sequence_indices(meta.recording.to_numpy())
        dest=out/family;dest.mkdir(exist_ok=True)
        variants=[(f'frozen_{k}',k,True) for k in KINDS+['label_informed'] if (LEGACY/'results'/family/f'{k}.joblib').exists()]
        variants += [(f'expanded_{k}',k,False) for k in KINDS_ALL]+[('raw_features','raw',False)]
        for variant,kind,frozen in variants:
            done=dest/f'{variant}_metrics.json'
            if done.exists():continue
            started=time.monotonic();print('TRAIN',family,variant,'rows',len(train_all),flush=True)
            if frozen:
                bundle=joblib.load(LEGACY/'results'/family/f'{kind}.joblib')
                bundle['source_model_sha256']=sha256(LEGACY/'results'/family/f'{kind}.joblib')
            else:
                if kind=='raw':bundle={'kind':'raw','dimensions':int(x.shape[1]*2),'type':'raw'}
                elif kind=='task_encoder':
                    development_fit=validation_all[eligible[validation_all]]
                    bundle=fit_task_encoder(x,mask,train_idx,development_fit,y[train_idx],y[development_fit],max_epochs=args.epochs)
                elif kind=='label_informed':
                    bundle=fit_label_informed(x,mask,train_all,validation_all,train_idx,
                        meta.fault_label.to_numpy()[train_idx],max_epochs=args.epochs)
                else:
                    bundle=fit_encoder(kind,x,mask,train_all,validation_all,sequence=seq,max_epochs=args.epochs)
                bundle['preprocessor']=pre
            z=representation(bundle,frame,meta)
            probes,cuts,scores,caught=fit_readouts(z,y,train_idx,val_idx,kind=='raw')
            payload=dict(encoder=bundle,readouts=probes,thresholds=cuts,targets=targets,family=family,
                         variant=variant,train_rows=len(train_idx),fit_rows=len(train_all),
                         input_preparation_sha256=sha256(args.cache/'preparation.json'),warnings=caught)
            joblib.dump(payload,dest/f'{variant}.joblib',compress=3)
            rows=[]
            for method,p in scores.items():
                identity=dict(variant=variant,readout=method,candidate=f'{variant}/{method}')
                rows += score_rows(meta.iloc[val_idx].reset_index(drop=True),y[val_idx],p,targets,cuts[method],identity,'development')
                save_predictions(dest/f'{variant}_{method}_development.parquet',meta.iloc[val_idx],y[val_idx],p,targets)
            dump_json(done,dict(metrics=rows,seconds=time.monotonic()-started,warnings=caught,
                                best_epoch=bundle.get('best_epoch'),history=bundle.get('history',[])))
            print('DONE',family,variant,round(time.monotonic()-started,1),'seconds',flush=True)
    aggregate_development(out)


def aggregate_development(out):
    metrics=[]
    for p in sorted(out.glob('PUMP-*/*_metrics.json')):metrics.extend(json.loads(p.read_text())['metrics'])
    table=pd.DataFrame(metrics);table.to_csv(out/'development_metrics.csv',index=False)
    decisions={}
    for method in ['logistic','knn']:
        candidates=[f'expanded_{k}/{method}' for k in KINDS_ALL]
        rows=table[table.target.eq('ANY_FAULT')]
        decisions[method]=rank_common(rows,candidates)
    dump_json(out/'selection.json',dict(primary='expanded encoders, ANY_FAULT, logistic regression',
        selected_candidate=decisions['logistic'][0]['candidate'] if decisions['logistic'] else None,
        rankings=decisions,chosen_on='development only; historical holdout is reused as development',
        fresh_test_seeds=NEW_SEEDS,test_labels_used=False))


def freeze(args):
    aggregate_development(args.output)
    for family in FAMILIES:
        for kind in KINDS_ALL+['raw']:
            variant='raw_features' if kind=='raw' else f'expanded_{kind}'
            if not (args.output/family/f'{variant}_metrics.json').exists():raise ValueError(f'Missing {family}/{variant}')
    challenge=ROOT.parents[2]
    files=[*sorted(args.output.glob('PUMP-*/*.joblib')),args.output/'selection.json',args.output/'run_config.json',args.cache/'preparation.json',
           *sorted(ROOT.glob('*.py')),*sorted((ROOT/'generator').glob('*.py')),*sorted(LEGACY.glob('*.py')),
           *sorted((LEGACY/'vendor').glob('*.py')),*sorted((challenge/'src/retrofit_sim').glob('*.py')),
           challenge/'configs/fleet_public.yaml']
    destination=args.output/'frozen.json'
    if destination.exists():raise ValueError('Already frozen')
    dump_json(destination,dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hashes={str(p.resolve()):sha256(p) for p in files},test_seeds=NEW_SEEDS,
        test_generated=False,model_selection=json.loads((args.output/'selection.json').read_text())))
    print('Frozen',destination,flush=True)


def verify_freeze(out):
    frozen=json.loads((out/'frozen.json').read_text())
    changed=[p for p,digest in frozen['hashes'].items() if sha256(p)!=digest]
    if changed:raise ValueError(f'Frozen inputs changed: {changed}')
    return frozen


def test(args):
    frozen=verify_freeze(args.output)
    manifest=json.loads((args.test_data/'manifest.json').read_text())
    if manifest['seeds']!=frozen['test_seeds']:raise ValueError('Unexpected test seeds')
    if manifest['frozen_pipeline_manifest_sha256']!=sha256(args.output/'frozen.json'):raise ValueError('Test generation did not use this freeze')
    test_cache=args.cache.parent/'fresh_test'
    prepare_sources([(args.test_data,'test',False)],test_cache)
    results=[]
    for family in FAMILIES:
        meta,frame=load_family(test_cache,family)
        use=np.flatnonzero(meta.minute_index.ge(120)&meta.label_eligible)
        for p in sorted((args.output/family).glob('*.joblib')):
            payload=joblib.load(p);variant=payload['variant'];targets=payload['targets']
            z=representation(payload['encoder'],frame,meta)[use];y,_=make_targets(meta.iloc[use],targets)
            for method,probe in payload['readouts'].items():
                scores=predict_readout(z,probe,len(targets))
                identity=dict(variant=variant,readout=method,candidate=f'{variant}/{method}')
                rows=score_rows(meta.iloc[use].reset_index(drop=True),y,scores,targets,payload['thresholds'][method],identity,'fresh_test')
                results.extend(rows)
                save_predictions(args.output/family/f'{variant}_{method}_test.parquet',meta.iloc[use],y,scores,targets)
            print('TEST',family,variant,flush=True)
        pd.DataFrame(results).to_csv(args.output/'test_metrics.partial.csv',index=False)
    pd.DataFrame(results).to_csv(args.output/'test_metrics.csv',index=False)
    verify_freeze(args.output)
    dump_json(args.output/'test_completed.json',dict(frozen_manifest_sha256=sha256(args.output/'frozen.json'),
        generated_data_sha256=sha256(args.test_data/'manifest.json'),metric_rows=len(results),frozen_inputs_unchanged=True,
        completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','freeze','test'])
    p.add_argument('--synthetic-root',type=Path);p.add_argument('--test-data',type=Path)
    p.add_argument('--cache',type=Path,default=ROOT/'.cache/development')
    p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--families',nargs='+')
    args=p.parse_args()
    if args.command=='prepare':
        if args.synthetic_root is None:p.error('--synthetic-root required')
        sources=[(ROOT.parents[1]/'RETROFIT_AI_PARTICIPANTS_2026/data/full','train',True)]
        sources += [(args.synthetic_root/r,'train',False) for r in TRAIN_ROLES]
        sources += [(args.synthetic_root/r,'development',False) for r in ['development','holdout']]
        prepare_sources(sources,args.cache)
    elif args.command=='train':train(args)
    elif args.command=='freeze':freeze(args)
    else:test(args)


if __name__=='__main__':main()
