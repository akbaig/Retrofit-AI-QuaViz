"""Run from this directory with .venv/bin/python run.py; resumable per model."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse,json,time,joblib
import numpy as np
import pandas as pd
from data import ROOT,Preprocessor,prepare,feature_sets,sequence_indices,audit,sha256,preparation_fingerprint,fingerprint,check_fingerprint
from models import fit_model,transform_model
from evaluate import evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=['raw_pca','physical_pca','rbf_pca','denoising_ae','temporal_ae']);parser.add_argument('--skip-test',action='store_true');args=parser.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True);(out/'models').mkdir(exist_ok=True)
    signature=preparation_fingerprint()
    if (out/'features_development.parquet').exists():
        marker=out/'preparation_manifest.json'
        check_fingerprint(json.loads(marker.read_text()).get('fingerprint') if marker.exists() else None,signature)
    if not (out/'features_development.parquet').exists():
        audit(out);meta,frame,y,names=prepare()
        meta.to_parquet(out/'metadata_development.parquet',index=False);frame.to_parquet(out/'features_development.parquet',index=False)
        np.save(out/'labels_development.npy',y);(out/'fault_names.json').write_text(json.dumps(names))
        (out/'preparation_manifest.json').write_text(json.dumps({'fingerprint':signature,'protocol':2},indent=2))
    meta=pd.read_parquet(out/'metadata_development.parquet');frame=pd.read_parquet(out/'features_development.parquet')
    y=np.load(out/'labels_development.npy');names=json.loads((out/'fault_names.json').read_text())
    train=np.flatnonzero(meta.split.eq('train'));seq=sequence_indices(meta.asset_id)
    sets=feature_sets(frame);preprocessors={};prepared={}
    for name,cols in sets.items():
        pre=Preprocessor().fit(frame.loc[train,cols]);preprocessors[name]=pre;prepared[name]=pre.transform(frame)
        joblib.dump(pre,out/'models'/f'preprocessor_{name}.joblib')
    summaries=[]
    for kind in args.models:
        feature_set='raw' if kind=='raw_pca' else 'physical';x,mask=prepared[feature_set]
        path=out/'models'/f'{kind}.joblib'
        model_signature=fingerprint([ROOT/'models.py',out/'features_development.parquet'],{'preparation':signature,'kind':kind,'seed':42,'dimensions':16})
        if path.exists() and (out/f'{kind}_development.npy').exists():
            bundle=joblib.load(path);z=np.load(out/f'{kind}_development.npy');print('Using saved',kind,flush=True)
            check_fingerprint(bundle.get('input_fingerprint'),model_signature)
        else:
            print('Fitting',kind,'inputs',x.shape[1],flush=True)
            z,bundle=fit_model(kind,x,mask,train,sequence=seq)
            bundle['feature_set']=feature_set;bundle['fit_assets']=sorted(meta.iloc[train].asset_id.unique().tolist())
            bundle['preprocessor']=preprocessors[feature_set]
            bundle['input_fingerprint']=model_signature
            joblib.dump(bundle,path);np.save(out/f'{kind}_development.npy',z)
        assert np.isfinite(z).all() and z.shape==(len(meta),16)
        metrics,km,clusters,probe=evaluate(z,meta,y,names)
        joblib.dump(km,out/'models'/f'{kind}_kmeans.joblib');np.save(out/f'{kind}_clusters_development.npy',clusters)
        np.savez_compressed(out/f'{kind}_probe.npz',**probe)
        metrics.update({'model':kind,'latent_dimensions':16,'input_features':x.shape[1],'fit_seconds':bundle['seconds']})
        (out/f'{kind}_metrics.json').write_text(json.dumps(metrics,indent=2));(out/f'{kind}_training.json').write_text(json.dumps(bundle['history'],indent=2))
        summaries.append({k:v for k,v in metrics.items() if k not in ['per_fault','per_asset']})
        print(kind,'AP',metrics['macro_ap'],'running AP',metrics['running_macro_ap'],'asset NMI',metrics['asset_cluster_nmi'],flush=True)
    table=pd.DataFrame(summaries).sort_values('macro_ap',ascending=False);table.to_csv(out/'model_comparison.csv',index=False)
    winner=str(table.iloc[0]['model'])
    selection={'selected_model':winner,'criterion':'validation macro average precision of fixed 15-neighbor multilabel probe',
        'model_candidates':args.models,'selected_before_test_transform':True,'dimensions':16,'seed':42,
        'fitted_assets':sorted(meta.iloc[train].asset_id.unique().tolist()),'model_sha256':sha256(out/'models'/f'{winner}.joblib')}
    (out/'selection.json').write_text(json.dumps(selection,indent=2));print('SELECTED',winner,flush=True)
    if not args.skip_test:
        if not (out/'features_test.parquet').exists():
            mt,ft,_,_=prepare(('test',));mt.to_parquet(out/'metadata_test.parquet',index=False);ft.to_parquet(out/'features_test.parquet',index=False)
        mt=pd.read_parquet(out/'metadata_test.parquet');ft=pd.read_parquet(out/'features_test.parquet');st=sequence_indices(mt.asset_id)
        for kind in args.models:
            bundle=joblib.load(out/'models'/f'{kind}.joblib');xt,maskt=bundle['preprocessor'].transform(ft)
            zt=transform_model(bundle,xt,maskt,st);assert np.isfinite(zt).all()
            z=np.concatenate([np.load(out/f'{kind}_development.npy'),zt]);np.save(out/f'{kind}_embeddings.npy',z)
            allmeta=pd.concat([meta,mt],ignore_index=True)
            km=joblib.load(out/'models'/f'{kind}_kmeans.joblib');clusters=km.predict(z)
            export=allmeta[['asset_id','window_start_utc','available_utc','split']].copy();export['cluster']=clusters
            for j in range(16):export[f'z{j:02d}']=z[:,j]
            export.to_parquet(out/f'{kind}_embeddings.parquet',index=False)
        allmeta.to_parquet(out/'metadata.parquet',index=False)
        np.save(out/'selected_embeddings.npy',np.load(out/f'{winner}_embeddings.npy'))
    print(table[['model','macro_ap','running_macro_ap','asset_cluster_nmi']].to_string(index=False),flush=True)

if __name__=='__main__':main()
