"""Explicit label-informed comparison; retains the original five-model results."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import json,time,joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data import ROOT,sequence_indices,sha256,evaluation_eligible,fingerprint,check_fingerprint,preparation_fingerprint
from models import fit_contrastive,encode_neural,transform_model
from evaluate import evaluate

out=ROOT/'results'
check_fingerprint(json.loads((out/'preparation_manifest.json').read_text())['fingerprint'],preparation_fingerprint())
meta=pd.read_parquet(out/'metadata_development.parquet');frame=pd.read_parquet(out/'features_development.parquet')
y=np.load(out/'labels_development.npy');names=json.loads((out/'fault_names.json').read_text())
pre=joblib.load(out/'models/preprocessor_physical.joblib');x,mask=pre.transform(frame)
valid=meta.split.eq('train')&evaluation_eligible(meta)&meta.minute_index.mod(5).eq(0)
train=np.flatnonzero(valid);alltrain=np.flatnonzero(meta.split.eq('train'))
original=out/'unsupervised_comparison.csv'
if not original.exists():original.write_bytes((out/'model_comparison.csv').read_bytes())
sel=out/'unsupervised_selection.json'
if not sel.exists():sel.write_bytes((out/'selection.json').read_bytes())
rows=[]
for seed in [42,7,123]:
    kind='supervised_contrastive' if seed==42 else f'supervised_contrastive_seed{seed}'
    path=out/'models'/f'{kind}.joblib'
    signature=fingerprint([ROOT/'models.py',ROOT/'extend_contrastive.py',out/'features_development.parquet',out/'labels_development.npy',out/'preparation_manifest.json'],{'seed':seed,'epochs':30})
    if path.exists():
        bundle=joblib.load(path);check_fingerprint(bundle.get('input_fingerprint'),signature);z=np.load(out/f'{kind}_development.npy')
    else:
        start=time.monotonic();net,history=fit_contrastive(x,mask,train,y,seed=seed)
        z=encode_neural(net,x,mask);scaler=StandardScaler().fit(z[alltrain]);z=scaler.transform(z).astype(np.float32)
        bundle={'type':'neural','temporal':False,'state_dict':net.state_dict(),'input_dim':x.shape[1],
            'latent_scaler':scaler,'kind':kind,'seed':seed,'seconds':time.monotonic()-start,'history':history,
            'feature_set':'physical','preprocessor':pre,'fit_assets':sorted(meta.iloc[train].asset_id.unique().tolist()),'uses_training_labels':True,'input_fingerprint':signature}
        joblib.dump(bundle,path);np.save(out/f'{kind}_development.npy',z)
    metrics,km,clusters,probe=evaluate(z,meta,y,names)
    joblib.dump(km,out/'models'/f'{kind}_kmeans.joblib');np.save(out/f'{kind}_clusters_development.npy',clusters)
    np.savez_compressed(out/f'{kind}_probe.npz',**probe)
    metrics.update({'model':kind,'latent_dimensions':16,'input_features':x.shape[1],'fit_seconds':bundle['seconds'],'uses_training_labels':True})
    (out/f'{kind}_metrics.json').write_text(json.dumps(metrics,indent=2));(out/f'{kind}_training.json').write_text(json.dumps(bundle['history'],indent=2))
    rows.append({k:v for k,v in metrics.items() if k not in ['per_fault','per_asset']})
    print(kind,'macro AP',metrics['macro_ap'],'per asset',metrics['per_asset'],flush=True)
table=pd.concat([pd.read_csv(original).assign(uses_training_labels=False),pd.DataFrame(rows)],ignore_index=True).sort_values('macro_ap',ascending=False)
table.to_csv(out/'model_comparison.csv',index=False)
# Seed 42 is prespecified. Other seeds measure sensitivity, never cherry-picked.
eligible=table[~table.model.str.contains('_seed')];winner=str(eligible.iloc[0]['model'])
selection={'selected_model':winner,'best_unsupervised':json.loads(sel.read_text())['selected_model'],
    'criterion':'validation macro average precision of fixed 15-neighbor multilabel probe; prespecified seed 42',
    'dimensions':16,'seed':42,'selected_before_test_transform':True,
    'uses_training_labels':winner=='supervised_contrastive','fitted_assets':sorted(meta.iloc[train].asset_id.unique().tolist()),
    'model_sha256':sha256(out/'models'/f'{winner}.joblib'),
    'seed_sensitivity':pd.DataFrame(rows)[['model','macro_ap']].to_dict(orient='records')}
(out/'selection.json').write_text(json.dumps(selection,indent=2))
# Test features were encoded once in the first run, never inspected for selection.
mt=pd.read_parquet(out/'metadata_test.parquet');ft=pd.read_parquet(out/'features_test.parquet');st=sequence_indices(mt.asset_id)
xt,maskt=pre.transform(ft);allmeta=pd.concat([meta,mt],ignore_index=True)
for row in rows:
    kind=row['model'];bundle=joblib.load(out/'models'/f'{kind}.joblib')
    zt=transform_model(bundle,xt,maskt,st);z=np.concatenate([np.load(out/f'{kind}_development.npy'),zt])
    assert np.isfinite(z).all()
    np.save(out/f'{kind}_embeddings.npy',z)
    km=joblib.load(out/'models'/f'{kind}_kmeans.joblib');export=allmeta[['asset_id','window_start_utc','available_utc','split']].copy();export['cluster']=km.predict(z)
    for j in range(16):export[f'z{j:02d}']=z[:,j]
    export.to_parquet(out/f'{kind}_embeddings.parquet',index=False)
np.save(out/'selected_embeddings.npy',np.load(out/f'{winner}_embeddings.npy'))
print('SELECTED',winner,flush=True)
