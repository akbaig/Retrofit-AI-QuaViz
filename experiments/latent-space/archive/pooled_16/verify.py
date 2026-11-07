"""Verify exported artifacts, source integrity, saved encoders, and split rules."""
import json,joblib
from pathlib import Path
import numpy as np
import pandas as pd
from data import ROOT,DATA,sha256,preparation_fingerprint,check_fingerprint,sequence_indices,evaluation_eligible
from models import transform_model

out=ROOT/'results'
meta=pd.read_parquet(out/'metadata.parquet')
assert len(meta)==120960 and not meta.duplicated(['asset_id','available_utc']).any()
assert (meta.available_utc-meta.window_start_utc).dt.total_seconds().eq(60).all()
assert meta.loc[meta.split.eq('test'),'label'].eq('UNLABELED_TEST').all()
check_fingerprint(json.loads((out/'preparation_manifest.json').read_text())['fingerprint'],preparation_fingerprint())
audit=json.loads((out/'data_audit.json').read_text())
for name,digest in audit['source_hashes'].items():assert sha256(DATA/name)==digest,name
frame=pd.read_parquet(out/'features_development.parquet')
dev=pd.read_parquet(out/'metadata_development.parquet')
test=pd.read_parquet(out/'features_test.parquet');testmeta=pd.read_parquet(out/'metadata_test.parquet')
selection=json.loads((out/'selection.json').read_text());models=pd.read_csv(out/'model_comparison.csv').model.tolist()
checks=[]
for name in models:
    z=np.load(out/f'{name}_embeddings.npy')
    assert z.shape==(120960,16) and np.isfinite(z).all(),name
    export=pd.read_parquet(out/f'{name}_embeddings.parquet')
    np.testing.assert_array_equal(export[[f'z{i:02d}' for i in range(16)]].to_numpy(),z)
    pd.testing.assert_frame_equal(export[['asset_id','available_utc','split']],meta[['asset_id','available_utc','split']])
    bundle=joblib.load(out/'models'/f'{name}.joblib')
    assert bundle['fit_assets']==[f'PUMP-{i:02d}' for i in range(1,8)],name
    n=10100
    x,mask=bundle['preprocessor'].transform(frame.iloc[:n])
    reloaded=transform_model(bundle,x,mask,sequence_indices(dev.asset_id.iloc[:n]))
    np.testing.assert_allclose(reloaded,z[:n],atol=1e-5,rtol=1e-5)
    checks.append({'model':name,'shape':list(z.shape),'reload_max_abs_error':float(np.max(np.abs(reloaded-z[:n]))),'fit_assets':bundle['fit_assets']})
winner=selection['selected_model'];bundle=joblib.load(out/'models'/f'{winner}.joblib')
check_fingerprint(selection['model_sha256'],sha256(out/'models'/f'{winner}.joblib'))
xt,mt=bundle['preprocessor'].transform(test)
zt=transform_model(bundle,xt,mt,sequence_indices(testmeta.asset_id))
np.testing.assert_allclose(zt,np.load(out/'selected_embeddings.npy')[-len(test):],atol=1e-5,rtol=1e-5)
np.testing.assert_array_equal(np.load(out/'selected_embeddings.npy'),np.load(out/f'{winner}_embeddings.npy'))
assert not evaluation_eligible(meta).loc[meta.has_unobservable_event|meta.has_predetectable_event].any()
result={'passed':True,'rows':len(meta),'dimensions':16,'models':checks,'source_hashes_unchanged':True,
        'test_labels_unknown':True,'selected_test_reload':True,'selected_model':winner,
        'ambiguous_minutes_excluded':int((dev.has_unobservable_event|dev.has_predetectable_event).sum())}
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
