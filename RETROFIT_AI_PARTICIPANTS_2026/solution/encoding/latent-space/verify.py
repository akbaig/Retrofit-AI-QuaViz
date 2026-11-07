"""Recompute saved representations and scores; audit per-pump protocol."""
import json,joblib
import numpy as np
import pandas as pd
from data import ROOT,DATA,INPUT_FILES,prepare,sha256,ObservedScaler
from models import KINDS,encode_reconstruct,sequence_indices
from label_informed import LABEL_KIND,LABEL_FILES,training_supervision
from evaluate import reconstruction_metrics


def main():
    out=ROOT/'results';meta,frame=prepare(out)
    preparation=json.loads((out/'preparation.json').read_text())
    assert preparation['event_labels_read'] is False
    assert set(preparation['source_hashes'])==set(INPUT_FILES)
    for name,digest in preparation['source_hashes'].items():assert sha256(DATA/name)==digest
    comparison=pd.read_csv(out/'model_comparison.csv');selected=pd.read_csv(out/'selected_models.csv')
    labels_manifest=json.loads((out/'labels_manifest.json').read_text())
    for name,digest in labels_manifest['event_sources'].items():assert sha256(DATA/name)==digest
    events=pd.concat([pd.read_csv(DATA/name) for name in LABEL_FILES],ignore_index=True)
    availability={r['asset_id']:r for r in labels_manifest['pumps']}
    expected_models=60+sum(r['available'] for r in availability.values())
    assert len(comparison)==expected_models and len(selected)==12
    assert set(comparison.model)==set(KINDS+[LABEL_KIND]) and set(comparison.dimensions)=={32}
    # The extension must preserve every baseline fitted model and full vector file.
    baseline=json.loads((ROOT/'archive/per_pump_32_unsupervised/artifact_manifest.json').read_text())
    for key,digest in baseline['artifact_sha256'].items():
        path=ROOT/key
        if path.parent.name.startswith('PUMP-') and any(path.name in [f'{kind}.joblib',f'{kind}_embeddings.npy'] for kind in KINDS):
            assert sha256(path)==digest,('Baseline changed',key)
    total=0;checks=[]
    for asset,group in meta.groupby('asset_id',sort=True):
        dest=out/asset;m=group.reset_index(drop=True);f=frame.loc[group.index].reset_index(drop=True)
        train=np.flatnonzero(m.split.eq('train'));val=np.flatnonzero(m.split.eq('validation')&m.score_eligible);test=np.flatnonzero(m.split.eq('test')&m.score_eligible)
        assert len(m)==10080 and (len(train),len(val),len(test))==(6048,1896,1896)
        assert m.available_utc.is_monotonic_increasing and m.available_utc.is_unique
        assert m.split.iloc[6047]=='train' and m.split.iloc[6048]=='validation' and m.split.iloc[8064]=='test'
        pre=joblib.load(dest/'preprocessor.joblib');expected=ObservedScaler().fit(f.iloc[train])
        assert pre.columns==expected.columns and pre.excluded==expected.excluded
        np.testing.assert_array_equal(pre.center,expected.center);np.testing.assert_array_equal(pre.scale,expected.scale)
        x,mask=pre.transform(f);seq=sequence_indices(m.asset_id)
        selection=json.loads((dest/'selection.json').read_text());winner=selection['selected_model']
        assert selection['uses_labels']==(winner==LABEL_KIND) and selection['test_used_for_selection'] is False
        scores=comparison[comparison.asset_id.eq(asset)].set_index('model')
        assert scores.validation_nmse.idxmin()==winner
        assert sha256(dest/f'{winner}.joblib')==selection['model_sha256']
        kinds=KINDS+([LABEL_KIND] if availability[asset]['available'] else [])
        assert set(scores.index)==set(kinds)
        for kind in kinds:
            bundle=joblib.load(dest/f'{kind}.joblib')
            assert bundle['uses_labels']==(kind==LABEL_KIND) and bundle['asset_id']==asset
            if kind==LABEL_KIND:
                supervised,states=training_supervision(m,events[events.asset_id.eq(asset)],True)
                np.testing.assert_array_equal(bundle['supervised_indices'],supervised)
                assert np.isin(supervised,train).all() and len(np.unique(states))>=2
                assert set(bundle['training_states'])==set(states)
            assert bundle['train_rows']==6048 and bundle['fit_last_available_utc']==str(m.available_utc.iloc[6047])
            z,recon=encode_reconstruct(bundle,x,mask,seq)
            saved=np.load(dest/f'{kind}_embeddings.npy')
            assert saved.shape==(10080,32) and saved.dtype==np.float32 and np.isfinite(saved).all()
            np.testing.assert_allclose(z,saved,rtol=1e-5,atol=1e-5)
            table=pd.read_parquet(dest/f'{kind}_embeddings.parquet')
            np.testing.assert_array_equal(table.minute_index,m.minute_index)
            np.testing.assert_array_equal(table[[f'z{j:02d}' for j in range(32)]].to_numpy(),saved)
            clusters=joblib.load(dest/f'{kind}_kmeans.joblib').predict(z)
            np.testing.assert_array_equal(clusters,table.cluster)
            for name,idx in [('train',train),('validation',val),('test',test)]:
                loss=reconstruction_metrics(x[idx],recon[idx],mask[idx])['nmse']
                assert np.isclose(loss,scores.loc[kind,f'{name}_nmse'],rtol=1e-6,atol=1e-8),(asset,kind,name)
            # Causal prefix inference must be identical to the prefix of a full batch.
            prefix=512
            pz,_=encode_reconstruct(bundle,x[:prefix],mask[:prefix],seq[:prefix])
            np.testing.assert_allclose(pz,z[:prefix],rtol=2e-5,atol=2e-5)
            total+=1
        np.testing.assert_array_equal(np.load(dest/'selected_embeddings.npy'),np.load(dest/f'{winner}_embeddings.npy'))
        for name,count in [('train',6048),('validation',2016),('test',2016)]:
            split=pd.read_parquet(dest/f'{name}.parquet')
            assert len(split)==count and set(split.split)=={name}
        checks.append({'pump':asset,'models_verified':len(kinds),'selected_model':winner,'rows':len(m)})
        print('Verified',asset,len(kinds),'encoders, split exports, scores, prefix inference',flush=True)
    combined=pd.read_parquet(out/'selected_embeddings.parquet');z=np.load(out/'selected_embeddings.npy')
    assert len(combined)==120960 and z.shape==(120960,32)
    assert not combined.duplicated(['asset_id','available_utc']).any()
    np.testing.assert_array_equal(z,combined[[f'z{i:02d}' for i in range(32)]].to_numpy())
    summary={'passed':True,'models_verified':total,'rows':len(meta),'dimensions':32,'pumps':checks,'source_hashes_unchanged':True,'label_files_read':True,
             'label_informed_training_rows_verified':True,'baseline_models_and_embeddings_unchanged':True,'selection_uses_labels':False}
    (out/'verification.json').write_text(json.dumps(summary,indent=2))
    print(f'PASS: {total} fitted models; 120,960 selected 32D rows; 12 independent chronological splits.',flush=True)
if __name__=='__main__':main()
