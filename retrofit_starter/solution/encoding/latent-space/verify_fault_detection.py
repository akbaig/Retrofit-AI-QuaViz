"""Audit the fault-probe exports against frozen inputs and saved classifiers."""
import json
import numpy as np
import pandas as pd
import joblib
from data import ROOT,sha256
from fault_detection import MODELS,ANY,target_vector,choose_threshold,binary_metrics,selection_from_validation


def main():
    source=ROOT/'results';out=source/'fault_detection'
    manifest=json.loads((out/'manifest.json').read_text());prior=json.loads((source/'artifact_manifest.json').read_text())
    for path,digest in manifest['input_sha256'].items():assert sha256(ROOT/path)==digest
    frozen=0
    for path,digest in prior['artifact_sha256'].items():
        p=ROOT/path
        if p.parent.name.startswith('PUMP-') and p.name in [name+suffix for name in MODELS for suffix in ['.joblib','_embeddings.npy']]:
            assert sha256(p)==digest,('encoder changed',path);frozen+=1
    assert frozen==134,('expected all 67 frozen encoders and vector arrays',frozen)
    metrics=pd.read_csv(out/'metrics.csv');predictions=pd.read_parquet(out/'predictions.parquet')
    selection=json.loads((out/'selection.json').read_text())
    calculated=selection_from_validation(metrics[metrics.split.eq('validation')])
    assert selection['uses_test_for_selection'] is False
    for key in calculated:
        assert selection['decisions'][key]['winner']==calculated[key]['winner']
        assert selection['decisions'][key]['common_cases']==calculated[key]['common_cases']
    assert selection['decisions']['any_fault_logistic']['common_cases']==[['PUMP-01',ANY],['PUMP-09',ANY]]
    assert not metrics[metrics.split.eq('test')].ranking_evaluable.any()
    checked=0
    for model_file in sorted(out.glob('PUMP-*/*/*.joblib')):
        bundle=joblib.load(model_file);asset=bundle['asset_id'];model=bundle['encoder'];target=bundle['target'];method=model_file.stem.split('__')[-1]
        meta=pd.read_parquet(source/asset/'metadata.parquet');labels=pd.read_parquet(source/asset/'label_metadata.parquet');z=np.load(source/asset/f'{model}_embeddings.npy');y=target_vector(labels,target)
        expected_train=np.flatnonzero(meta.split.eq('train').to_numpy()&(y>=0))
        np.testing.assert_array_equal(bundle['train_indices'],expected_train)
        np.testing.assert_allclose(bundle['probe'].steps[0][1].mean_,z[expected_train].astype(float).mean(axis=0),atol=1e-10,rtol=1e-10)
        use=predictions.asset_id.eq(asset)&predictions.model.eq(model)&predictions.target.eq(target)&predictions.readout.eq(method)
        data=predictions[use]
        val=data[data.split.eq('validation')]
        threshold=choose_threshold(val.y.to_numpy(),val.score.to_numpy())
        np.testing.assert_allclose(bundle['threshold'],.5 if threshold is None else threshold,atol=1e-15)
        for split in ['validation','test']:
            rows=data[data.split.eq(split)];idx=rows.minute_index.to_numpy();expected=np.flatnonzero(meta.split.eq(split).to_numpy()&meta.score_eligible.to_numpy()&(y>=0))
            np.testing.assert_array_equal(idx,expected);np.testing.assert_array_equal(rows.y,y[idx])
            if len(idx):
                # Replay the original full batch: float32 BLAS reductions can
                # differ slightly when the same rows are scored in tiny batches.
                recomputed=bundle['probe'].predict_proba(z[idx])[:,1]
                np.testing.assert_allclose(rows.score.to_numpy(),recomputed,atol=1e-12,rtol=1e-12)
            stat=binary_metrics(rows.y.to_numpy(),rows.score.to_numpy(),bundle['threshold'])
            recorded=metrics[metrics.asset_id.eq(asset)&metrics.model.eq(model)&metrics.target.eq(target)&metrics.readout.eq(method)&metrics.split.eq(split)].iloc[0]
            for key in ['tp','fp','tn','fn','ap','auroc','recall','false_positive_rate','f1']:
                if stat[key] is None:assert pd.isna(recorded[key]),(asset,model,target,key)
                else:assert np.isclose(stat[key],recorded[key],rtol=1e-10,atol=1e-12),(asset,model,target,key)
        checked+=1
    assert checked==manifest['probe_fits'] and checked==192
    result={'passed':True,'probes_verified':checked,'frozen_encoder_artifacts_verified':frozen,'test_ranking_evaluable_cases':0,
            'train_only_scaling_and_fit_indices':True,'predictions_replayed':True,'scores_recomputed':True,'validation_selection_recomputed':True}
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
