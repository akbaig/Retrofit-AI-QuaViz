"""Verify the frozen run, saved score tables, labels, metrics and selected-model replay."""
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import FAMILIES,load_family,make_targets,dump_json,sha256
from integrity import verify_data_seal,validate_test_dataset
from run_expanded import verify_freeze,representation,predict_readout
from fault_detection import binary_metrics


def main():
    out=ROOT/'results';verify_freeze(out);verify_data_seal(out/'data_seal.json')
    original_meta=pd.concat([load_family(ROOT/'.cache/development',f)[0] for f in FAMILIES],ignore_index=True)
    validate_test_dataset(ROOT/'synthetic_data/holdout',out/'frozen.json',original_meta,
        ROOT/'generator/synthetic.py',ROOT.parents[2]/'configs/fleet_public.yaml')
    metrics=pd.read_csv(out/'test_metrics.csv');checked=0;files=0
    for (family,variant,readout),g in metrics.groupby(['family','variant','readout']):
        path=out/family/f'{variant}_{readout}_test.parquet';pred=pd.read_parquet(path);files+=1
        payload=joblib.load(out/family/f'{variant}.joblib')
        meta,_=load_family(ROOT/'.cache/fresh_test',family)
        use=np.flatnonzero(meta.minute_index.ge(120)&meta.label_eligible)
        expected=meta.iloc[use].reset_index(drop=True)
        assert pred.recording.tolist()==expected.recording.tolist()
        assert pd.to_datetime(pred.available_utc,utc=True).tolist()==pd.to_datetime(expected.available_utc,utc=True).tolist()
        y,_=make_targets(expected,payload['targets'])
        for j,target in enumerate(payload['targets']):np.testing.assert_array_equal(pred[f'y__{target}'],y[:,j])
        for row in g.itertuples():
            part=pred[pred.recording.eq(row.recording)];values=part[f'y__{row.target}'].to_numpy();scores=part[f'score__{row.target}'].to_numpy()
            keep=(values>=0)&np.isfinite(scores)
            result=binary_metrics(values[keep],scores[keep],row.threshold)
            for key in ['rows','positives','negatives','tp','fp','tn','fn','ap','auroc','recall','false_positive_rate','f1']:
                actual=getattr(row,key);expected_value=result[key]
                if expected_value is None:assert pd.isna(actual)
                else:assert np.isclose(actual,expected_value,rtol=1e-9,atol=1e-9),(family,variant,readout,key)
            checked+=1
    plan=json.loads((out/'comparison_plan.json').read_text());variant,readout=plan['selected_encoder'].split('/')
    for family in FAMILIES:
        payload=joblib.load(out/family/f'{variant}.joblib');meta,frame=load_family(ROOT/'.cache/fresh_test',family)
        use=np.flatnonzero(meta.minute_index.ge(120)&meta.label_eligible)
        z=representation(payload['encoder'],frame,meta)[use]
        chosen=np.linspace(0,len(z)-1,min(100,len(z)),dtype=int)
        scores=predict_readout(z[chosen],payload['readouts'][readout],len(payload['targets']))
        saved=pd.read_parquet(out/family/f'{variant}_{readout}_test.parquet')
        expected=saved[[f'score__{t}' for t in payload['targets']]].to_numpy()[chosen]
        np.testing.assert_allclose(scores,expected,atol=1e-6,rtol=1e-6,equal_nan=True)
    verify_freeze(out)
    dump_json(out/'verification.json',dict(frozen_inputs_unchanged=True,generated_test_artifacts_verified=True,
        metric_rows_recomputed=checked,prediction_files_checked=files,prediction_labels_and_times_verified=True,
        selected_model_replayed_families=len(FAMILIES),max_replay_rows_per_family=100,
        selected_candidate=plan['selected_encoder'],test_metrics_sha256=sha256(out/'test_metrics.csv')))
    print(json.loads((out/'verification.json').read_text()))


if __name__=='__main__':main()
