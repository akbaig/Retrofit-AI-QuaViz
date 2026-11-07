"""Bounded development-only improvement: boosted-tree probes for two leading encoders."""
from pathlib import Path
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import FAMILIES,fit_indices,make_targets,load_family,rank_common,dump_json
from models import KINDS
from run_expanded import representation,fit_readouts,score_rows,save_predictions,aggregate_development
from integrity import seal_prepared_data,verify_data_seal


def main():
    out=ROOT/'results';cache=ROOT/'.cache/development'
    if (out/'frozen.json').exists():raise ValueError('No extension after freeze')
    seal_prepared_data(cache,out/'data_seal.json')
    verify_data_seal(out/'data_seal.json')
    selection=json.loads((out/'selection.json').read_text())
    dev=pd.read_csv(out/'development_metrics.csv');any_rows=dev[dev.target.eq('ANY_FAULT')]
    leaders=[r['candidate'].split('/')[0] for r in selection['rankings']['logistic'][:2]]
    reference=rank_common(any_rows,[f'frozen_{k}/logistic' for k in KINDS])[0]['candidate'].split('/')[0]
    variants=leaders+[reference]
    dump_json(out/'head_extension_plan.json',dict(variants=variants,rule='top two expanded encoders and best original frozen reference on development logistic AP',
        settings=dict(max_iter=150,max_leaf_nodes=15,learning_rate=.08,l2_regularization=3,min_samples_leaf=30),test_data_used=False))
    for family in FAMILIES:
        meta,frame=load_family(cache,family);indices=fit_indices(meta)
        train=indices[meta.label_eligible.to_numpy()[indices]]
        val=np.flatnonzero(meta.role.eq('development')&meta.minute_index.ge(120)&meta.label_eligible)
        for variant in variants:
            dest=out/family;path=dest/f'{variant}.joblib';payload=joblib.load(path)
            z=representation(payload['encoder'],frame,meta);y,_=make_targets(meta,payload['targets'])
            started=time.monotonic();print('HEAD',family,variant,flush=True)
            probes,cuts,scores,warnings=fit_readouts(z,y,train,val,raw=True)
            # Keep original readouts unchanged; only add the fixed boosted-tree classifier.
            payload['readouts']['histogram']=probes['histogram'];payload['thresholds']['histogram']=cuts['histogram']
            joblib.dump(payload,path,compress=3)
            identity=dict(variant=variant,readout='histogram',candidate=f'{variant}/histogram')
            metrics=score_rows(meta.iloc[val].reset_index(drop=True),y[val],scores['histogram'],payload['targets'],cuts['histogram'],identity,'development')
            dump_json(dest/f'{variant}_histogram_metrics.json',dict(metrics=metrics,seconds=time.monotonic()-started,warnings=warnings))
            save_predictions(dest/f'{variant}_histogram_development.parquet',meta.iloc[val],y[val],scores['histogram'],payload['targets'])
    aggregate_development(out)
    verify_data_seal(out/'data_seal.json')
    dev=pd.read_csv(out/'development_metrics.csv')
    eligible=[selection['selected_candidate'],*[f'{v}/histogram' for v in leaders]]
    ranking=rank_common(dev[dev.target.eq('ANY_FAULT')],eligible)
    dump_json(out/'head_selection.json',dict(selected_candidate=ranking[0]['candidate'],rankings=ranking,
        added_candidates=[f'{v}/histogram' for v in variants],original_logistic_selection=selection['selected_candidate'],test_data_used=False))
    print(ranking,flush=True)


if __name__=='__main__':main()
