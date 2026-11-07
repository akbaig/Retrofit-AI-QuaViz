import numpy as np
import pandas as pd


def test_undefined_single_class_ranking_is_not_perfect_detection():
    from fault_detection import binary_metrics
    score=binary_metrics(np.ones(4,dtype=int),np.array([.1,.2,.8,.9]),.5)
    assert score['ap'] is None and score['auroc'] is None
    assert score['recall']==.5 and score['false_positive_rate'] is None
    normal=binary_metrics(np.zeros(4,dtype=int),np.array([.1,.2,.8,.9]),.5)
    assert normal['recall'] is None and normal['false_positive_rate']==.5


def test_unknown_annotation_never_becomes_a_negative_target():
    from fault_detection import target_vector,training_targets
    labels=pd.DataFrame({'fault_label':['NO_LABELED_EVENT','BEARING_WEAR|SENSOR_DRIFT','AMBIGUOUS_EVENT','UNLABELED','FUTURE_FAULT'],
                         'label_eligible':[True,True,False,False,True]})
    np.testing.assert_array_equal(target_vector(labels,'ANY_FAULT'),[0,1,-1,-1,1])
    assert training_targets(labels.iloc[:4])==['ANY_FAULT','BEARING_WEAR','SENSOR_DRIFT']


def test_probe_fits_training_rows_only():
    from fault_detection import fit_probe
    rng=np.random.default_rng(4);z=rng.normal(size=(100,32));y=(z[:,0]>0).astype(int);train=np.arange(60)
    for method in ['logistic','knn']:
        first=fit_probe(z,y,train,method);other_z=z.copy();other_y=y.copy();other_z[60:]*=1000;other_y[60:]=1-other_y[60:]
        second=fit_probe(other_z,other_y,train,method)
        np.testing.assert_allclose(first.predict_proba(z),second.predict_proba(z),atol=1e-10)


def test_threshold_is_selected_from_validation_scores():
    from fault_detection import choose_threshold
    assert choose_threshold(np.array([0,0,1,1]),np.array([.1,.6,.4,.9]))==.4
    assert choose_threshold(np.ones(4),np.array([.1,.6,.4,.9])) is None


def test_comparison_uses_identical_cases_for_each_encoder():
    from fault_detection import common_ranking
    rows=pd.DataFrame([
        {'asset_id':'A','target':'ANY_FAULT','model':'one','ap':.6},
        {'asset_id':'A','target':'ANY_FAULT','model':'two','ap':.8},
        {'asset_id':'B','target':'ANY_FAULT','model':'one','ap':1.},
    ])
    ranking,cases=common_ranking(rows,['one','two'])
    assert cases==[('A','ANY_FAULT')]
    assert ranking[0]['model']=='two' and ranking[0]['macro_ap']==.8
    assert ranking[1]['model']=='one' and ranking[1]['macro_ap']==.6


def test_full_run_selection_and_fits_do_not_use_test_data(tmp_path,monkeypatch):
    import json
    import joblib
    from pathlib import Path
    import fault_detection as fd
    monkeypatch.setattr(fd,'ROOT',tmp_path)
    monkeypatch.setattr(fd,'READOUTS',['logistic'])
    (tmp_path/'fault_detection.py').write_text(Path(fd.__file__).read_text())
    source=tmp_path/'results';folder=source/'PUMP-SYN';folder.mkdir(parents=True)
    (source/'labels_manifest.json').write_text('{}')
    pd.DataFrame({'asset_id':['PUMP-SYN']*len(fd.MODELS),'model':fd.MODELS}).to_csv(source/'model_comparison.csv',index=False)
    rng=np.random.default_rng(8);z=rng.normal(size=(100,32));y=(z[:,0]>0).astype(int)
    meta=pd.DataFrame({'asset_id':['PUMP-SYN']*100,'available_utc':pd.date_range('2026-01-01',periods=100,freq='min',tz='UTC'),
                       'split':['train']*60+['validation']*20+['test']*20,'score_eligible':True})
    labels=meta[['asset_id','available_utc']].copy();labels['label_eligible']=True
    labels['fault_label']=np.where(y,'FAULT_A','NO_LABELED_EVENT')
    meta.to_parquet(folder/'metadata.parquet',index=False)
    labels.to_parquet(folder/'label_metadata.parquet',index=False)
    for name in fd.MODELS:np.save(folder/f'{name}_embeddings.npy',z)
    fd.run()
    out=source/'fault_detection';first_selection=json.loads((out/'selection.json').read_text())
    assert first_selection['selected_model'] in fd.MODELS
    assert first_selection['decisions']['any_fault_logistic']['common_cases']==[['PUMP-SYN','ANY_FAULT']]
    first_models={str(p.relative_to(out)):joblib.load(p) for p in out.glob('PUMP-*/*/*.joblib')}
    changed=z.copy();changed[80:]=1000*rng.normal(size=changed[80:].shape)
    for name in fd.MODELS:np.save(folder/f'{name}_embeddings.npy',changed)
    labels.loc[80:,'fault_label']=np.where(1-y[80:],'FAULT_A','NO_LABELED_EVENT')
    labels.to_parquet(folder/'label_metadata.parquet',index=False)
    fd.run()
    assert json.loads((out/'selection.json').read_text())==first_selection
    for path,before in first_models.items():
        after=joblib.load(out/path)
        assert after['threshold']==before['threshold']
        np.testing.assert_array_equal(after['train_indices'],before['train_indices'])
        np.testing.assert_allclose(after['probe'].predict_proba(z),before['probe'].predict_proba(z),atol=1e-12)
