import numpy as np
import pandas as pd
import data
import pytest


def test_labels_preserve_predetectability_overlap_and_exclusive_end():
    meta = pd.DataFrame({'asset_id':['A']*6+['T'], 'available_utc':pd.to_datetime([
        '2025-01-01T00:01Z','2025-01-01T00:02Z','2025-01-01T00:03Z',
        '2025-01-01T00:04Z','2025-01-01T00:05Z','2025-01-01T00:06Z','2025-01-01T00:03Z']),
        'split':['train']*6+['test']})
    events=pd.DataFrame([
        dict(asset_id='A',event_id='E1',event_class='MACHINE',fault_type='F1',onset_utc='2025-01-01T00:02Z',detectable_utc='2025-01-01T00:03Z',end_utc='2025-01-01T00:05Z',observable=True),
        dict(asset_id='A',event_id='E2',event_class='SENSOR',fault_type='F2',onset_utc='2025-01-01T00:04Z',detectable_utc='2025-01-01T00:04Z',end_utc=None,observable=True)])
    out,y,names=data.annotate(meta,events)
    assert out.label.tolist()==['NO_LABELED_EVENT','PRE_DETECTABLE','F1','MULTIPLE','F2','F2','UNLABELED_TEST']
    assert y[3].tolist()==[1,1]
    assert y[4].tolist()==[0,1]
    assert names==['F1','F2']


def test_missing_inputs_and_frozen_scaling():
    train=pd.DataFrame({'x':[0.,1.,2.,3.],'never':[np.nan]*4})
    p=data.Preprocessor().fit(train)
    before=p.transform(pd.DataFrame({'x':[1.],'never':[8.]}))
    p.transform(pd.DataFrame({'x':[1e12],'never':[8.]}))
    np.testing.assert_array_equal(before,p.transform(pd.DataFrame({'x':[1.],'never':[8.]})))
    x,mask=p.transform(pd.DataFrame({'x':[np.nan,100.],'never':[np.nan,3.]}))
    assert np.isfinite(x).all() and mask.tolist()==[[False],[True]]
    assert p.columns==['x']


def test_sequence_indices_never_cross_pumps_or_look_forward():
    idx=data.sequence_indices(np.array(['A']*3+['B']*3),length=3,step=1)
    assert idx.tolist()==[[0,0,0],[0,0,1],[0,1,2],[3,3,3],[3,3,4],[3,4,5]]


def test_visible_event_does_not_hide_unobservable_overlap():
    meta=pd.DataFrame({'asset_id':['A'],'available_utc':pd.to_datetime(['2025-01-01T00:05Z']),'split':['train']})
    base=dict(asset_id='A',event_class='MACHINE',onset_utc='2025-01-01T00:01Z',end_utc=None)
    events=pd.DataFrame([dict(base,event_id='E1',fault_type='VISIBLE',detectable_utc='2025-01-01T00:02Z',observable=True),
                         dict(base,event_id='E2',fault_type='HIDDEN',detectable_utc=None,observable=False)])
    out,y,names=data.annotate(meta,events)
    assert out.label.iloc[0]=='VISIBLE'
    assert not data.evaluation_eligible(out).iloc[0]
    assert out.active_event_ids.iloc[0]=='E1|E2'


def test_changed_source_invalidates_cache(tmp_path):
    p=tmp_path/'source.txt';p.write_text('old')
    old=data.fingerprint([p])
    p.write_text('changed')
    with pytest.raises(RuntimeError,match='Cached artifacts'):
        data.check_fingerprint(old,data.fingerprint([p]))
