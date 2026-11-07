import numpy as np
import pandas as pd
import pytest
import protocol
from label_informed import annotate_labels
from models import sequence_indices


def test_sampling_restarts_and_excludes_warmup_for_each_recording():
    meta = pd.DataFrame({'recording': ['a'] * 130 + ['b'] * 130,
                         'minute_index': list(range(130)) * 2,
                         'role': ['train'] * 260})
    assert protocol.fit_indices(meta).tolist() == [120, 125, 250, 255]
    seq = sequence_indices(meta.recording.to_numpy())
    assert seq[130].tolist() == [130] * 12
    assert seq[140].min() == 130 and seq[140].max() == 140


def test_role_guard_rejects_reused_recording_or_seed():
    with pytest.raises(ValueError, match='recording'):
        protocol.check_roles(pd.DataFrame({'recording': ['x', 'x'], 'role': ['train', 'development'], 'seed': [1, 2]}))
    with pytest.raises(ValueError, match='seed'):
        protocol.check_roles(pd.DataFrame({'recording': ['x', 'y'], 'role': ['train', 'development'], 'seed': [1, 1]}))


def test_ranking_weights_families_equally_and_requires_common_cases():
    rows = pd.DataFrame([
        {'candidate': c, 'family': f, 'recording': r, 'ap': v}
        for c, f, r, v in [('a','p1','r1',1.),('a','p1','r2',1.),('a','p2','r3',0.),
                            ('b','p1','r1',.6),('b','p1','r2',.6),('b','p2','r3',.6),
                            ('a','p3','unshared',1.)]])
    ranked = protocol.rank_common(rows, ['a','b'])
    assert ranked[0]['candidate'] == 'b'
    assert ranked[0]['macro_ap'] == pytest.approx(.6)
    assert ranked[1]['macro_ap'] == pytest.approx(.5)
    assert ranked[0]['recordings'] == 3


def test_labels_exclude_predetectable_and_unobservable_overlaps():
    t = pd.date_range('2025-01-01', periods=6, freq='min', tz='UTC')
    meta = pd.DataFrame({'asset_id': ['x']*6, 'available_utc': t})
    ev = pd.DataFrame([
        dict(asset_id='x', onset_utc=t[1], detectable_utc=t[2], end_utc=t[4], observable=True, fault_type='F'),
        dict(asset_id='x', onset_utc=t[3], detectable_utc=pd.NaT, end_utc=t[4], observable=False, fault_type='G')])
    labels = annotate_labels(meta, ev, True)
    y, names = protocol.make_targets(labels)
    assert names == ['ANY_FAULT','F']
    assert y[:,0].tolist() == [0,-1,1,-1,0,0]


def test_single_class_recordings_are_not_ap_ranked():
    from fault_detection import binary_metrics
    m = binary_metrics(np.zeros(5), np.ones(5)*.8, .5)
    assert m['ap'] is None and m['false_positive_rate'] == 1.


def test_task_encoder_rejects_overlapping_fit_and_selection_rows():
    from task_encoder import fit_task_encoder
    x = np.zeros((20,40), np.float32); mask = np.ones_like(x, bool)
    with pytest.raises(ValueError, match='overlap'):
        fit_task_encoder(x, mask, np.arange(10), np.arange(9,20),
                         np.zeros((10,2)), np.zeros((11,2)), max_epochs=1)


def test_task_encoder_roundtrip_and_32d_output(tmp_path):
    import joblib
    from task_encoder import fit_task_encoder, encode_task
    rng = np.random.default_rng(7); x = rng.normal(size=(120,40)).astype(np.float32)
    mask = np.ones_like(x,bool); y = (x[:,:2]>0).astype(np.float32)
    bundle = fit_task_encoder(x,mask,np.arange(80),np.arange(80,120),y[:80],y[80:],max_epochs=2)
    z = encode_task(bundle,x,mask)
    assert z.shape == (120,32) and np.isfinite(z).all()
    joblib.dump(bundle,tmp_path/'model.joblib')
    np.testing.assert_array_equal(z,encode_task(joblib.load(tmp_path/'model.joblib'),x,mask))
