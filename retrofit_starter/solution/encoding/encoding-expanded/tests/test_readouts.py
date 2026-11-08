import numpy as np
import pandas as pd
from run_expanded import fit_readouts,predict_readout,score_rows


def test_multioutput_knn_keeps_target_order_and_roundtrip_scores():
    rng=np.random.default_rng(19);z=rng.normal(size=(100,32)).astype(np.float32)
    y=np.column_stack([z[:,0]>0,np.zeros(100),z[:,1]>0]).astype(np.int8)
    probes,cuts,scores,warnings=fit_readouts(z,y,np.arange(70),np.arange(70,100))
    for method in ['logistic','knn']:
        restored=predict_readout(z[70:],probes[method],3)
        np.testing.assert_allclose(restored,scores[method],equal_nan=True)
        assert np.isnan(restored[:,1]).all()
        assert set(cuts[method])=={0,2}


def test_all_negative_recording_retains_false_alarms_and_frozen_cutoff():
    meta=pd.DataFrame(dict(recording=['r']*3,family=['p']*3,seed=[1]*3))
    y=np.zeros((3,1),dtype=np.int8);scores=np.array([[.1],[.7],[.8]])
    result=score_rows(meta,y,scores,['ANY_FAULT'],{0:{'value':.6,'origin':'development_max_f1'}},
                      {'candidate':'x'},'fresh_test')[0]
    assert result['ap'] is None and result['fp']==2 and result['tn']==1
    assert result['threshold']==.6 and result['false_positive_rate']==2/3


def test_boosted_tree_readout_roundtrip_preserves_scores():
    rng=np.random.default_rng(21);z=rng.normal(size=(140,32)).astype(np.float32)
    y=np.column_stack([z[:,0]>0,z[:,1]*z[:,2]>0]).astype(np.int8)
    probes,cuts,scores,warnings=fit_readouts(z,y,np.arange(100),np.arange(100,140),raw=True)
    replay=predict_readout(z[100:],probes['histogram'],2)
    np.testing.assert_array_equal(replay,scores['histogram'])
    assert np.isfinite(replay).all() and set(cuts['histogram'])=={0,1}
