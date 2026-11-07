import numpy as np
import pandas as pd
import pytest
import data
import evaluate


def test_chronological_split_and_warmup():
    split,eligible=data.chronological_split(10080,warmup=120)
    assert pd.Series(split).value_counts().to_dict()=={'train':6048,'validation':2016,'test':2016}
    assert split[6047]=='train' and split[6048]=='validation' and split[8064]=='test'
    assert not eligible[6048:6168].any() and eligible[6168]
    assert not eligible[8064:8184].any() and eligible[8184]


def test_observed_scaling_is_train_only_and_does_not_clip_targets():
    frame=pd.DataFrame({'a':[0.,2.,4.,6.],'absent':[np.nan]*4,'constant':[7.]*4})
    scaler=data.ObservedScaler().fit(frame)
    x,m=scaler.transform(pd.DataFrame({'a':[np.nan,1000.],'absent':[0.,1.],'constant':[7.,7.]}))
    assert scaler.columns==['a'] and x.shape==(2,1)
    assert m[:,0].tolist()==[False,True] and x[0,0]==0 and x[1,0]>100
    np.testing.assert_allclose(scaler.center,[3.])


def test_reconstruction_metric_balances_features_and_ignores_missing():
    target=np.array([[1.,2.],[3.,99.]])
    predicted=np.zeros((2,2));observed=np.array([[True,True],[True,False]])
    score=evaluate.reconstruction_metrics(target,predicted,observed)
    # Feature a MSE=(1+9)/2=5; feature b MSE=4; macro MSE=4.5.
    assert score['nmse']==4.5 and score['mean_baseline_nmse']==4.5 and score['skill_vs_train_mean']==0
    assert score['valid_cells']==3


def test_heldout_only_feature_not_silently_scored():
    scale=data.ObservedScaler().fit(pd.DataFrame({'seen':[1.,2.,3.],'absent':[np.nan]*3}))
    assert scale.excluded['absent']=='unobserved_or_insufficient_training_samples'


def test_selection_ignores_test_results():
    rows=[{'model':'a','validation_nmse':.1,'test_nmse':100.}, {'model':'b','validation_nmse':.2,'test_nmse':.001}]
    assert evaluate.select_model(rows)=='a'
