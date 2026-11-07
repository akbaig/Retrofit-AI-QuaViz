import numpy as np
import pandas as pd
import torch


def fixture():
    meta=pd.DataFrame({'asset_id':['PUMP-01']*8,'available_utc':pd.date_range('2025-01-01',periods=8,freq='min',tz='UTC'),'split':['train']*5+['validation','test','test']})
    events=pd.DataFrame([
        {'asset_id':'PUMP-01','fault_type':'BEARING_WEAR','onset_utc':'2025-01-01T00:01:00Z','detectable_utc':'2025-01-01T00:02:00Z','end_utc':'2025-01-01T00:04:00Z','observable':True},
        {'asset_id':'PUMP-01','fault_type':'SENSOR_DRIFT','onset_utc':'2025-01-01T00:03:00Z','detectable_utc':None,'end_utc':'2025-01-01T00:04:00Z','observable':False},
        {'asset_id':'PUMP-01','fault_type':'FUTURE_FAULT','onset_utc':'2025-01-01T00:06:00Z','detectable_utc':'2025-01-01T00:06:00Z','end_utc':None,'observable':True}])
    return meta,events


def test_training_supervision_excludes_future_and_ambiguous_overlaps():
    from label_informed import training_supervision
    meta,events=fixture();idx,states=training_supervision(meta,events,True)
    assert idx.tolist()==[0,2,4]
    assert states.tolist()==['NO_LABELED_EVENT','BEARING_WEAR','NO_LABELED_EVENT']
    changed=events.copy();changed.loc[2,'fault_type']='OTHER_FUTURE_FAULT'
    j,s=training_supervision(meta,changed,True)
    np.testing.assert_array_equal(j,idx);np.testing.assert_array_equal(s,states)


def test_unlabeled_pump_never_becomes_a_healthy_training_group():
    from label_informed import training_supervision,annotate_labels
    meta,events=fixture();idx,states=training_supervision(meta,events,False)
    assert len(idx)==0 and len(states)==0
    labels=annotate_labels(meta,events,False)
    assert not labels.label_eligible.any() and set(labels.fault_label)=={'UNLABELED'}


def test_contrastive_loss_depends_on_label_pairing():
    from label_informed import contrastive_loss
    z=torch.tensor([[1.,0.],[.99,.01],[-1.,0.],[-.99,-.01]],requires_grad=True)
    good=contrastive_loss(z,torch.tensor([0,0,1,1]));bad=contrastive_loss(z,torch.tensor([0,1,0,1]))
    assert good.item()<bad.item();good.backward();assert torch.isfinite(z.grad).all()


def test_label_informed_model_has_32d_label_free_inference():
    from label_informed import fit_label_informed
    from models import encode_reconstruct
    rng=np.random.default_rng(7);x=rng.normal(size=(96,40)).astype('float32');mask=np.ones_like(x,bool)
    train=np.arange(64);validation=np.arange(64,80);idx=np.arange(64);states=np.array(['a']*32+['b']*32)
    bundle=fit_label_informed(x,mask,train,validation,idx,states,max_epochs=2)
    z,recon=encode_reconstruct(bundle,x,mask)
    assert z.shape==(96,32) and recon.shape==x.shape and bundle['uses_labels'] is True
    changed=x.copy();changed[80:]*=1000
    other=fit_label_informed(changed,mask,train,validation,idx,states,max_epochs=2)
    oz,_=encode_reconstruct(other,x,mask)
    np.testing.assert_allclose(z,oz,rtol=1e-6,atol=1e-6)
    import pytest
    with pytest.raises(ValueError,match='training'):
        fit_label_informed(x,mask,train,validation,np.array([64,65]),np.array(['a','b']),max_epochs=1)
