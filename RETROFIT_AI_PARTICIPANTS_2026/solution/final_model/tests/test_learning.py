import pandas as pd
import numpy as np
from learning import make_targets, FAULTS
from learning import LearnedModel, expert_columns
from test_causality import toy
from features import build_features


def test_overlap_is_multilabel_and_invisible_interval_not_clean():
    t=pd.date_range('2025-01-01',periods=10,freq='min',tz='UTC')
    meta=pd.DataFrame(dict(asset_id='A',available_utc=t))
    events=pd.DataFrame([
        dict(asset_id='A',event_id='1',fault_type='BEARING_WEAR',onset_utc=t[1],detectable_utc=t[3],end_utc=t[8],observable=True),
        dict(asset_id='A',event_id='2',fault_type='SENSOR_DRIFT',onset_utc=t[4],detectable_utc=t[5],end_utc=t[9],observable=True),
        dict(asset_id='A',event_id='3',fault_type='IMPELLER_WEAR',onset_utc=t[2],detectable_utc=None,end_utc=t[4],observable=False)])
    y,weights,healthy=make_targets(meta,events)
    assert y.loc[6,'BEARING_WEAR']==1 and y.loc[6,'SENSOR_DRIFT']==1
    assert y.loc[2,'BEARING_WEAR']==-1
    assert y.loc[2,'IMPELLER_WEAR']==-1
    assert healthy[0] and not healthy[2] and not healthy[6]
    assert set(y.columns)==set(FAULTS)


def test_healthy_regressors_accept_entirely_unavailable_predictors():
    f,a,c=toy(100)
    c['sea_temp_c']=np.nan
    meta,x=build_features(f,a,c)
    model=LearnedModel().fit_reference(x,meta,np.ones(len(meta),bool))
    z=model.transform(x,meta)
    assert 'residual__motor_current_a' in z
    assert np.isfinite(z['residual__motor_current_a']).all()


def test_machine_experts_use_fault_measurements_without_equipment_identity():
    columns=['profile__PROCESS_TRANSFER','control__FIXED_SPEED','rating__motor_rated_kw',
        'age__motor_power_kw','fresh__vibration_rms_mm_s','snapshot__age_min',
        'snapshot__velocity_2x','snapshot__envelope_bpfo_fraction',
        'phys__bearing_thermal_residual_k','phys__head_curve_residual',
        'phys__suction_level_loss_pu','obs__gap_minutes']
    for fault in FAULTS[:8]:
        selected=expert_columns(fault,columns)
        assert not any(c.startswith(('profile__','control__','rating__','age__','fresh__')) for c in selected)
        assert 'snapshot__age_min' not in selected
    assert 'snapshot__velocity_2x' in expert_columns('MISALIGNMENT',columns)
    assert 'phys__head_curve_residual' in expert_columns('IMPELLER_WEAR',columns)
    assert expert_columns('COMMUNICATION_LOSS',columns)==['obs__gap_minutes']
