import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from features import build_features, CHANNELS


def toy(n=360):
    start = pd.Timestamp('2025-01-01', tz='UTC')
    a = pd.Series(dict(asset_id='TOY', window_start_utc=start,
        window_end_utc=start+pd.Timedelta(minutes=n), motor_rated_rpm=1500,
        motor_rated_kw=30., motor_rated_current_a=55., pump_rated_flow_m3_h=100.,
        pump_rated_head_m=30., pump_shutoff_head_m=36., pump_bep_efficiency=.8,
        design_flow_m3_h=95., design_head_m=28., static_head_m=5.,
        npsh3_at_bep_m=3., min_speed_ratio=.5, control='VFD_PRESSURE',
        profile='INDUSTRIAL_CW', instrumentation_tier='T1_RETROFIT',
        bearing_bpfo_order=3.05, bearing_bpfi_order=4.95, impeller_vanes=6,
        channels_available='|'.join(CHANNELS), wireless_update_s=300,
        vibration_alarm_mm_s=6.4, fluid='Cooling water'))
    f = pd.DataFrame(dict(asset_id='TOY',window_start_utc=pd.date_range(start,periods=n,freq='min'),
        operating_mode='RUNNING',running_fraction=1.,quality_any=0,rows=60))
    for c in CHANNELS:
        f[c+'__mean'] = 10+np.sin(np.arange(n)/10)*.1
        f[c+'__count'] = 60
        f[c+'__std'] = .1
        f[c+'__min'] = f[c+'__mean']-.2
        f[c+'__max'] = f[c+'__mean']+.2
    f['motor_speed_rpm__mean'] = 1500.
    context = pd.DataFrame(dict(asset_id='TOY',timestamp_utc=f.window_start_utc,
        duty_demand_pct=70.,supply_water_temp_c=25.,ambient_temp_c=20.,
        suction_static_head_m=2.,controlled_setpoint=30.))
    return f,a,context


def test_completed_window_prefix_and_future_mutation():
    f,a,c=toy()
    cutoff=f.window_start_utc.iloc[180]+pd.Timedelta(seconds=30)
    meta,x=build_features(f,a,c,until=cutoff)
    assert meta.available_utc.max()<=cutoff
    mf,xf=build_features(f,a,c)
    assert_frame_equal(x,xf[mf.available_utc<=cutoff].reset_index(drop=True))
    f.loc[f.window_start_utc>=cutoff,'motor_current_a__mean']=1e6
    c.loc[c.timestamp_utc>=cutoff,'duty_demand_pct']=1e6
    mm,xx=build_features(f,a,c)
    assert_frame_equal(x,xx[mm.available_utc<=cutoff].reset_index(drop=True))


def test_sparse_freshness_and_terminal_grid():
    f,a,c=toy()
    name='bearing_temp_c'
    sparse=np.arange(len(f))%5==0
    f.loc[~sparse,name+'__mean']=np.nan
    f.loc[~sparse,name+'__count']=0
    f.loc[sparse,name+'__count']=1
    f[name+'__std']=np.nan
    m,x=build_features(f.iloc[:300],a,c)
    assert len(m)==360
    assert x.loc[300:,'obs__gap'].eq(1).all()
    assert x.loc[100:104,'fresh__bearing_temp_c'].sum()==1
    assert x.loc[104,'age__bearing_temp_c']==4
    assert np.isnan(x.loc[310,'value__bearing_temp_c'])


def test_no_identifiers_or_labels_in_inputs():
    f,a,c=toy()
    f['fault_type']='FAULT';f['latent_bearing_wear']=100
    m,x=build_features(f,a,c)
    assert all(np.issubdtype(t,np.number) for t in x.dtypes)
    assert not any(s in col for col in x for s in ['asset_id','event_id','window_start','timestamp','latent','fault_type','deadline','seed','template'])
