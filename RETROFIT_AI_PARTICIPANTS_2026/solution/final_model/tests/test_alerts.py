import pandas as pd
import numpy as np
from alerts import make_alerts, AlertPolicy
from learning import FAULTS


def test_persistent_incident_rearms_after_recovery_without_repeat_spam():
    t=pd.date_range('2025-01-01',periods=200,freq='min',tz='UTC')
    meta=pd.DataFrame(dict(asset_id='A',available_utc=t,steady=True,rows=60,running_fraction=1.))
    p=pd.DataFrame(0.,index=range(200),columns=FAULTS)
    p.loc[10:40,'CAVITATION']=.9;p.loc[100:120,'CAVITATION']=.9
    out=make_alerts(meta,p,AlertPolicy(threshold=.7,persistence=5,recovery=20))
    assert list(out.fault_type)==['CAVITATION','CAVITATION']
    assert list(pd.to_datetime(out.alert_time_utc,utc=True))==[t[14],t[104]]


def test_simultaneous_distinct_diagnoses_do_not_merge_into_scoring_burst():
    t=pd.date_range('2025-01-01',periods=60,freq='min',tz='UTC')
    meta=pd.DataFrame(dict(asset_id='A',available_utc=t,steady=True,rows=60,running_fraction=1.))
    p=pd.DataFrame(0.,index=range(60),columns=FAULTS)
    p.loc[10:40,['BEARING_WEAR','SENSOR_DRIFT']]=.9
    out=make_alerts(meta,p,AlertPolicy(threshold=.7,persistence=5,recovery=20))
    assert set(out.fault_type)=={'BEARING_WEAR','SENSOR_DRIFT'}
    assert pd.to_datetime(out.alert_time_utc,utc=True).diff().dropna().ge(pd.Timedelta(minutes=6)).all()
