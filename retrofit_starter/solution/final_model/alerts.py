"""Causal incident-state alerts from learned probabilities; no labels available."""
from dataclasses import dataclass,field
import numpy as np
import pandas as pd
from learning import FAULTS,SENSORS

ACTIONS={
 'BEARING_WEAR':'Inspect bearing lubrication and condition; confirm with vibration measurements.',
 'SUCTION_STRAINER_CLOGGING':'Inspect suction strainer and suction-side pressure loss.',
 'CAVITATION':'Check suction pressure, liquid level and strainer; confirm cavitation.',
 'DISCHARGE_RESTRICTION':'Inspect discharge valves and downstream restriction.',
 'MISALIGNMENT':'Check coupling alignment and confirm with vibration measurements.',
 'IMBALANCE':'Inspect rotating assembly for fouling, damage or imbalance.',
 'IMPELLER_WEAR':'Check hydraulic performance and inspect impeller or wear rings.',
 'AIR_ENTRAINMENT':'Check for air ingress and unstable suction or delivered flow.',
 'SENSOR_DRIFT':'Compare the indicated instrument with a reference; recalibrate if confirmed.',
 'SENSOR_STUCK':'Check frozen readings, transmitter and wiring against an independent measurement.',
 'SENSOR_CALIBRATION_SHIFT':'Check calibration and recent instrument configuration changes.',
 'SENSOR_NOISE_SPIKES':'Check instrument wiring, signal quality and a reference measurement.',
 'COMMUNICATION_LOSS':'Check gateway power, network and data acquisition.'}
COLS=['asset_id','alert_time_utc','event_class','fault_type','confidence','evidence','recommended_action']


@dataclass(frozen=True)
class AlertPolicy:
    threshold:float=.8
    persistence:int=10
    recovery:int=60
    reminder:int=1440
    thresholds:dict=field(default_factory=dict)


def make_alerts(meta,probabilities,policy,x=None):
    rows=[]
    for aid,ids in meta.groupby('asset_id',sort=False).indices.items():
        probs=probabilities.iloc[ids].reindex(columns=FAULTS).to_numpy()
        # Suppress nongateway decisions in stopped/startup periods, when the PLC
        # overwrites several readings. Gateway detection uses the clock alone.
        steady=meta.iloc[ids].steady.to_numpy()
        threshold=np.array([policy.thresholds.get(f,policy.threshold) for f in FAULTS])
        above=np.zeros(len(FAULTS),int);below=np.zeros(len(FAULTS),int)
        active=np.zeros(len(FAULTS),bool);last=np.full(len(FAULTS),-100000);last_global=-100000
        for j,idx in enumerate(ids):
            p=probs[j].copy()
            if not steady[j]:
                for c,ft in enumerate(FAULTS):
                    if ft!='COMMUNICATION_LOSS':p[c]=0.
            flag=p>=threshold
            above=np.where(flag,above+1,0)
            low=p<(threshold*.6)
            below=np.where(low,below+1,0)
            active[below>=policy.recovery]=False
            ready=(above>=policy.persistence)&(~active|((j-last)>=policy.reminder))
            if j-last_global<6 or not ready.any():continue
            # Highest relative confidence wins this minute. Other independently
            # supported diagnoses remain eligible after the burst guard expires.
            c=int(np.argmax(np.where(ready,p/threshold,-np.inf)))
            ft=FAULTS[c]
            evidence=f'learned {ft} score={p[c]:.3f}; sustained {int(above[c])} min'
            if x is not None:
                residuals=x.iloc[idx].filter(like='residual__').dropna()
                strongest=residuals.abs().nlargest(3).index
                if len(strongest):evidence+='; '+', '.join(f'{name}={residuals[name]:+.2f}' for name in strongest)
            rows.append(dict(asset_id=aid,alert_time_utc=meta.available_utc.iloc[idx].isoformat(),
                event_class='SENSOR' if ft in SENSORS else 'MACHINE',fault_type=ft,
                confidence=float(p[c]),evidence=evidence,recommended_action=ACTIONS[ft]))
            active[c]=True;last[c]=last_global=j
    return pd.DataFrame(rows,columns=COLS)
