"""Causal observations, operating context and past waveform features for ML."""
from __future__ import annotations
import numpy as np
import pandas as pd
from physics_features import add_physics_features, extract_snapshot_features

CHANNELS=['motor_speed_rpm','motor_current_a','motor_power_kw','suction_pressure_kpa',
 'discharge_pressure_kpa','fluid_temp_c','bearing_temp_c','vibration_rms_mm_s',
 'vibration_accel_rms_g','vibration_crest_factor','flow_m3_h','motor_winding_temp_c',
 'valve_position_pct','ambient_temp_c']
SPARSE={'bearing_temp_c','vibration_rms_mm_s','vibration_accel_rms_g','vibration_crest_factor'}
CONTEXT=['duty_demand_pct','supply_water_temp_c','sea_temp_c','ambient_temp_c',
 'sea_state_hs_m','suction_static_head_m']


def run_lengths(mask):
    mask=np.asarray(mask,bool);i=np.arange(len(mask))
    return np.where(mask,i-np.maximum.accumulate(np.where(~mask,i,-1)),0)


def _scale(channel,asset):
    mapping={'motor_speed_rpm':'motor_rated_rpm','motor_current_a':'motor_rated_current_a',
     'motor_power_kw':'motor_rated_kw','flow_m3_h':'design_flow_m3_h'}
    if channel in mapping:
        return max(float(asset[mapping[channel]]),1e-3)
    if 'pressure' in channel:
        return max(float(asset.pump_rated_head_m)*9.81,1)
    if 'temp' in channel:return 30.
    if channel=='valve_position_pct':return 100.
    return 1.


def build_features(features,asset,context=None,snapshots=None,until=None):
    """Return metadata and numeric inputs. Only observations completed by until.

    The reference for inference is a previously fitted training model, so there
    is no local first-day calibration and no future-dependent normalization.
    """
    start,end=pd.Timestamp(asset.window_start_utc),pd.Timestamp(asset.window_end_utc)
    until=min(end,pd.Timestamp(until)) if until is not None else end
    n=max(0,int((until-start).total_seconds()//60))
    grid=pd.date_range(start,periods=n,freq='min')
    f=features[features.asset_id==asset.asset_id].copy()
    f['window_start_utc']=pd.to_datetime(f.window_start_utc,utc=True)
    f=f[(f.window_start_utc>=start)&(f.window_start_utc+pd.Timedelta(minutes=1)<=until)]
    if f.window_start_utc.duplicated().any():
        f=f.drop_duplicates(subset=['window_start_utc'],keep='last')
    f=f.set_index('window_start_utc').reindex(grid).rename_axis('window_start_utc').reset_index()
    f['available_utc']=f.window_start_utc+pd.Timedelta(minutes=1)
    f['asset_id']=asset.asset_id
    if context is not None and len(context):
        ctx=context[context.asset_id==asset.asset_id].copy()
        ctx['timestamp_utc']=pd.to_datetime(ctx.timestamp_utc,utc=True)
        ctx=ctx.drop(columns=['asset_id']).sort_values('timestamp_utc')
        ctx=ctx[ctx.timestamp_utc<=until]
        # Context is observed at its timestamp. Use minute-start context only.
        f=pd.merge_asof(f,ctx,left_on='window_start_utc',right_on='timestamp_utc',
            direction='backward',tolerance=pd.Timedelta(minutes=5),suffixes=('','_ctx'))
    rows=f.get('rows',pd.Series(0,index=f.index)).fillna(0)
    running=f.get('running_fraction',pd.Series(0,index=f.index)).fillna(0)
    steady=run_lengths((f.operating_mode=='RUNNING')&running.ge(.99))>=10
    data={'obs__gap':rows.lt(30).astype(float),'obs__row_fraction':rows/60.,
        'obs__running':running,'obs__steady':steady.astype(float),
        'obs__run_minutes':np.minimum(run_lengths(running.ge(.99)),240),
        'obs__gap_minutes':np.minimum(run_lengths(rows.lt(30)),240)}
    available=set(str(asset.channels_available).split('|'))
    i=np.arange(n)
    for channel in CHANNELS:
        value=f.get(channel+'__mean',pd.Series(np.nan,index=f.index)).astype(float)
        count=f.get(channel+'__count',pd.Series(0,index=f.index)).fillna(0)
        if channel not in available:count=pd.Series(0,index=f.index)
        fresh=value.notna()&count.gt(0)&(channel in available)
        value=value.where(fresh)
        sparse=channel in SPARSE and pd.notna(asset.get('wireless_update_s'))
        cadence=max(1,int(np.ceil(float(asset.wireless_update_s)/60))) if sparse else 1
        last=np.maximum.accumulate(np.where(fresh,i,-1000000))
        age=i-last
        held=value.ffill(limit=cadence) if cadence>1 else value
        held=held.where(age<cadence)
        f[channel+'__mean']=held
        scale=_scale(channel,asset)
        level=held/scale
        std=f.get(channel+'__std',pd.Series(np.nan,index=f.index)).where(fresh&count.ge(2))
        span=(f.get(channel+'__max',pd.Series(np.nan,index=f.index))-
              f.get(channel+'__min',pd.Series(np.nan,index=f.index))).where(fresh&count.ge(2))
        # The physical feature branch must receive the same missingness mask.
        # Stale aggregate statistics are not fresh measured evidence.
        f[channel+'__count']=count.where(fresh,0)
        for stat in ['std','min','max']:
            f[channel+'__'+stat]=f.get(channel+'__'+stat,pd.Series(np.nan,index=f.index)).where(fresh&count.ge(2))
        data['value__'+channel]=held
        data['level__'+channel]=level
        data['fresh__'+channel]=fresh.astype(float)
        data['age__'+channel]=np.minimum(age,240).astype(float)
        data['count__'+channel]=count/60.
        data['std__'+channel]=std/scale
        data['range__'+channel]=span/scale
        data['stdmean__'+channel]=std/(held.abs()+.01*scale)
        for window in [10,60]:
            roll=level.rolling(window,min_periods=2)
            data[f'mean{window}__'+channel]=roll.mean()
            data[f'span{window}__'+channel]=roll.max()-roll.min()
            data[f'delta{window}__'+channel]=level-level.shift(window)
            data[f'stdmean{window}__'+channel]=(std/scale).rolling(window,min_periods=1).mean()
        # Consecutive equal *observed* values, not duplicated held values.
        repeated=np.zeros(n);age_flat=np.zeros(n)
        last_value=np.nan;last_obs=-100000;begin=-1;repeats=0
        vals=value.to_numpy();obs=fresh.to_numpy()
        for j in range(n):
            if j-last_obs>cadence+1 or running.iloc[j]<.99:repeats=0
            if obs[j]:
                same=repeats>0 and abs(vals[j]-last_value)<1e-8
                repeats=repeats+1 if same else 1
                begin=begin if same else j
                last_value,last_obs=vals[j],j
            if repeats>0:
                repeated[j]=min(repeats,240)
                age_flat[j]=min(j-begin,240)
        data['equal_updates__'+channel]=repeated
        data['flat_minutes__'+channel]=age_flat
    for key in CONTEXT:
        data['ctx__'+key]=pd.to_numeric(f.get(key,pd.Series(np.nan,index=f.index)),errors='coerce')
    for control in ['FIXED_SPEED','VFD_TEMPERATURE','VFD_PRESSURE','FLOW_VALVE']:
        data['control__'+control]=np.repeat(float(asset.control==control),n)
    for profile in ['SHIP_SW_COOLING','SHIP_LT_FRESHWATER','INDUSTRIAL_CW','PROCESS_TRANSFER']:
        data['profile__'+profile]=np.repeat(float(asset.profile==profile),n)
    for name in ['motor_rated_rpm','pump_bep_efficiency','npsh3_at_bep_m','static_head_m',
                 'pump_rated_flow_m3_h','pump_rated_head_m','motor_rated_kw']:
        data['rating__'+name]=np.repeat(float(asset.get(name,np.nan)),n)
    physics=add_physics_features(f,asset)
    for name in physics:
        data[name]=physics[name]
        if name in ['phys__speed_command_residual','phys__head_curve_residual',
                    'phys__bearing_thermal_residual_k','phys__winding_thermal_residual_k',
                    'phys__suction_level_loss_pu','phys__flow_required_residual']:
            data['mean60__'+name]=physics[name].rolling(60,min_periods=2).mean()
            data['delta60__'+name]=physics[name]-physics[name].shift(60)
    # Snapshot vectors are derived from measured arrays, never private truth.
    empty_snapshot=extract_snapshot_features([],10240,0,asset)
    shotrows=[]
    if snapshots is not None and len(snapshots):
        s=snapshots[snapshots.asset_id==asset.asset_id].copy()
        s['timestamp_utc']=pd.to_datetime(s.timestamp_utc,utc=True)
        for _,shot in s.iterrows():
            complete=shot.timestamp_utc+pd.Timedelta(seconds=float(shot.n_samples)/float(shot.sample_rate_hz))
            if complete>until:continue
            values=extract_snapshot_features(shot.samples,shot.sample_rate_hz,shot.rpm_at_capture,asset)
            shotrows.append({'shot_available':complete,**values})
    if shotrows:
        sf=pd.DataFrame(shotrows).sort_values('shot_available')
        joined=pd.merge_asof(f[['available_utc']],sf,left_on='available_utc',right_on='shot_available',
            direction='backward',tolerance=pd.Timedelta(hours=6))
        age=(f.available_utc-joined.shot_available).dt.total_seconds()/60.
        # An old pre-stop waveform is not evidence for the restarted machine.
        stopped=f.operating_mode.isin(['STOPPED','LOCKED_OUT','STOPPING'])
        last_stop=np.maximum.accumulate(np.where(stopped,i,-1000000))
        since_stop=i-last_stop
        valid=age.le(since_stop)
        for key in empty_snapshot:
            data[key]=joined[key].where(valid)
        data['snapshot__age_min']=age.where(valid)
    else:
        for key in empty_snapshot:data[key]=np.repeat(np.nan,n)
        data['snapshot__age_min']=np.repeat(np.nan,n)
    out=pd.DataFrame(data).replace([np.inf,-np.inf],np.nan).astype(np.float32)
    # Generic sensor-failure shape features transfer between different tags.
    for prefix in ['equal_updates__','flat_minutes__','stdmean__','stdmean60__']:
        cols=[c for c in out if c.startswith(prefix)]
        out['max__'+prefix]=out[cols].max(axis=1)
    meta=f[['asset_id','window_start_utc','available_utc']].copy()
    meta['steady']=steady;meta['rows']=rows;meta['running_fraction']=running
    return meta,out
