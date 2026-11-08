"""Run the saved fault model on observations; labels are never read.

Use --until for an explicit live clock, including a terminal gateway outage.
Without it, the clock stops at the latest available observation/context record,
so truncated input cannot create alerts in the unobserved remainder of a week.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import joblib
import pandas as pd
from features import build_features
from alerts import AlertPolicy,make_alerts,COLS


def observed_horizon(features,asset,context=None,snapshots=None):
    candidates=[pd.Timestamp(asset.window_start_utc)]
    f=features[features.asset_id==asset.asset_id]
    if len(f):candidates.append(pd.to_datetime(f.window_start_utc,utc=True).max()+pd.Timedelta(minutes=1))
    if context is not None:
        c=context[context.asset_id==asset.asset_id]
        if len(c):candidates.append(pd.to_datetime(c.timestamp_utc,utc=True).max())
    if snapshots is not None:
        s=snapshots[snapshots.asset_id==asset.asset_id]
        if len(s):candidates.append((pd.to_datetime(s.timestamp_utc,utc=True)+pd.to_timedelta(s.n_samples/s.sample_rate_hz,unit='s')).max())
    return min(max(candidates),pd.Timestamp(asset.window_end_utc))


def infer_asset(model,policy,features,asset,context=None,snapshots=None,until=None):
    horizon=pd.Timestamp(until) if until is not None else observed_horizon(features,asset,context,snapshots)
    if horizon<pd.Timestamp(asset.window_start_utc)+pd.Timedelta(minutes=1):
        return pd.DataFrame(columns=COLS)
    meta,x=build_features(features,asset,context,snapshots,until=horizon)
    transformed=model.transform(x,meta)
    probabilities=model.probabilities_from_transformed(transformed,meta) if getattr(model,'is_composite',False) else model.probabilities_from_transformed(transformed)
    alerts=make_alerts(meta,probabilities,policy,x=transformed)
    if len(alerts):
        t=pd.to_datetime(alerts.alert_time_utc,utc=True)
        alerts=alerts[t.lt(pd.Timestamp(asset.window_end_utc))&t.le(horizon)].reset_index(drop=True)
    return alerts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--policy',required=True)
    p.add_argument('--data',required=True,help='Directory with assets.csv/features_1min.parquet and optional context/snapshots')
    p.add_argument('--output',required=True)
    p.add_argument('--asset-ids',nargs='+')
    p.add_argument('--split',choices=['train','validation','test'])
    p.add_argument('--until',help='UTC time at which observations are available')
    a=p.parse_args();root=Path(a.data)
    assets=pd.read_csv(root/'assets.csv')
    if a.asset_ids:assets=assets[assets.asset_id.isin(a.asset_ids)]
    if a.split:assets=assets[assets.split.eq(a.split)]
    if assets.empty:raise ValueError('No requested assets')
    ids=assets.asset_id.tolist()
    features=pd.read_parquet(root/'features_1min.parquet',filters=[('asset_id','in',ids)])
    cp=root/'operating_context.csv';sp=root/'vibration_snapshots.parquet'
    context=pd.read_csv(cp) if cp.exists() else None
    snapshots=pd.read_parquet(sp,filters=[('asset_id','in',ids)]) if sp.exists() else None
    model=joblib.load(a.model);policy=AlertPolicy(**json.loads(Path(a.policy).read_text()))
    result=pd.concat([infer_asset(model,policy,features,asset,context,snapshots,a.until)
        for _,asset in assets.iterrows()],ignore_index=True)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output,index=False)
    print(f'Wrote {len(result)} alerts for {len(assets)} assets to {a.output}')


if __name__=='__main__':main()
