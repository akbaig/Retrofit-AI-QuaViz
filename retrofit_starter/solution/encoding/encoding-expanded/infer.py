"""Apply one saved family encoder/readout to observed data, without reading labels."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from run_expanded import representation,predict_readout
from features import build_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--asset-id',required=True)
    p.add_argument('--readout',default='logistic');p.add_argument('--until')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    model=joblib.load(args.model);assets=pd.read_csv(args.data_dir/'assets.csv')
    rows=assets[assets.asset_id.eq(args.asset_id)]
    if len(rows)!=1:raise ValueError('Exactly one matching asset is required')
    asset=rows.iloc[0];family=asset.get('template_asset_id',asset.asset_id)
    if family!=model['family']:raise ValueError('Use the encoder for this engineering template family')
    meta,frame=build_features(pd.read_parquet(args.data_dir/'features_1min.parquet'),asset,
        pd.read_csv(args.data_dir/'operating_context.csv'),pd.read_parquet(args.data_dir/'vibration_snapshots.parquet'),until=args.until)
    if meta.empty:raise ValueError('No completed minute is available at this cutoff')
    meta['recording']=asset.asset_id
    z=representation(model['encoder'],frame,meta)
    scores=predict_readout(z,model['readouts'][args.readout],len(model['targets']))
    result=meta[['asset_id','available_utc']].copy()
    result['past_warmup']=np.arange(len(result))>=120
    for j,target in enumerate(model['targets']):
        if j not in model['thresholds'][args.readout]:continue
        cut=model['thresholds'][args.readout][j]['value']
        result[f'score__{target}']=scores[:,j]
        result[f'decision__{target}']=scores[:,j]>=cut
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_parquet(args.output,index=False)
    print(f'Wrote {len(result)} minute scores; no event labels read. Scores are not calibrated probabilities or official alert submissions.')


if __name__=='__main__':main()
