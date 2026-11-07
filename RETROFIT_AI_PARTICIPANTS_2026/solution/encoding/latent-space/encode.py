"""Apply a saved pump-specific encoder without labels or refitting."""
import argparse,joblib
from pathlib import Path
import pandas as pd
import numpy as np
from data import build_features
from models import encode_reconstruct,sequence_indices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True,help='Public-format assets.csv, features_1min.parquet, optional context/snapshots')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--asset-id',help='Must match the pump used to fit the model; defaults to that pump')
    parser.add_argument('--until',help='UTC availability cutoff; excludes unfinished minutes')
    args=parser.parse_args();bundle=joblib.load(args.model)
    asset_id=args.asset_id or bundle['asset_id']
    if asset_id!=bundle['asset_id']:raise ValueError(f"This encoder belongs to {bundle['asset_id']}; choose that pump's own model")
    assets=pd.read_csv(args.data_dir/'assets.csv');assets=assets[assets.asset_id.eq(asset_id)]
    if len(assets)!=1:raise ValueError('Expected exactly one matching asset')
    features=pd.read_parquet(args.data_dir/'features_1min.parquet')
    context=pd.read_csv(args.data_dir/'operating_context.csv') if (args.data_dir/'operating_context.csv').exists() else None
    snapshots=pd.read_parquet(args.data_dir/'vibration_snapshots.parquet') if (args.data_dir/'vibration_snapshots.parquet').exists() else None
    meta,frame=build_features(features,assets.iloc[0],context,snapshots,until=args.until)
    if not len(meta):raise ValueError('No completed minutes before cutoff')
    x,mask=bundle['preprocessor'].transform(frame);z,_=encode_reconstruct(bundle,x,mask,sequence_indices(meta.asset_id))
    assert z.shape==(len(meta),32) and np.isfinite(z).all()
    for j in range(32):meta[f'z{j:02d}']=z[:,j]
    args.output.parent.mkdir(parents=True,exist_ok=True);meta.to_parquet(args.output,index=False)
    print(f'Encoded {len(meta):,} completed minutes for {asset_id} to {args.output}')
if __name__=='__main__':main()
