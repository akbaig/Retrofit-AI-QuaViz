"""Apply a saved encoder to a compatible public-format dataset, without refitting."""
import argparse,joblib
from pathlib import Path
import pandas as pd
import numpy as np
from data import build_features,sequence_indices,ROOT
from models import transform_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True,help='Directory containing assets.csv, features_1min.parquet and optional context/snapshots')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--asset-id',help='Optional single asset to encode')
    parser.add_argument('--until',help='Optional UTC cutoff; excludes unfinished minutes')
    args=parser.parse_args();bundle=joblib.load(args.model)
    assets=pd.read_csv(args.data_dir/'assets.csv');features=pd.read_parquet(args.data_dir/'features_1min.parquet')
    context=pd.read_csv(args.data_dir/'operating_context.csv') if (args.data_dir/'operating_context.csv').exists() else None
    snapshots=pd.read_parquet(args.data_dir/'vibration_snapshots.parquet') if (args.data_dir/'vibration_snapshots.parquet').exists() else None
    if args.asset_id:assets=assets[assets.asset_id.eq(args.asset_id)]
    if assets.empty:raise ValueError('No matching assets')
    outputs=[]
    for _,asset in assets.iterrows():
        meta,frame=build_features(features,asset,context,snapshots,until=args.until)
        if not len(meta):continue
        x,mask=bundle['preprocessor'].transform(frame)
        z=transform_model(bundle,x,mask,sequence_indices(meta.asset_id))
        for i in range(16):meta[f'z{i:02d}']=z[:,i]
        outputs.append(meta)
    if not outputs:raise ValueError('No completed minutes before cutoff')
    output=pd.concat(outputs,ignore_index=True);assert np.isfinite(output[[f'z{i:02d}' for i in range(16)]]).all().all()
    args.output.parent.mkdir(parents=True,exist_ok=True);output.to_parquet(args.output,index=False)
    print(f'Encoded {len(output):,} completed minutes to {args.output}')

if __name__=='__main__':main()
