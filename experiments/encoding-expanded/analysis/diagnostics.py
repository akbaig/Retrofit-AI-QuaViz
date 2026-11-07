"""Secondary reconstruction and supported-target coverage, without refitting."""
from pathlib import Path
import argparse
import sys
import joblib
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import FAMILIES,load_family,make_targets
from models import encode_reconstruct,sequence_indices,Autoencoder,neural_output
from evaluate import reconstruction_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','test'],required=True)
    args=p.parse_args();cache=ROOT/('.cache/development' if args.split=='development' else '.cache/fresh_test')
    out=ROOT/'results';coverage=[];reconstruction=[]
    for family in FAMILIES:
        meta,frame=load_family(cache,family)
        eligible=meta.minute_index.ge(120)
        eligible &= meta.role.eq('development' if args.split=='development' else 'test')
        # Label eligibility affects classification coverage, not reconstruction of observations.
        use=np.flatnonzero(eligible)
        fitted=joblib.load(out/family/'expanded_pca.joblib')
        y,targets=make_targets(meta.iloc[use])
        known=fitted['targets'];supported=set(fitted['thresholds']['logistic'])
        for j,target in enumerate(targets):
            coverage.append(dict(split=args.split,family=family,target=target,
                positive_minutes=int((y[:,j]==1).sum()),negative_minutes=int((y[:,j]==0).sum()),
                ambiguous_minutes=int((y[:,j]<0).sum()),
                supported=target in known and known.index(target) in supported))
        seq=sequence_indices(meta.recording.to_numpy())
        for path in sorted((out/family).glob('expanded_*.joblib')):
            payload=joblib.load(path);bundle=payload['encoder'];x,mask=bundle['preprocessor'].transform(frame)
            if bundle['kind']=='task_encoder':
                net=Autoencoder(bundle['input_dim'],32);net.load_state_dict(bundle['state_dict'])
                _,pred=neural_output(net,np.clip(x,-10,10).astype(np.float32),mask)
            else:_,pred=encode_reconstruct(bundle,x,mask,seq)
            score=reconstruction_metrics(x[use],pred[use],mask[use])
            reconstruction.append(dict(split=args.split,family=family,variant=payload['variant'],
                nmse=score['nmse'],skill_vs_train_mean=score['skill_vs_train_mean'],rows=len(use),
                dimensions=len(bundle['preprocessor'].columns)))
    pd.DataFrame(coverage).to_csv(out/f'{args.split}_target_coverage.csv',index=False)
    pd.DataFrame(reconstruction).to_csv(out/f'{args.split}_reconstruction.csv',index=False)
    print('Unsupported targets:')
    print(pd.DataFrame(coverage).query('~supported').to_string(index=False))
    print('Reconstruction (same per-family train scaling; task inputs clipped but targets un-clipped):')
    print(pd.DataFrame(reconstruction).groupby('variant')[['nmse','skill_vs_train_mean']].mean().to_string())


if __name__=='__main__':main()
