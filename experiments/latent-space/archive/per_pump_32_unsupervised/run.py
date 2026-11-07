"""Train all five 32D models independently per pump; select on validation only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,json,joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from data import ROOT,prepare,ObservedScaler,fingerprint,sha256
from models import KINDS,fit_encoder,encode_reconstruct,sequence_indices
from evaluate import reconstruction_metrics,select_model

def serializable(metrics):return {k:v for k,v in metrics.items() if k!='row_nmse'}

def aggregate_results(out):
    """Rebuild global exports from every complete pump, including partial reruns."""
    comparisons=[];selected=[];embeddings=[]
    for dest in sorted(out.glob('PUMP-*')):
        required=['comparison.csv','selection.json','selected_embeddings.parquet']
        if not all((dest/name).exists() for name in required):continue
        rows=pd.read_csv(dest/'comparison.csv')
        winner=json.loads((dest/'selection.json').read_text())['selected_model']
        comparisons.append(rows);selected.append(rows[rows.model.eq(winner)])
        embeddings.append(pd.read_parquet(dest/'selected_embeddings.parquet').assign(selected_model=winner))
    if not comparisons:return
    pd.concat(comparisons,ignore_index=True).to_csv(out/'model_comparison.csv',index=False)
    pd.concat(selected,ignore_index=True).to_csv(out/'selected_models.csv',index=False)
    combined=pd.concat(embeddings,ignore_index=True)
    combined.to_parquet(out/'selected_embeddings.parquet',index=False)
    np.save(out/'selected_embeddings.npy',combined[[f'z{i:02d}' for i in range(32)]].to_numpy(dtype=np.float32))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pumps',nargs='+');args=parser.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True);meta,frame=prepare(out)
    for asset,group in meta.groupby('asset_id',sort=True):
        if args.pumps and asset not in args.pumps:continue
        indices=group.index.to_numpy();m=group.reset_index(drop=True);f=frame.loc[indices].reset_index(drop=True)
        dest=out/asset;dest.mkdir(exist_ok=True)
        train=np.flatnonzero(m.split.eq('train'));validation=np.flatnonzero(m.split.eq('validation')&m.score_eligible);test=np.flatnonzero(m.split.eq('test')&m.score_eligible)
        pre=ObservedScaler().fit(f.iloc[train]);x,mask=pre.transform(f);seq=sequence_indices(m.asset_id)
        m.to_parquet(dest/'metadata.parquet',index=False);joblib.dump(pre,dest/'preprocessor.joblib')
        (dest/'features.json').write_text(json.dumps({'included':pre.columns,'excluded':pre.excluded},indent=2))
        m.groupby('split').agg(rows=('split','size'),scored_rows=('score_eligible','sum'),start=('available_utc','min'),end=('available_utc','max')).to_csv(dest/'splits.csv')
        print(asset,'inputs',x.shape[1],'train',len(train),'val',len(validation),'test',len(test),flush=True)
        rows=[];outputs={}
        for kind in KINDS:
            signature=fingerprint([ROOT/'models.py',ROOT/'evaluate.py',out/'preparation.json'],{'asset':asset,'kind':kind,'dimensions':32,'epochs':50,'seed':42})
            path=dest/f'{kind}.joblib'
            if path.exists():
                bundle=joblib.load(path)
                if bundle.get('fingerprint')!=signature:raise RuntimeError(f'Stale model cache: {path}')
            else:
                bundle=fit_encoder(kind,x,mask,train,validation,32,seq)
                bundle.update(fingerprint=signature,asset_id=asset,preprocessor=pre,train_rows=len(train),fit_last_available_utc=str(m.available_utc.iloc[train[-1]]),uses_labels=False)
                joblib.dump(bundle,path)
            z,recon=encode_reconstruct(bundle,x,mask,seq)
            assert z.shape==(len(m),32) and np.isfinite(z).all() and np.isfinite(recon).all()
            tr=reconstruction_metrics(x[train],recon[train],mask[train]);va=reconstruction_metrics(x[validation],recon[validation],mask[validation])
            row={'asset_id':asset,'model':kind,'input_features':len(pre.columns),'dimensions':32,'effective_dimensions':bundle['effective_dimensions'],
                 'train_nmse':tr['nmse'],'validation_nmse':va['nmse'],'validation_skill':va['skill_vs_train_mean'],
                 'validation_mean_baseline':va['mean_baseline_nmse'],'best_epoch':bundle.get('best_epoch'),'fit_seconds':bundle['fit_seconds']}
            rows.append(row);outputs[kind]=(z,recon)
            (dest/f'{kind}_training.json').write_text(json.dumps(bundle['history'],indent=2))
            print(asset,kind,'validation NMSE',round(va['nmse'],5),'epoch',bundle.get('best_epoch'),flush=True)
        winner=select_model(rows)
        selection={'asset_id':asset,'selected_model':winner,'criterion':'lowest validation feature-balanced observed NMSE','dimensions':32,
                   'uses_labels':False,'test_used_for_selection':False,'model_sha256':sha256(dest/f'{winner}.joblib')}
        (dest/'selection.json').write_text(json.dumps(selection,indent=2))
        for row in rows:
            kind=row['model'];z,recon=outputs[kind];te=reconstruction_metrics(x[test],recon[test],mask[test]);row.update(test_nmse=te['nmse'],test_skill=te['skill_vs_train_mean'],test_mean_baseline=te['mean_baseline_nmse'])
            full=reconstruction_metrics(x,recon,mask)
            clustering=KMeans(n_clusters=8,n_init=10,random_state=42).fit(z[train]);clusters=clustering.predict(z)
            joblib.dump(clustering,dest/f'{kind}_kmeans.joblib')
            export=m.copy();export['cluster']=clusters;export['reconstruction_error']=full['row_nmse']
            for j in range(32):export[f'z{j:02d}']=z[:,j]
            export.to_parquet(dest/f'{kind}_embeddings.parquet',index=False);np.save(dest/f'{kind}_embeddings.npy',z)
            scores={s:serializable(reconstruction_metrics(x[idx],recon[idx],mask[idx])) for s,idx in [('train',train),('validation',validation),('test',test)]}
            (dest/f'{kind}_metrics.json').write_text(json.dumps(scores,indent=2))
            if kind==winner:
                export.to_parquet(dest/'selected_embeddings.parquet',index=False);np.save(dest/'selected_embeddings.npy',z)
                for split in ['train','validation','test']:export[export.split.eq(split)].to_parquet(dest/f'{split}.parquet',index=False)
        pd.DataFrame(rows).to_csv(dest/'comparison.csv',index=False)
        print('SELECTED',asset,winner,'test NMSE',next(r['test_nmse'] for r in rows if r['model']==winner),flush=True)
    aggregate_results(out)
    print('Completed per-pump experiment',flush=True)
if __name__=='__main__':main()
