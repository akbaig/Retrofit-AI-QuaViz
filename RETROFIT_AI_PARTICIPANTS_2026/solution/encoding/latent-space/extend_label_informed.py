"""Add label-informed candidates without refitting the five saved baselines."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,json,joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from data import ROOT,DATA,prepare,fingerprint,sha256
from models import encode_reconstruct
from label_informed import LABEL_KIND,LABEL_FILES,training_supervision,annotate_labels,fit_label_informed
from evaluate import reconstruction_metrics,select_model
from run import aggregate_results,serializable


def extend(out=None,pumps=None):
    out=out or ROOT/'results';meta,frame=prepare(out)
    events=pd.concat([pd.read_csv(DATA/name) for name in LABEL_FILES],ignore_index=True)
    assets=pd.read_csv(DATA/'assets.csv');covered=set(assets.loc[assets.split.isin(['train','validation']),'asset_id'])
    for asset,group in meta.groupby('asset_id',sort=True):
        if pumps and asset not in pumps:continue
        dest=out/asset;m=group.reset_index(drop=True);f=frame.loc[group.index].reset_index(drop=True)
        own=events[events.asset_id.eq(asset)]
        idx,states=training_supervision(m,own,asset in covered)
        available=asset in covered and len(np.unique(states))>=2
        reason='' if available else 'No public event labels for this pump' if asset not in covered else 'Fewer than two unambiguous training states'
        status={'asset_id':asset,'model':LABEL_KIND,'available':available,'reason':reason,'supervised_rows':len(idx),'training_states':dict(zip(*[v.tolist() for v in np.unique(states,return_counts=True)]))}
        (dest/'label_availability.json').write_text(json.dumps(status,indent=2))
        rows=pd.read_csv(dest/'comparison.csv');rows=rows[rows.model.ne(LABEL_KIND)].copy();rows['uses_training_labels']=False
        if not available:
            print(asset,'label-informed unavailable:',reason,flush=True)
            winner=select_model(rows.to_dict('records'))
            selection={'asset_id':asset,'selected_model':winner,'criterion':'lowest validation feature-balanced observed NMSE across available models','dimensions':32,
                       'uses_labels':False,'test_used_for_selection':False,'model_sha256':sha256(dest/f'{winner}.joblib')}
            (dest/'selection.json').write_text(json.dumps(selection,indent=2))
            selected=pd.read_parquet(dest/f'{winner}_embeddings.parquet');selected.to_parquet(dest/'selected_embeddings.parquet',index=False)
            np.save(dest/'selected_embeddings.npy',np.load(dest/f'{winner}_embeddings.npy'))
            for split in ['train','validation','test']:selected[selected.split.eq(split)].to_parquet(dest/f'{split}.parquet',index=False)
            rows.to_csv(dest/'comparison.csv',index=False)
            annotate_labels(m,own,asset in covered).to_parquet(dest/'label_metadata.parquet',index=False)
            continue
        train=np.flatnonzero(m.split.eq('train'));validation=np.flatnonzero(m.split.eq('validation')&m.score_eligible);test=np.flatnonzero(m.split.eq('test')&m.score_eligible)
        pre=joblib.load(dest/'preprocessor.joblib');x,mask=pre.transform(f)
        signature=fingerprint([ROOT/'label_informed.py',ROOT/'models.py',ROOT/'evaluate.py',out/'preparation.json',dest/'metadata.parquet',*[DATA/name for name in LABEL_FILES]],{'seed':42,'dimensions':32,'max_epochs':50,'weight':.1,'asset_id':asset})
        path=dest/f'{LABEL_KIND}.joblib'
        if path.exists():
            bundle=joblib.load(path)
            if bundle.get('fingerprint')!=signature:raise RuntimeError(f'Stale label-informed cache: {path}')
        else:
            bundle=fit_label_informed(x,mask,train,validation,idx,states)
            bundle.update(fingerprint=signature,asset_id=asset,preprocessor=pre,train_rows=len(train),fit_last_available_utc=str(m.available_utc.iloc[train[-1]]))
            joblib.dump(bundle,path)
        z,recon=encode_reconstruct(bundle,x,mask)
        assert z.shape==(len(m),32) and np.isfinite(z).all()
        tr=reconstruction_metrics(x[train],recon[train],mask[train]);va=reconstruction_metrics(x[validation],recon[validation],mask[validation])
        row={'asset_id':asset,'model':LABEL_KIND,'input_features':len(pre.columns),'dimensions':32,'effective_dimensions':bundle['effective_dimensions'],
             'train_nmse':tr['nmse'],'validation_nmse':va['nmse'],'validation_skill':va['skill_vs_train_mean'],'validation_mean_baseline':va['mean_baseline_nmse'],
             'best_epoch':bundle['best_epoch'],'fit_seconds':bundle['fit_seconds'],'uses_training_labels':True}
        winner=select_model([*rows.to_dict('records'),row])
        selection={'asset_id':asset,'selected_model':winner,'criterion':'lowest validation feature-balanced observed NMSE across available models','dimensions':32,
                   'uses_labels':winner==LABEL_KIND,'test_used_for_selection':False,'model_sha256':sha256(dest/f'{winner}.joblib')}
        (dest/'selection.json').write_text(json.dumps(selection,indent=2))
        te=reconstruction_metrics(x[test],recon[test],mask[test]);row.update(test_nmse=te['nmse'],test_skill=te['skill_vs_train_mean'],test_mean_baseline=te['mean_baseline_nmse'])
        km=KMeans(n_clusters=8,n_init=10,random_state=42).fit(z[train]);joblib.dump(km,dest/f'{LABEL_KIND}_kmeans.joblib')
        export=m.copy();export['cluster']=km.predict(z);export['reconstruction_error']=reconstruction_metrics(x,recon,mask)['row_nmse']
        for j in range(32):export[f'z{j:02d}']=z[:,j]
        np.save(dest/f'{LABEL_KIND}_embeddings.npy',z);export.to_parquet(dest/f'{LABEL_KIND}_embeddings.parquet',index=False)
        for split in ['train','validation','test']:export[export.split.eq(split)].to_parquet(dest/f'{LABEL_KIND}_{split}.parquet',index=False)
        scores={s:serializable(reconstruction_metrics(x[ii],recon[ii],mask[ii])) for s,ii in [('train',train),('validation',validation),('test',test)]}
        (dest/f'{LABEL_KIND}_metrics.json').write_text(json.dumps(scores,indent=2))
        (dest/f'{LABEL_KIND}_training.json').write_text(json.dumps(bundle['history'],indent=2))
        selected=pd.read_parquet(dest/f'{winner}_embeddings.parquet');selected.to_parquet(dest/'selected_embeddings.parquet',index=False)
        np.save(dest/'selected_embeddings.npy',np.load(dest/f'{winner}_embeddings.npy'))
        for split in ['train','validation','test']:selected[selected.split.eq(split)].to_parquet(dest/f'{split}.parquet',index=False)
        pd.concat([rows,pd.DataFrame([row])],ignore_index=True).to_csv(dest/'comparison.csv',index=False)
        annotate_labels(m,own,True).to_parquet(dest/'label_metadata.parquet',index=False)
        print(asset,'label-informed validation',round(va['nmse'],5),'test',round(te['nmse'],5),'selected',winner,flush=True)
    aggregate_results(out)
    availability=[json.loads(p.read_text()) for p in sorted(out.glob('PUMP-*/label_availability.json'))]
    pd.DataFrame(availability).to_csv(out/'label_availability.csv',index=False)
    manifest={'model':LABEL_KIND,'event_sources':{name:sha256(DATA/name) for name in LABEL_FILES},'uses_training_labels':True,
              'validation_labels_used_for_selection':False,'test_labels_used_for_selection':False,'inference_requires_labels':False,
              'training_label_scope':'Each pump first60% only; pre-detectable/unobservable overlaps excluded','pumps':availability}
    (out/'labels_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Completed label-informed extension',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pumps',nargs='+');args=parser.parse_args();extend(pumps=args.pumps)
