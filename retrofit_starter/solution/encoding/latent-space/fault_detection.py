"""Frozen-encoder fault probes with explicit single-class evaluation limits."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import json,time,warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve
from data import ROOT,sha256
from models import KINDS
from label_informed import LABEL_KIND
MODELS=KINDS+[LABEL_KIND]
READOUTS=['logistic','knn']
ANY='ANY_FAULT'


def target_vector(labels,target):
    result=np.full(len(labels),-1,dtype=np.int8)
    known=labels.label_eligible.to_numpy(dtype=bool).copy()
    names=labels.fault_label.fillna('UNLABELED').to_numpy()
    known &= ~np.isin(names,['UNLABELED','AMBIGUOUS_EVENT'])
    values=np.array([value!='NO_LABELED_EVENT' if target==ANY else target in value.split('|') for value in names],dtype=np.int8)
    result[known]=values[known]
    return result


def training_targets(labels):
    known=labels.loc[labels.label_eligible,'fault_label']
    names={name for value in known for name in value.split('|')}
    names-= {'NO_LABELED_EVENT','UNLABELED','AMBIGUOUS_EVENT'}
    return [ANY,*sorted(names)]


def fit_probe(z,y,train,method):
    train=np.asarray(train)
    if set(np.unique(y[train]))!={0,1}:raise ValueError('Probe requires both known classes in training')
    if method=='logistic':
        readout=LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=2000,tol=1e-6,random_state=42)
    elif method=='knn':readout=KNeighborsClassifier(n_neighbors=min(15,len(train)),weights='uniform',n_jobs=1)
    else:raise ValueError(method)
    probe=make_pipeline(StandardScaler(),readout)
    probe.fit(z[train],y[train])
    return probe


def choose_threshold(y,probabilities):
    if set(np.unique(y))!={0,1}:return None
    precision,recall,thresholds=precision_recall_curve(y,probabilities)
    f1=np.divide(2*precision[:-1]*recall[:-1],precision[:-1]+recall[:-1],out=np.zeros(len(thresholds)),where=precision[:-1]+recall[:-1]>0)
    # Prefer the more conservative (higher) threshold in an exact tie.
    best=np.flatnonzero(np.isclose(f1,f1.max(),rtol=0,atol=1e-12))[-1]
    return float(thresholds[best])


def binary_metrics(y,probabilities,threshold=.5):
    y=np.asarray(y);probabilities=np.asarray(probabilities)
    if len(y)!=len(probabilities) or not np.isin(y,[0,1]).all():raise ValueError('Metrics require aligned known binary labels')
    pos=int((y==1).sum());neg=int((y==0).sum());both=pos>0 and neg>0
    predicted=probabilities>=threshold
    tp=int(((y==1)&predicted).sum());fp=int(((y==0)&predicted).sum());fn=pos-tp;tn=neg-fp
    return {'rows':len(y),'positives':pos,'negatives':neg,'prevalence':pos/len(y) if len(y) else None,
            'ap':float(average_precision_score(y,probabilities)) if both else None,
            'auroc':float(roc_auc_score(y,probabilities)) if both else None,
            'recall':tp/pos if pos else None,'false_positive_rate':fp/neg if neg else None,
            'precision':tp/(tp+fp) if tp+fp else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,'threshold':threshold,
            'ranking_evaluable':both,'ranking_unavailable_reason':None if both else 'no_scored_rows' if not len(y) else 'only_positive_labels' if pos else 'only_negative_labels'}


def common_ranking(rows,models=MODELS):
    """Equal pump weight; exact same pump/target cases for every encoder."""
    if rows.empty:return [],[]
    valid=rows[rows.ap.notna()&rows.model.isin(models)].copy()
    cases=[]
    for case,group in valid.groupby(['asset_id','target']):
        if set(group.model)==set(models):cases.append(case)
    cases=sorted(cases)
    if not cases:return [],[]
    keep=pd.MultiIndex.from_frame(valid[['asset_id','target']]).isin(cases);shared=valid.loc[keep]
    rows=[]
    for model in models:
        scores=shared[shared.model.eq(model)].groupby('asset_id').ap.mean()
        rows.append({'model':model,'macro_ap':float(scores.mean()),'pumps':len(scores),'cases':len(cases)})
    return sorted(rows,key=lambda r:(-r['macro_ap'],r['model'])),cases


def selection_from_validation(rows):
    decisions={}
    for task in ['any_fault','fault_types']:
        for readout in READOUTS:
            subset=rows[rows.readout.eq(readout)&(rows.target.eq(ANY) if task=='any_fault' else rows.target.ne(ANY))]
            ranking,cases=common_ranking(subset)
            decisions[f'{task}_{readout}']={'ranking':ranking,'common_cases':[list(x) for x in cases],
                                           'winner':ranking[0]['model'] if ranking else None,'metric':'equal-pump macro average precision on identical mixed-label validation cases'}
    return decisions


def run():
    started=time.monotonic();out=ROOT/'results/fault_detection';out.mkdir(exist_ok=True)
    source=ROOT/'results';comparison=pd.read_csv(source/'model_comparison.csv')
    # Inputs are frozen outputs from the prior experiment. Preserve their hashes.
    inputs=[source/'model_comparison.csv',source/'labels_manifest.json']
    jobs=[];validation_rows=[];coverage=[];predictions=[];warning_records=[]
    for asset in sorted(comparison.asset_id.unique()):
        folder=source/asset;meta=pd.read_parquet(folder/'metadata.parquet');labels=pd.read_parquet(folder/'label_metadata.parquet')
        inputs += [folder/'metadata.parquet',folder/'label_metadata.parquet']
        assert np.array_equal(meta.asset_id,labels.asset_id) and np.array_equal(meta.available_utc,labels.available_utc)
        train_period=meta.split.eq('train').to_numpy();targets=training_targets(labels.loc[train_period])
        reporting_targets=training_targets(labels)  # Coverage only; never defines fitted targets.
        for target in reporting_targets:
            y=target_vector(labels,target)
            for split in ['train','validation','test']:
                indices=np.flatnonzero(meta.split.eq(split).to_numpy()&meta.score_eligible.to_numpy()&(y>=0))
                values=y[indices];coverage.append({'asset_id':asset,'target':target,'split':split,'rows':len(indices),'positives':int((values==1).sum()),'negatives':int((values==0).sum()),'target_in_training_vocabulary':target in targets,'both_classes':set(np.unique(values))=={0,1}})
        valid_targets=[]
        for target in targets:
            y=target_vector(labels,target);train=np.flatnonzero(train_period&(y>=0))
            if set(np.unique(y[train]))=={0,1}:valid_targets.append((target,y,train))
        if not valid_targets:
            print(asset,'no trainable binary targets with both known classes',flush=True);continue
        for model in comparison.loc[comparison.asset_id.eq(asset),'model']:
            path=folder/f'{model}_embeddings.npy';inputs.append(path);z=np.load(path)
            for target,y,train in valid_targets:
                val=np.flatnonzero(meta.split.eq('validation').to_numpy()&meta.score_eligible.to_numpy()&(y>=0))
                test=np.flatnonzero(meta.split.eq('test').to_numpy()&meta.score_eligible.to_numpy()&(y>=0))
                for method in READOUTS:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');probe=fit_probe(z,y,train,method)
                    warning_records += [{'asset_id':asset,'model':model,'target':target,'readout':method,'message':str(w.message)} for w in caught]
                    pv=probe.predict_proba(z[val])[:,1] if len(val) else np.empty(0)
                    threshold=choose_threshold(y[val],pv);origin='validation_max_f1' if threshold is not None else 'fixed_0.5_no_mixed_validation'
                    threshold=.5 if threshold is None else threshold
                    identity={'asset_id':asset,'model':model,'target':target,'readout':method,'threshold_origin':origin}
                    row={**identity,'split':'validation',**binary_metrics(y[val],pv,threshold)};validation_rows.append(row)
                    dest=out/asset/model;dest.mkdir(parents=True,exist_ok=True)
                    model_file=dest/f'{target}__{method}.joblib'
                    joblib.dump({'probe':probe,'threshold':threshold,'threshold_origin':origin,'train_indices':train,'target':target,'encoder':model,'asset_id':asset,'dimensions':32},model_file)
                    jobs.append({'identity':identity,'probe':probe,'threshold':threshold,'test':test,'val':val,'z':z,'y':y,'meta':meta,'model_file':str(model_file.relative_to(out))})
                    for i,prob in zip(val,pv):predictions.append({**identity,'split':'validation','minute_index':int(i),'available_utc':str(meta.available_utc.iloc[i]),'y':int(y[i]),'score':float(prob),'predicted':bool(prob>=threshold)})
            print('Fitted probes',asset,model,'targets',len(valid_targets),flush=True)
    val_frame=pd.DataFrame(validation_rows);decisions=selection_from_validation(val_frame)
    selection={'primary':'any_fault_logistic','selected_model':decisions['any_fault_logistic']['winner'],'uses_test_for_selection':False,
               'encoder_fits_frozen':True,'decisions':decisions,'protocol':'same train-only probes, fixed settings, validation AP on common cases; validation-only F1 thresholds'}
    # Write a durable decision before calculating any test predictions or scores.
    (out/'selection.json').write_text(json.dumps(selection,indent=2,allow_nan=False))
    test_rows=[]
    for job in jobs:
        idx=job['test'];p=job['probe'].predict_proba(job['z'][idx])[:,1] if len(idx) else np.empty(0)
        test_rows.append({**job['identity'],'split':'test',**binary_metrics(job['y'][idx],p,job['threshold'])})
        for i,prob in zip(idx,p):predictions.append({**job['identity'],'split':'test','minute_index':int(i),'available_utc':str(job['meta'].available_utc.iloc[i]),'y':int(job['y'][i]),'score':float(prob),'predicted':bool(prob>=job['threshold'])})
    metrics=pd.concat([val_frame,pd.DataFrame(test_rows)],ignore_index=True)
    metrics.to_csv(out/'metrics.csv',index=False);pd.DataFrame(coverage).to_csv(out/'label_coverage.csv',index=False)
    pd.DataFrame(predictions).to_parquet(out/'predictions.parquet',index=False)
    (out/'warnings.json').write_text(json.dumps(warning_records,indent=2))
    manifest={'protocol_version':1,'seed':42,'probe_fits':len(jobs),'runtime_seconds':time.monotonic()-started,
              'input_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in dict.fromkeys(inputs)},'source_sha256':sha256(ROOT/'fault_detection.py'),
              'encoder_checkpoint_selection':'previous validation reconstruction; not refitted for fault labels','validation_threshold':'max F1, highest threshold in exact tie','single_class_ap_policy':'unavailable, never reported as perfect detection'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Selected provisional validation winner:',selection['selected_model'],flush=True)
    for key,value in decisions.items():print(key,value,flush=True)
    print('Fitted',len(jobs),'probes; warnings',len(warning_records),'test mixed-label rows',int(metrics[metrics.split.eq('test')].ranking_evaluable.sum()),flush=True)

if __name__=='__main__':run()
