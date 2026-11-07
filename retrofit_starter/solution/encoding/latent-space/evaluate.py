"""Label-free, feature-balanced reconstruction metrics."""
import numpy as np
def reconstruction_metrics(target,predicted,observed):
    target=np.asarray(target,dtype=np.float64);predicted=np.asarray(predicted,dtype=np.float64);observed=np.asarray(observed,bool)
    count=observed.sum(axis=0);valid=count>0
    if not valid.any():raise ValueError('No observed reconstruction targets')
    errors=np.where(observed,(target-predicted)**2,0).sum(axis=0);baseline=np.where(observed,target**2,0).sum(axis=0)
    mse=np.full(target.shape[1],np.nan);base=np.full_like(mse,np.nan)
    mse[valid]=errors[valid]/count[valid];base[valid]=baseline[valid]/count[valid]
    mean=float(mse[valid].mean());base_mean=float(base[valid].mean())
    per_row=np.where(observed,(target-predicted)**2,0).sum(axis=1)/np.maximum(observed.sum(axis=1),1)
    return {'nmse':mean,'mean_baseline_nmse':base_mean,'skill_vs_train_mean':1-mean/base_mean if base_mean>0 else None,
            'valid_cells':int(count.sum()),'features_scored':int(valid.sum()),'feature_nmse':mse.tolist(),
            'feature_baseline_nmse':base.tolist(),'feature_counts':count.tolist(),'row_nmse':per_row.astype(np.float32)}
def select_model(rows):return min(rows,key=lambda r:r['validation_nmse'])['model']
