"""Fixed offline diagnostics; no labels are consumed by representation fitting."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score,adjusted_rand_score,normalized_mutual_info_score,silhouette_score
from data import evaluation_eligible


def evaluate(z,meta,y,names,seed=42):
    valid=evaluation_eligible(meta)
    spaced=meta.minute_index.mod(5).eq(0)
    train=np.flatnonzero(meta.split.eq('train')&valid&spaced)
    validation=np.flatnonzero(meta.split.eq('validation')&valid&spaced)
    nn=NearestNeighbors(n_neighbors=15,n_jobs=4).fit(z[train])
    _,neighbors=nn.kneighbors(z[validation])
    probabilities=y[train][neighbors].mean(axis=1)
    per_fault=[]
    for j,name in enumerate(names):
        support=int(y[validation,j].sum())
        if support:
            per_fault.append({'fault_type':name,'positive_minutes_sampled':support,
                'prevalence':float(y[validation,j].mean()),'average_precision':float(average_precision_score(y[validation,j],probabilities[:,j]))})
    by_asset=[]
    for a in meta.iloc[validation].asset_id.unique():
        rows=meta.iloc[validation].asset_id.eq(a).to_numpy();scores=[]
        for j in range(len(names)):
            if y[validation[rows],j].sum()>0:scores.append(average_precision_score(y[validation[rows],j],probabilities[rows,j]))
        by_asset.append({'asset_id':a,'macro_ap':float(np.mean(scores)),'supported_fault_types':len(scores)})
    clustering_train=np.flatnonzero(meta.split.eq('train')&spaced)
    km=KMeans(n_clusters=12,n_init=10,random_state=seed).fit(z[clustering_train])
    clusters=km.predict(z)
    alt=KMeans(n_clusters=12,n_init=10,random_state=7).fit(z[clustering_train])
    v=validation[meta.iloc[validation].label.ne('MULTIPLE').to_numpy()]
    rng=np.random.default_rng(seed);sample=rng.choice(validation,min(2000,len(validation)),replace=False)
    silhouette=float(silhouette_score(z[sample],clusters[sample])) if len(np.unique(clusters[sample]))>1 else float('nan')
    metrics={'macro_ap':float(np.mean([r['average_precision'] for r in per_fault])),
        'prevalence_ap':float(np.mean([r['prevalence'] for r in per_fault])),
        'fault_cluster_ari':float(adjusted_rand_score(meta.iloc[v].label,clusters[v])),
        'fault_cluster_nmi':float(normalized_mutual_info_score(meta.iloc[v].label,clusters[v])),
        'asset_cluster_nmi':float(normalized_mutual_info_score(meta.iloc[clustering_train].asset_id,clusters[clustering_train])),
        'tier_cluster_nmi':float(normalized_mutual_info_score(meta.iloc[clustering_train].instrumentation_tier,clusters[clustering_train])),
        'mode_cluster_nmi':float(normalized_mutual_info_score(meta.iloc[clustering_train].operating_mode,clusters[clustering_train])),
        'cluster_stability_ari':float(adjusted_rand_score(clusters[clustering_train],alt.labels_)),
        'validation_silhouette':silhouette,'probe_train_rows':len(train),'probe_validation_rows':len(validation),
        'per_fault':per_fault,'per_asset':by_asset}
    running=meta.iloc[validation].running_fraction.ge(.99).to_numpy()
    running_scores=[average_precision_score(y[validation[running],j],probabilities[running,j]) for j in range(len(names)) if y[validation[running],j].sum()>0]
    metrics['running_macro_ap']=float(np.mean(running_scores))
    return metrics,km,clusters,{'indices':validation,'probabilities':probabilities}
