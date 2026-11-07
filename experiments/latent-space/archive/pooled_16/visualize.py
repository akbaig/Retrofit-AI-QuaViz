"""Train-fitted projection maps, standalone figures, and an offline explorer."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/latent-space-matplotlib')
os.environ.setdefault('NUMBA_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import json,joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import umap
from plotly.offline import get_plotlyjs
from data import ROOT

OUT=ROOT/'results';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
MODELS=['raw_pca','physical_pca','rbf_pca','denoising_ae','temporal_ae','supervised_contrastive']
TITLES={'raw_pca':'Raw-observation PCA','physical_pca':'Physics + history PCA','rbf_pca':'RBF kernel PCA',
        'denoising_ae':'Denoising autoencoder','temporal_ae':'Temporal autoencoder','supervised_contrastive':'Label-informed contrastive'}
palette=['#158bb8','#e48332','#8e62bb','#d55564','#43a478','#b29a31','#6c75d0','#c96f9c','#409b91','#a25b43','#8c9b3a','#5d819c','#b66dba']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def scatter(ax,xy,labels,title):
    categories=sorted(set(labels));color={v:palette[i%len(palette)] for i,v in enumerate(categories)}
    color.update({'NO_LABELED_EVENT':'#c8cdd3','PRE_DETECTABLE':'#f2dca3','UNOBSERVABLE_EVENT':'#aaaaaa','MULTIPLE':'#292f3b'})
    for label in sorted(categories,key=lambda x:x!='NO_LABELED_EVENT'):
        selected=np.asarray(labels)==label
        ax.scatter(xy[selected,0],xy[selected,1],s=5,alpha=.65,color=color[label],label=label,rasterized=True,linewidths=0)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=11)
    ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2')
    return color


def main():
    meta=pd.read_parquet(OUT/'metadata_development.parquet')
    indices=np.flatnonzero(meta.minute_index.mod(10).eq(0));sample=meta.iloc[indices].reset_index(drop=True)
    train=sample.split.eq('train').to_numpy();val=sample.split.eq('validation').to_numpy()
    selected=json.loads((OUT/'selection.json').read_text())['selected_model']
    comparison=pd.read_csv(OUT/'model_comparison.csv').set_index('model')
    projections={};clusters={};quality=[]
    for name in MODELS:
        z=np.load(OUT/f'{name}_development.npy');zs=z[indices]
        target=OUT/f'{name}_projection.npz'
        if target.exists():
            p=np.load(target);xy=p['umap'];pc=p['pca']
        else:
            print('Projecting',name,flush=True)
            reducer=umap.UMAP(n_neighbors=30,min_dist=.15,n_components=2,n_epochs=200,random_state=42,n_jobs=1)
            xy=np.zeros((len(sample),2),np.float32);xy[train]=reducer.fit_transform(zs[train]);xy[val]=reducer.transform(zs[val])
            pca=PCA(n_components=2).fit(z[meta.split.eq('train')]);pc=pca.transform(zs)
            joblib.dump(reducer,OUT/'models'/f'{name}_umap.joblib');joblib.dump(pca,OUT/'models'/f'{name}_projection_pca.joblib')
            np.savez_compressed(target,umap=xy,pca=pc,indices=indices)
        projections[name]={'umap':xy.round(5).tolist(),'pca':pc.round(5).tolist()}
        clusters[name]=np.load(OUT/f'{name}_clusters_development.npy')[indices].astype(int).tolist()
        vs=np.flatnonzero(val)[::2]
        quality.append({'model':name,'validation_umap_trustworthiness':float(trustworthiness(zs[vs],xy[vs],n_neighbors=15))})
    pd.DataFrame(quality).to_csv(OUT/'projection_quality.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    for ax,name in zip(axes.flat,MODELS):
        scatter(ax,np.asarray(projections[name]['umap'])[val],sample.loc[val,'label'].to_numpy(),f"{TITLES[name]}  ·  AP {comparison.loc[name,'macro_ap']:.3f}")
    handles,labels=axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles,[x.replace('_',' ').title() for x in labels],loc='lower center',bbox_to_anchor=(.5,.065),ncol=4,fontsize=9,frameon=False)
    fig.suptitle('Do similar fault states become neighbors on unseen pumps?',fontsize=19,fontweight='bold',x=.05,ha='left')
    fig.text(.05,.935,'Validation pumps 08–09 · 16-dimensional embeddings → train-fitted UMAP · one point every 10 minutes',fontsize=11,color='#526070')
    fig.text(.05,.025,'AP = multilabel average precision of a fixed 15-neighbor probe in 16D. Shapes and gaps in UMAP are not evidence of distinct fault classes.',fontsize=9,color='#526070')
    fig.subplots_adjust(top=.88,bottom=.19,hspace=.24,wspace=.13)
    fig.savefig(FIG/'model_comparison.png',dpi=170);fig.savefig(FIG/'model_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    xy=np.asarray(projections[selected]['umap'])
    for ax,col,title in zip(axes.flat,['broad_label','asset_id','instrumentation_tier','cluster'],['Fault state','Pump identity','Sensor tier','KMeans clusters in 16D']):
        labels=np.asarray(clusters[selected]).astype(str) if col=='cluster' else sample[col].to_numpy()
        scatter(ax,xy,labels,title);ax.legend(loc='best',fontsize=7,markerscale=2,frameon=False)
    fig.suptitle(f'{TITLES[selected]}: check what the clusters represent',fontsize=17,fontweight='bold')
    fig.text(.07,.025,'Training + validation · test omitted · cluster colors have no intrinsic fault meaning',fontsize=10,color='#526070')
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(FIG/'selected_diagnostics.png',dpi=170);plt.close(fig)
    # Sensor availability plot uses raw observed counts, without imputing missing hardware.
    counts=pd.read_csv(OUT/'observed_sample_counts.csv',index_col=0)
    fig,ax=plt.subplots(figsize=(13,5))
    values=counts.to_numpy(dtype=float)/(7*24*3600)
    im=ax.imshow(np.log10(values+1e-5),aspect='auto',cmap='viridis',vmin=-5,vmax=0)
    ax.set_yticks(range(len(counts)),counts.index);ax.set_xticks(range(len(counts.columns)),[c.replace('__count','').replace('_',' ') for c in counts],rotation=45,ha='right',fontsize=9)
    ax.set_title('Sensor coverage differs sharply between pumps',loc='left',fontsize=16,fontweight='bold')
    fig.colorbar(im,ax=ax,label='log10(observed samples / expected seconds)')
    fig.tight_layout();fig.savefig(FIG/'sensor_coverage.png',dpi=170);plt.close(fig)
    # Export inspectable plot rows; full embeddings remain a separate artifact.
    export=sample.copy();export['source_row_index']=indices
    export['umap_x']=xy[:,0];export['umap_y']=xy[:,1];export['cluster']=clusters[selected]
    export.to_csv(OUT/'plot_rows.csv',index=False)
    payload={'selected':selected,'titles':TITLES,'models':MODELS,'projections':projections,'clusters':clusters,
             'rows':sample.astype({'available_utc':str,'window_start_utc':str}).to_dict(orient='records'),
             'metrics':comparison.loc[MODELS].reset_index().replace({np.nan:None}).to_dict(orient='records')}
    html=TEMPLATE.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(payload,default=str).replace('</','<\\/'))
    (OUT/'explorer.html').write_text(html)
    print('Wrote explorer and figures',flush=True)


TEMPLATE=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pump latent space explorer</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#203046;font:15px system-ui,-apple-system,sans-serif}main{max-width:1500px;margin:auto;padding:32px}h1{font-size:34px;letter-spacing:-1px;margin:5px 0 10px}p{line-height:1.5;color:#536175}.eyebrow{font-size:12px;font-weight:700;letter-spacing:2px;color:#247c9b}.toolbar{display:flex;gap:20px;flex-wrap:wrap;background:white;border:1px solid #dfe5eb;padding:18px;border-radius:12px}label{display:flex;flex-direction:column;gap:7px;font-size:12px;font-weight:700;color:#526070}select{font:14px system-ui;border:1px solid #ccd6df;border-radius:6px;background:#fff;padding:9px;min-width:145px;color:#203046}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}.card{background:white;border:1px solid #dfe5eb;border-radius:12px;padding:18px}.value{font-size:26px;font-weight:650}.caption{font-size:12px;color:#627287;margin-top:5px}#plot{height:650px;background:white;border:1px solid #dfe5eb;border-radius:12px}.foot{display:grid;grid-template-columns:1.2fr 1fr;gap:18px;margin-top:18px}.foot h2{font-size:16px}#point{white-space:pre-wrap;font-size:13px;line-height:1.65;overflow-wrap:anywhere}a{color:#126986}.tag{background:#e4f1f7;padding:5px 9px;border-radius:20px;font-size:12px;color:#176780}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:9px;border-bottom:1px solid #edf0f3}th{color:#6c7a8d;font-weight:500}@media(max-width:800px){main{padding:16px}.cards{grid-template-columns:repeat(2,1fr)}.foot{grid-template-columns:1fr}#plot{height:540px}}</style>
<script>__PLOTLY__</script></head><body><main><div class="eyebrow">RETROFIT AI · DATA EXPLORATION</div><h1>Inside the pump latent space</h1><p>Compare representations of machine behavior. Each point is a completed minute, sampled every 10 minutes for display.<br>Fit: pumps 01–07. Validation: pumps 08–09. Test vectors are exported separately and are not shown here.</p>
<div class="toolbar"><label>Representation<select id="model"></select></label><label>View<select id="split"><option value="validation">Validation pumps</option><option value="train">Training pumps</option><option value="all">Training + validation</option></select></label><label>Color by<select id="color"><option value="label">Fault type</option><option value="broad_label">Machine / sensor</option><option value="cluster">KMeans cluster</option><option value="asset_id">Pump identity</option><option value="instrumentation_tier">Sensor tier</option><option value="operating_mode">Operating mode</option><option value="benign_type">Benign context</option></select></label><label>Projection<select id="projection"><option value="umap">UMAP (nonlinear)</option><option value="pca">PCA (linear)</option></select></label></div>
<div class="cards"><div class="card"><div class="value" id="ap"></div><div class="caption">Validation neighborhood macro AP</div></div><div class="card"><div class="value" id="nmi"></div><div class="caption">Asset / cluster NMI · lower means less identity grouping</div></div><div class="card"><div class="value">16</div><div class="caption">Dimensions in the reusable vectors</div></div><div class="card"><div class="value" id="points"></div><div class="caption">Points displayed · hover or click to inspect</div></div></div>
<div id="plot"></div><div class="foot"><div class="card"><h2>Reading this map</h2><p id="training"></p><p>Clustering uses KMeans with 12 clusters in the original 16-dimensional representation. UMAP and PCA only project it onto this screen; UMAP can exaggerate gaps. Cluster IDs do not name fault types.</p><p>“No labeled event” is not a guarantee of health. Some faults are pre-detectable or unobservable. Those ambiguous intervals are excluded from scoring, including when they overlap visible faults. Multiple simultaneous visible faults remain multilabel targets.</p><p>The neighborhood score is an exploratory diagnostic, not classification accuracy or the official alert score. Only two validation pumps are available; many adjacent minutes belong to the same incident.</p></div><div class="card"><h2>Selected observation</h2><div id="point">Click a point to see its pump, time, operating state and event metadata.</div></div></div><div class="card" style="margin-top:18px"><h2>Model comparison</h2><table id="comparison"></table></div><p><span class="tag">Synthetic data only</span> &nbsp;Source records remain unchanged. This explorer runs offline.</p></main>
<script>const D=__DATA__;const $=x=>document.getElementById(x);let visible=[];
D.models.forEach(m=>{let o=document.createElement('option');o.value=m;o.textContent=D.titles[m];$('model').appendChild(o)});$('model').value=D.selected;
function fmt(x){return Number(x).toFixed(3)}
function draw(){const model=$('model').value,split=$('split').value,color=$('color').value,projection=$('projection').value;const metric=D.metrics.find(x=>x.model===model);$('ap').textContent=fmt(metric.macro_ap);$('nmi').textContent=fmt(metric.asset_cluster_nmi);$('training').textContent=model==='supervised_contrastive'?'Label-informed encoder: training-event labels shape the representation through a contrastive objective. Labels are never inference inputs.':'Unsupervised encoder: fault labels are used only for evaluation and coloring, never for fitting this representation.';
visible=D.rows.map((r,i)=>i).filter(i=>split==='all'||D.rows[i].split===split);$('points').textContent=visible.length.toLocaleString();const groups={};visible.forEach(i=>{let key=color==='cluster'?'Cluster '+D.clusters[model][i]:D.rows[i][color]||'None';(groups[key]??=[]).push(i)});
const palette=['#158bb8','#e48332','#8e62bb','#d55564','#43a478','#b29a31','#6c75d0','#c96f9c','#409b91','#a25b43','#8c9b3a','#5d819c','#b66dba'];const traces=Object.keys(groups).sort().map((key,j)=>{const ids=groups[key];return {type:'scattergl',mode:'markers',name:key.replaceAll('_',' '),x:ids.map(i=>D.projections[model][projection][i][0]),y:ids.map(i=>D.projections[model][projection][i][1]),customdata:ids,text:ids.map(i=>D.rows[i].asset_id+' · '+D.rows[i].available_utc+'<br>'+D.rows[i].label),hovertemplate:'%{text}<extra></extra>',marker:{size:5,opacity:.75,color:key==='NO_LABELED_EVENT'?'#c7cdd4':palette[j%palette.length]}}});
Plotly.react('plot',traces,{margin:{l:55,r:20,t:30,b:50},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'system-ui',color:'#34455b'},xaxis:{title:{text:projection.toUpperCase()+' 1'},zeroline:false,gridcolor:'#edf1f5'},yaxis:{title:{text:projection.toUpperCase()+' 2'},zeroline:false,gridcolor:'#edf1f5'},legend:{orientation:'h',y:-.13,font:{size:10}},hovermode:'closest'},{responsive:true,displaylogo:false});}
['model','split','color','projection'].forEach(id=>$(id).addEventListener('change',draw));draw();$('plot').on('plotly_click',ev=>{const i=ev.points[0].customdata,r=D.rows[i];$('point').textContent='Pump: '+r.asset_id+' ('+r.split+')\nAvailable at: '+r.available_utc+'\nMinute starts: '+r.window_start_utc+'\nOperating mode: '+r.operating_mode+'\nSensor tier: '+r.instrumentation_tier+'\nVisible faults: '+(r.fault_types||'None labeled')+'\nActive event IDs: '+(r.active_event_ids||'None')+'\nPre-detectable overlap: '+r.has_predetectable_event+'\nUnobservable overlap: '+r.has_unobservable_event+'\nBenign context: '+(r.benign_type||'None labeled')+'\nTelemetry rows received: '+r.rows+' / 60\nCluster: '+D.clusters[$('model').value][i]});
$('comparison').innerHTML='<tr><th>Representation</th><th>Training labels</th><th>Macro AP</th><th>Running-only AP</th><th>Asset NMI</th></tr>'+D.metrics.map(m=>'<tr><td>'+D.titles[m.model]+'</td><td>'+(m.uses_training_labels?'Yes':'No')+'</td><td>'+fmt(m.macro_ap)+'</td><td>'+fmt(m.running_macro_ap)+'</td><td>'+fmt(m.asset_cluster_nmi)+'</td></tr>').join('');
</script></body></html>'''

if __name__=='__main__':main()
