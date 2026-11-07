"""Independent train-fitted maps for each pump and each 32D encoder."""
import os,tempfile
os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),'latent-space-matplotlib'))
os.environ.setdefault('NUMBA_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import json,joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import umap
from plotly.offline import get_plotlyjs
from data import ROOT,fingerprint
from models import KINDS
from label_informed import LABEL_KIND
OUT=ROOT/'results';FIG=OUT/'figures'
TITLES={'pca':'PCA','kernel_pca':'RBF kernel PCA','autoencoder':'Autoencoder','denoising_ae':'Denoising AE','temporal_ae':'Temporal AE','label_informed':'Label-informed AE'}
SPLIT_COLORS={'train':'#247b9c','validation':'#d58b25','test':'#9c519b'}
CLUSTER_COLORS=['#247b9c','#db8d28','#9650a0','#c75258','#40926b','#7c80ce','#97734a','#bd638a']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def plot_map(ax,xy,labels,title,colors):
    for label in sorted(set(labels)):
        keep=np.asarray(labels)==label
        ax.scatter(xy[keep,0],xy[keep,1],s=8,alpha=.7,color=colors[label],label=str(label),linewidths=0,rasterized=True)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=11);ax.set_xticks([]);ax.set_yticks([])


def main():
    FIG.mkdir(exist_ok=True)
    comparison=pd.read_csv(OUT/'model_comparison.csv')
    selected=pd.read_csv(OUT/'selected_models.csv').set_index('asset_id')
    payload={'titles':TITLES,'models':[*KINDS,LABEL_KIND],'pumps':{},'metrics':comparison.replace({np.nan:None}).to_dict('records')}
    quality=[];map_rows=[]
    for asset in selected.index:
        dest=OUT/asset;meta=pd.read_parquet(dest/'metadata.parquet')
        labelmeta=pd.read_parquet(dest/'label_metadata.parquet')
        meta['fault_label']=labelmeta.fault_label;meta['label_eligible']=labelmeta.label_eligible
        kinds=comparison.loc[comparison.asset_id.eq(asset),'model'].tolist()
        availability=json.loads((dest/'label_availability.json').read_text())
        indices=np.flatnonzero(meta.minute_index.mod(10).eq(0));sample=meta.iloc[indices].reset_index(drop=True)
        train=sample.split.eq('train').to_numpy();held=~train;eligible=sample.score_eligible.to_numpy()
        winner=selected.loc[asset,'model'];maps={};rows=sample.copy()
        rows['available_utc']=rows.available_utc.astype(str);rows['window_start_utc']=rows.window_start_utc.astype(str)
        info={'selected':winner,'rows':rows.to_dict('records'),'maps':maps,'models':kinds,'label_status':availability,
              'features':json.loads((dest/'features.json').read_text()),'selection':json.loads((dest/'selection.json').read_text())}
        for kind in kinds:
            print('Projecting',asset,kind,flush=True)
            z=np.load(dest/f'{kind}_embeddings.npy');zs=z[indices]
            target=dest/f'{kind}_projection.npz';stamp=dest/f'{kind}_projection.json'
            settings={'neighbors':25,'min_dist':.15,'epochs':200,'seed':42,'stride':10,'training_only':True}
            sig=fingerprint([dest/f'{kind}_embeddings.npy',dest/'metadata.parquet'],settings)
            if target.exists() and stamp.exists() and json.loads(stamp.read_text())['fingerprint']==sig:
                cached=np.load(target);xy=cached['umap'];pc=cached['pca']
            else:
                reducer=umap.UMAP(n_neighbors=25,min_dist=.15,n_components=2,n_epochs=200,random_state=42,n_jobs=1)
                xy=np.zeros((len(sample),2),np.float32);xy[train]=reducer.fit_transform(zs[train]);xy[held]=reducer.transform(zs[held])
                linear=PCA(n_components=2).fit(z[meta.split.eq('train')]);pc=linear.transform(zs)
                joblib.dump(reducer,dest/f'{kind}_umap.joblib');joblib.dump(linear,dest/f'{kind}_projection_pca.joblib')
                np.savez_compressed(target,umap=xy,pca=pc,indices=indices)
                stamp.write_text(json.dumps({'fingerprint':sig,**settings},indent=2))
            export=pd.read_parquet(dest/f'{kind}_embeddings.parquet',columns=['cluster','reconstruction_error'])
            maps[kind]={'umap':xy.round(5).tolist(),'pca':pc.round(5).tolist(),'cluster':export.cluster.iloc[indices].astype(int).tolist(),
                        'error':export.reconstruction_error.iloc[indices].astype(float).tolist()}
            for split in ['validation','test']:
                keep=sample.split.eq(split).to_numpy()&eligible
                quality.append({'asset_id':asset,'model':kind,'split':split,'umap_trustworthiness_15':float(trustworthiness(zs[keep],xy[keep],n_neighbors=15))})
            if kind==winner:
                exported=sample.assign(umap_x=xy[:,0],umap_y=xy[:,1],cluster=export.cluster.iloc[indices].to_numpy(),reconstruction_error=export.reconstruction_error.iloc[indices].to_numpy(),selected_model=winner)
                map_rows.append(exported)
        payload['pumps'][asset]=info
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for ax,kind in zip(axes.flat,kinds):
            metric=comparison[(comparison.asset_id==asset)&(comparison.model==kind)].iloc[0]
            plot_map(ax,np.asarray(maps[kind]['umap']),sample.split.to_numpy(),f"{TITLES[kind]}{' *' if kind==winner else ''}\nval {metric.validation_nmse:.3g} · test {metric.test_nmse:.3g}",SPLIT_COLORS)
        if LABEL_KIND not in kinds:
            axes.flat[-1].axis('off');axes.flat[-1].text(.5,.5,'Label-informed AE unavailable\n'+availability['reason'],ha='center',va='center',wrap=True,fontsize=11,color='#657583')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
        fig.suptitle(f'{asset}: {len(kinds)} independent 32D encoders → UMAP',x=.04,ha='left',fontsize=17,fontweight='bold')
        fig.text(.04,.035,'* Selected using validation reconstruction only. Each map has independent axes. One point every 10 minutes.',fontsize=9,color='#596675')
        fig.tight_layout(rect=[0,.12,1,.88]);fig.savefig(FIG/f'{asset}_models.png',dpi=160);plt.close(fig)
    for colorby in ['split','cluster','label_informed']:
        fig,axes=plt.subplots(3,4,figsize=(17,12))
        for ax,(asset,info) in zip(axes.flat,payload['pumps'].items()):
            kind=LABEL_KIND if colorby=='label_informed' else info['selected']
            if kind not in info['maps']:
                ax.axis('off');ax.text(.5,.5,asset+'\nLabel-informed AE unavailable\n'+info['label_status']['reason'],ha='center',va='center',wrap=True,fontsize=10,color='#657583');continue
            metric=comparison[comparison.asset_id.eq(asset)&comparison.model.eq(kind)].iloc[0];xy=np.asarray(info['maps'][kind]['umap'])
            labels=np.array([r['split'] for r in info['rows']]) if colorby=='split' else np.array(info['maps'][kind]['cluster'])
            colors=SPLIT_COLORS if colorby=='split' else dict(enumerate(CLUSTER_COLORS))
            plot_map(ax,xy,labels,f'{asset} · {TITLES[kind]}\nvalidation {metric.validation_nmse:.3g} / test {metric.test_nmse:.3g}',colors)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=8,frameon=False,markerscale=2)
        fig.suptitle('One pump, one latent space',x=.05,ha='left',fontsize=23,fontweight='bold')
        fig.text(.05,.945,('Label-informed encoders · trained separately where supervision is sufficient · colors = cluster' if colorby=='label_informed' else f'12 pumps · 32 dimensions · selected by validation reconstruction · colors = {colorby}'),fontsize=12,color='#536071')
        fig.text(.05,.02,'UMAP fitted to training minutes; held-out minutes transformed. Axes and cluster numbers are local to each pump. Clusters are not fault labels.',fontsize=10,color='#536071')
        fig.subplots_adjust(top=.9,bottom=.1,hspace=.28,wspace=.16)
        fig.savefig(FIG/f'per_pump_{colorby}.png',dpi=170);fig.savefig(FIG/f'per_pump_{colorby}.pdf');plt.close(fig)
    pd.DataFrame(quality).to_csv(OUT/'projection_quality.csv',index=False)
    pd.concat(map_rows,ignore_index=True).to_parquet(OUT/'plot_rows.parquet',index=False)
    html=TEMPLATE.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(payload,default=str,allow_nan=False).replace('</','<\\/')).replace('__FIT_COUNT__',str(len(comparison)))
    (OUT/'explorer.html').write_text(html)
    print('Wrote per-pump explorer including label-informed comparisons',flush=True)

TEMPLATE=r'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pump latent spaces · 32D</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f2f4f6;color:#142638;font-family:system-ui,-apple-system,sans-serif}main{max-width:1560px;margin:auto;padding:32px}header{display:flex;justify-content:space-between;gap:24px;align-items:flex-start}h1{font-size:34px;letter-spacing:-1.2px;margin:4px 0 10px}h2{font-size:18px;margin:0 0 16px}.eyebrow{color:#507083;font-size:11px;text-transform:uppercase;letter-spacing:2px;font-weight:750}p{line-height:1.55}.sub{color:#586a79;max-width:950px;margin:0}.badge{border:1px solid #a7c5ba;background:#e4f1e9;padding:8px 14px;border-radius:30px;white-space:nowrap;font-size:12px}.controls{display:flex;gap:16px;flex-wrap:wrap;margin:26px 0 18px;background:white;border:1px solid #d9e0e5;padding:16px 20px;border-radius:12px}label{display:flex;flex-direction:column;gap:6px;color:#5b6c79;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px}select{min-width:150px;border:1px solid #b8c6d0;padding:8px;border-radius:6px;background:#fff;color:#132c40;font-size:14px;text-transform:none}.grid{display:grid;grid-template-columns:minmax(0,1fr) 370px;gap:18px}.card{background:white;border:1px solid #d9e0e5;border-radius:12px;padding:20px;min-width:0}.stats{display:flex;gap:25px;flex-wrap:wrap}.stat strong{display:block;font-size:23px;font-weight:650}.stat span{font-size:11px;color:#647583}.small{font-size:12px;line-height:1.5;color:#5a6d7d}.metric{font-size:13px}.metric td,.metric th{padding:9px 5px;text-align:right;border-bottom:1px solid #e7ebef}.metric td:first-child,.metric th:first-child{text-align:left}.metric{width:100%;border-collapse:collapse}.metric tr.active{background:#e9f3f6}.metric tr{cursor:pointer}.metric th{font-size:10px;color:#637586}.note{border-left:3px solid #3c8ca8;padding:10px 13px;background:#eff7fa;font-size:12px;line-height:1.55;margin-top:16px}.detail{margin-top:18px}.pump-cards{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:18px}.pump-card{border:1px solid #cad6df;background:#fff;color:#233a4b;border-radius:8px;padding:12px;text-align:left;cursor:pointer;font:inherit;font-size:12px}.pump-card.active{border:2px solid #287a99;background:#edf6f9;padding:11px}.pump-card strong{display:block;margin-bottom:5px}.pump-card span{color:#657787;font-size:11px}a{color:#247b9c}#scatter{height:570px}#timeline{height:250px}#point{min-height:84px;white-space:pre-line}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#576d7a}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.formula{font-family:ui-monospace,monospace;font-size:12px;background:#edf1f4;padding:12px;line-height:1.7;border-radius:6px}footer{padding:18px 0 0;font-size:12px;color:#607483}@media(max-width:1000px){main{padding:18px}.grid{grid-template-columns:1fr}.pump-cards{grid-template-columns:repeat(3,1fr)}header{display:block}.badge{display:inline-block;margin-top:12px}#scatter{height:480px}}
</style><script>__PLOTLY__</script></head><body><main>
<header><div><div class="eyebrow">Per-pump representation experiment</div><h1>One pump, one latent space.</h1><p class="sub">Five unsupervised encoders plus a label-informed autoencoder where training labels support it. Every minute becomes a 32-dimensional vector. Models are selected by validation reconstruction error.</p></div><div class="badge">12 pumps · __FIT_COUNT__ fits · 32 dimensions</div></header>
<div class="pump-cards" id="pumps"></div>
<div class="controls"><label>Pump<select id="pump" aria-label="Pump"></select></label><label>Encoder<select id="model" aria-label="Encoder"></select></label><label>Projection<select id="projection" aria-label="Projection"><option value="umap">UMAP</option><option value="pca">PCA · 2D</option></select></label><label>Color by<select id="color" aria-label="Color by"><option value="cluster">Unsupervised cluster</option><option value="split">Dataset split</option><option value="time">Time</option><option value="operating_mode">Operating mode</option><option value="error">Reconstruction error</option><option value="fault_label">Fault annotation</option></select></label><label>Show<select id="split" aria-label="Show"><option value="all">All splits</option><option value="train">Train</option><option value="validation">Validation · scored</option><option value="test">Test · scored</option></select></label></div>
<div class="grid"><section class="card"><div class="stats" id="stats"></div><div id="scatter"></div><div class="small">One point every 10 minutes. K-means uses the full 32D training vectors; UMAP is fitted on a training sample. Click a point for its timestamp. Axes and cluster numbers are independent across pumps and encoders.</div></section>
<aside class="card"><h2 id="rank-title">Reconstruction comparison</h2><p class="small">Lower is better. The ★ model is chosen using validation only. Click a row to inspect that encoder.</p><table class="metric"><thead><tr><th>Encoder</th><th>Labels</th><th>Train</th><th>Val</th><th>Test</th></tr></thead><tbody id="metrics"></tbody></table><div class="note" id="selection"></div><p class="small" id="label-status"></p><h2 style="margin-top:24px">Selected point</h2><div class="small" id="point">Click a point in the map.</div><div class="small" id="feature-note"></div></aside></div>
<section class="card detail"><h2>Reconstruction through time</h2><div class="legend"><span><i class="dot" style="background:#247b9c"></i>Train · 6,048 minutes</span><span><i class="dot" style="background:#d58b25"></i>Validation · 2,016 minutes</span><span><i class="dot" style="background:#9c519b"></i>Test · 2,016 minutes</span></div><div id="timeline"></div><p class="small">The first 120 minutes of validation and test are warmup and excluded from the scores. This timeline displays sampled per-minute error; the table averages errors by feature before averaging across features.</p></section>
<section class="card detail"><h2>What the score measures</h2><div class="formula">For each feature: average observed squared reconstruction error / training variance.<br>Score = average of those feature errors. Missing measurements are ignored.<br>Skill = 1 − model error / error from always predicting the training mean.</div><p class="small">Every model reconstructs the same features for a given pump. Scaling, feature selection, encoders and clusters are fitted on training data. Neural checkpoints and the winning encoder use validation error. Test error is reported after selection. Inputs include causal sensor, physics and history features. The label-informed AE uses only its own pump’s training-period fault annotations to shape the latent space. Validation/test annotations do not train encoders; the separate fault-detection comparison uses them for detector selection and evaluation. Inference always uses sensor features and masks, without labels.</p><p class="small">Good reconstruction means that information survives compression. It does not prove fault detection or separation. Cluster colors have no assigned fault meaning. This is a within-pump chronological experiment, separate from the original benchmark's unseen-pump split.</p><a href="fault_detection.html">Fault detection comparison</a> · <a href="figures/per_pump_label_informed.png" target="_blank">Label-informed maps</a> · <a href="figures/per_pump_split.png" target="_blank">All 12 maps by split</a> · <a href="figures/per_pump_cluster.png" target="_blank">All 12 maps by cluster</a> · <a href="model_comparison.csv" download>Download model scores</a> · <a href="selected_models.csv" download>Download selected encoders</a></section>
<footer>60% train / 20% validation / 20% test for every pump · single seed 42 · independent latent spaces; do not concatenate them as a shared semantic coordinate system.</footer>
</main><script>
const D=__DATA__, $=id=>document.getElementById(id), palette=['#247b9c','#db8d28','#9650a0','#c75258','#40926b','#7c80ce','#97734a','#bd638a'], splitColors={train:'#247b9c',validation:'#d58b25',test:'#9c519b'}, cfg={responsive:true,displaylogo:false,modeBarButtonsToRemove:['sendDataToCloud','shareChart']};
const pumps=Object.keys(D.pumps); $('pump').innerHTML=pumps.map(x=>`<option>${x}</option>`).join('');
function n(x){return x>=100?x.toExponential(2):x.toPrecision(3)}
function row(a,k){return D.metrics.find(x=>x.asset_id===a&&x.model===k)}
function choosePump(a){$('pump').value=a;const p=D.pumps[a];$('model').innerHTML=D.models.map(x=>`<option value="${x}" ${p.models.includes(x)?'':'disabled'}>${D.titles[x]}${p.models.includes(x)?'':' · unavailable'}</option>`).join('');$('model').value=p.selected;draw()}
$('pumps').innerHTML=pumps.map(a=>{const k=D.pumps[a].selected;return `<button class="pump-card" data-pump="${a}"><strong>${a}</strong>${D.titles[k]}<br><span>test ${n(row(a,k).test_nmse)}</span></button>`}).join('');
$('pumps').addEventListener('click',e=>{const b=e.target.closest('[data-pump]');if(b)choosePump(b.dataset.pump)});
function draw(){
 const a=$('pump').value,k=$('model').value,p=D.pumps[a],m=p.maps[k],metric=row(a,k),color=$('color').value,proj=$('projection').value,split=$('split').value;
 document.querySelectorAll('.pump-card').forEach(b=>b.classList.toggle('active',b.dataset.pump===a));
 $('stats').innerHTML=`<div class="stat"><strong>${a}</strong><span>${D.titles[k]} · 32 dimensions</span></div><div class="stat"><strong>${n(metric.validation_nmse)}</strong><span>Validation reconstruction</span></div><div class="stat"><strong>${n(metric.test_nmse)}</strong><span>Test reconstruction</span></div><div class="stat"><strong>${(metric.test_skill*100).toFixed(1)}%</strong><span>Test improvement over mean</span></div>`;
 $('metrics').innerHTML=p.models.map(name=>{const v=row(a,name);return `<tr data-model="${name}" class="${name===k?'active':''}"><td>${name===p.selected?'★ ':''}${D.titles[name]}</td><td>${v.uses_training_labels?'Train':'No'}</td><td>${n(v.train_nmse)}</td><td>${n(v.validation_nmse)}</td><td>${n(v.test_nmse)}</td></tr>`}).join('');
 $('selection').textContent=`${D.titles[p.selected]} has the lowest validation error for ${a}. The test period was not used to select it.`;
 $('label-status').textContent=p.label_status.available?`Label-informed AE: ${p.label_status.supervised_rows.toLocaleString()} unambiguous training minutes; labels never needed at inference.`:`Label-informed AE unavailable: ${p.label_status.reason}.`;
 $('feature-note').textContent=`${metric.input_features} features reconstructed. ${Object.keys(p.features.excluded).length} absent or constant training features excluded. Sensor tier: ${p.rows[0].instrumentation_tier}. Uses training labels: ${metric.uses_training_labels?'yes':'no'}.`;
 $('point').textContent='Click a point in the map.';
 const indices=p.rows.map((_,i)=>i).filter(i=>split==='all'||(p.rows[i].split===split&&p.rows[i].score_eligible));
 const numeric=['time','error'].includes(color);let groups=numeric?['all']:[...new Set(indices.map(i=>color==='cluster'?m.cluster[i]:p.rows[i][color]))].sort();
 const traces=groups.map((g,j)=>{const ids=indices.filter(i=>numeric|| (color==='cluster'?m.cluster[i]:p.rows[i][color])===g);const colors=numeric?ids.map(i=>color==='time'?p.rows[i].minute_index/1440:Math.log10(Math.max(m.error[i],1e-8))):(color==='split'?splitColors[g]:color==='cluster'?palette[g%8]:palette[j%8]);return {type:'scattergl',mode:'markers',name:color==='cluster'?`Cluster ${g}`:g,x:ids.map(i=>m[proj][i][0]),y:ids.map(i=>m[proj][i][1]),customdata:ids,marker:{size:6,opacity:.75,color:colors,colorscale:'Viridis',showscale:numeric,colorbar:{title:{text:color==='time'?'Day':'log₁₀ error'},thickness:12}},text:ids.map(i=>`${p.rows[i].available_utc}<br>${p.rows[i].split}${p.rows[i].score_eligible?'':' · warmup'} · cluster ${m.cluster[i]}<br>error ${n(m.error[i])} · ${p.rows[i].operating_mode}<br>${p.rows[i].fault_label}`),hovertemplate:'%{text}<extra></extra>'}});
 const layout={margin:{l:35,r:35,t:75,b:45},paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'system-ui',color:'#506575'},xaxis:{title:{text:proj.toUpperCase()+' 1'},zeroline:false,showgrid:false,showticklabels:false},yaxis:{title:{text:proj.toUpperCase()+' 2'},zeroline:false,showgrid:false,showticklabels:false},legend:{orientation:'h',y:1.03,yanchor:'bottom',font:{size:10}},hovermode:'closest',uirevision:a+k+proj};
 Plotly.react('scatter',traces,layout,cfg).then(()=>{const el=$('scatter');el.removeAllListeners('plotly_click');el.on('plotly_click',ev=>{const i=ev.points[0].customdata,r=p.rows[i];$('point').textContent=`${r.available_utc}\n${r.split}${r.score_eligible?'':' · warmup, excluded from score'} · ${r.operating_mode}\nCluster ${m.cluster[i]} · minute error ${n(m.error[i])}\nAnnotation: ${r.fault_label}\n32D row: ${a}/${k}_embeddings.parquet, minute_index ${r.minute_index}`})});
 const ts=Object.keys(splitColors).map(s=>{const ids=p.rows.map((r,i)=>r.split===s?i:-1).filter(i=>i>=0);return {type:'scatter',mode:'lines',name:s,x:ids.map(i=>p.rows[i].available_utc),y:ids.map(i=>Math.max(m.error[i],1e-8)),line:{width:1.2,color:splitColors[s]},hovertemplate:'%{x}<br>error %{y:.4g}<extra>%{fullData.name}</extra>'}});
 Plotly.react('timeline',ts,{margin:{l:65,r:20,t:15,b:40},xaxis:{type:'date'},yaxis:{type:'log',title:{text:'Minute error (log)'}},showlegend:false,font:{family:'system-ui',color:'#506575'},paper_bgcolor:'#fff',plot_bgcolor:'#fff'},cfg);
}
$('pump').onchange=()=>choosePump($('pump').value);['model','projection','color','split'].forEach(x=>$(x).onchange=draw);$('metrics').onclick=e=>{const r=e.target.closest('[data-model]');if(r){$('model').value=r.dataset.model;draw()}};choosePump(pumps[0]);
</script></body></html>'''
if __name__=='__main__':main()
