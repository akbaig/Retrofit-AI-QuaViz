"""Read-only analysis of saved predictions. Selection mode runs before final testing."""
from pathlib import Path
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import rank_common,dump_json,sha256
from models import KINDS


def aggregate_counts(rows):
    s=rows[['tp','fp','fn','tn']].sum()
    tp,fp,fn,tn=[int(s[k]) for k in ['tp','fp','fn','tn']]
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn) if tp+fn else None,
                false_positive_rate=fp/(fp+tn) if fp+tn else None,
                precision=tp/(tp+fp) if tp+fp else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--select',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'results');args=p.parse_args();out=args.output
    dev=pd.read_csv(out/'development_metrics.csv');dev_any=dev[dev.target.eq('ANY_FAULT')]
    if args.select:
        if (out/'frozen.json').exists() or list(out.glob('PUMP-*/*_test.parquet')):
            raise ValueError('Selection must precede freeze and fresh test evaluation')
        selection=json.loads((out/'selection.json').read_text())
        extension=json.loads((out/'head_selection.json').read_text()) if (out/'head_selection.json').exists() else None
        candidates=[f'frozen_{k}/logistic' for k in KINDS]
        baseline=rank_common(dev_any,candidates)
        dump_json(out/'comparison_plan.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            selected_encoder=extension['selected_candidate'] if extension else selection['selected_candidate'],
            selected_logistic_encoder=selection['selected_candidate'],
            added_candidates=extension['added_candidates'] if extension else [],selected_frozen_reference=baseline[0]['candidate'],
            frozen_reference_development_ranking=baseline,selection_source_sha256=sha256(out/'development_metrics.csv'),
            new_model_selection_sha256=sha256(out/'selection.json'),
            all_candidates_reported=True,test_tuning_allowed=False,
            note='Reference selection excludes frozen label-informed encoder, which was unavailable on two families; paired same-encoder comparisons are also reported.'))
        print(json.loads((out/'comparison_plan.json').read_text()));return
    plan=json.loads((out/'comparison_plan.json').read_text());test=pd.read_csv(out/'test_metrics.csv')
    if plan['selection_source_sha256']!=sha256(out/'development_metrics.csv'):raise ValueError('Development metrics changed')
    if plan['new_model_selection_sha256']!=sha256(out/'selection.json'):raise ValueError('Model selection changed')
    selected=plan['selected_encoder'];reference=plan['selected_frozen_reference']
    all_candidates=[f'expanded_{k}/logistic' for k in KINDS+['label_informed','task_encoder']]
    all_candidates += [f'frozen_{k}/logistic' for k in KINDS]+['raw_features/logistic','raw_features/histogram']
    all_candidates += plan.get('added_candidates',[])
    any_test=test[test.target.eq('ANY_FAULT')]
    rankings={split:rank_common(t[t.target.eq('ANY_FAULT')],all_candidates) for split,t in [('development',dev),('fresh_test',test)]}
    # Same candidate set shares the same supported records; secondary type scores have their own common cases.
    type_rank=rank_common(test[~test.target.eq('ANY_FAULT')],all_candidates)
    knn_candidates=[f'expanded_{k}/knn' for k in KINDS+['label_informed','task_encoder']]
    knn_rankings={split:rank_common(t[t.target.eq('ANY_FAULT')],knn_candidates) for split,t in [('development',dev),('fresh_test',test)]}
    counts={c:aggregate_counts(any_test[any_test.candidate.eq(c)]) for c in all_candidates}
    paired=[]
    for k in KINDS+['label_informed']:
        cs=[f'frozen_{k}/logistic',f'expanded_{k}/logistic']
        r=rank_common(any_test,cs);d={v['candidate']:v for v in r}
        if len(d)==2:
            paired.append(dict(encoder=k,frozen_ap=d[cs[0]]['macro_ap'],expanded_ap=d[cs[1]]['macro_ap'],
                               improvement=d[cs[1]]['macro_ap']-d[cs[0]]['macro_ap'],families=d[cs[0]]['families']))
    pairs=any_test[any_test.candidate.isin([selected,reference])].pivot(index=['seed','family','recording'],columns='candidate',values='ap').dropna()
    per_seed=[]
    for seed,g in pairs.groupby(level='seed'):
        per_seed.append(dict(seed=int(seed),selected=float(g[selected].mean()),reference=float(g[reference].mean()),
                             difference=float((g[selected]-g[reference]).mean()),mixed_recordings=len(g)))
    rng=np.random.default_rng(42);seeds=pairs.index.get_level_values('seed').unique().to_numpy();deltas=[]
    for _ in range(2000):
        sample=pd.concat([pairs.xs(s,level='seed',drop_level=False) for s in rng.choice(seeds,len(seeds),replace=True)])
        macro=sample.groupby(level='family')[[selected,reference]].mean().mean()
        deltas.append(float(macro[selected]-macro[reference]))
    interval=np.quantile(deltas,[.025,.975]).tolist()
    summary=dict(selection=plan,rankings=rankings,knn_rankings=knn_rankings,fault_type_test_ranking=type_rank,counts=counts,
                 paired_encoder_comparisons=paired,selected_vs_reference_by_seed=per_seed,
                 paired_seed_bootstrap_95pct_delta_ap=interval,bootstrap_replicates=2000,
                 warning='Five seeds from one public simulator; within known engineering families; no real-world or official event-score claim.')
    dump_json(out/'summary.json',summary)
    dev_lookup={r['candidate']:r['macro_ap'] for r in rankings['development']}
    table=[]
    for r in rankings['fresh_test']:
        c=r['candidate'];table.append(dict(candidate=c,development_ap=dev_lookup[c],fresh_test_ap=r['macro_ap'],
            **counts[c],families=r['families'],mixed_recordings=r['recordings']))
    pd.DataFrame(table).to_csv(out/'comparison.csv',index=False)
    # Primary candidates share all seven families; secondary KNN and paired comparisons are separate below.
    lines=['# Expanded encoding results','',f'Development-selected encoding: **{selected}**.',
           f'Development-selected frozen reference: **{reference}**.','',
           'Average precision (AP) measures ranking of faulty minutes, not accuracy. The same mixed-label recordings support every row below. All-negative recordings remain included in false-positive counts.','',
           '| Candidate | Development AP | Fresh-test AP | Recall | False-positive rate |',
           '|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['candidate']} | {r['development_ap']:.3f} | {r['fresh_test_ap']:.3f} | {r['recall']:.1%} | {r['false_positive_rate']:.1%} |")
    lines += ['', '## Same encoder before and after retraining', '',
              'Both sides receive classifiers fitted on the same augmented training rows. The frozen label-informed comparison has only five supported families; every other row has seven.', '',
              '| Encoder | Frozen test AP | Expanded test AP | Difference | Families |',
              '|---|---:|---:|---:|---:|']
    for r in paired:lines.append(f"| {r['encoder']} | {r['frozen_ap']:.3f} | {r['expanded_ap']:.3f} | {r['improvement']:+.3f} | {r['families']} |")
    lines += ['', '## Nearest-neighbor sensitivity check', '', '| Expanded encoder + 15 neighbors | Development AP | Fresh-test AP |', '|---|---:|---:|']
    kd={r['candidate']:r['macro_ap'] for r in knn_rankings['development']}
    for r in knn_rankings['fresh_test']:lines.append(f"| {r['candidate']} | {kd[r['candidate']]:.3f} | {r['macro_ap']:.3f} |")
    lines += ['',f'Test AP averages recordings within each engineering family, then weights families equally. Recall and false-positive rate use pooled minute counts at thresholds chosen on development F1. They are different summaries.',
              '',f'Paired seed-bootstrap 95% interval for selected-minus-reference AP: **[{interval[0]:.3f}, {interval[1]:.3f}]** (2,000 resamples, only five independent simulator seeds).',
              '', '## What changed','',
              '- Reused 108 additional training recordings and the original seven training pumps’ first 60% intervals.',
              '- Kept entire recordings together. A family encoder is shared across simulated variants derived from that pump template; different families still have different coordinate systems.',
              '- Refit PCA, kernel PCA, autoencoder, denoising autoencoder, temporal autoencoder and label-informed autoencoder.',
              '- Added a 32D task-trained encoder: the same MLP compresses readings; auxiliary fault heads teach it which information to retain, while a small reconstruction loss keeps general information.',
              '- Retrained identical classifiers on both the published frozen encoders and the expanded encoders. The comparison tests the expanded pipeline, including training-supported feature coverage and checkpoint selection; it does not isolate data volume alone.',
              '- Kept full-feature logistic regression and boosted trees as controls for information lost during compression.','',
              '## Boundaries','',
              'Development includes the earlier synthetic holdout, because it had already been examined. The fresh test set contains 35 newly generated three-day recordings from five unused seeds, created after model and threshold freeze. No test results changed training or thresholds.',
              '', 'These are minute-level classification tests within known simulated engineering families. They are not the hackathon’s event-alert/lead-time score, an unseen-family test, or evidence from real pumps. Original PUMP-08–12 were not fitted in this new experiment.',
              '', 'The published branch’s AP of 0.770 came from a different two-pump chronological validation experiment and cannot be directly compared with this table. Here, each family has both positive and negative development coverage.',
              '', 'No-event means no eligible annotation, not guaranteed health. Predetectable and unobservable overlaps are excluded. Missing observations remain masked. Sensor snapshots use the generator’s existing waveform path, which does not reproduce all channel sensor corruption; this is a remaining simulation limitation.',
              '', 'See `results/summary.json`, `comparison.csv`, `development_metrics.csv`, `test_metrics.csv`, saved prediction Parquets and `frozen.json` for the underlying results.']
    coverage=pd.read_csv(out/'test_target_coverage.csv')
    unsupported=coverage[~coverage.supported & coverage.positive_minutes.gt(0)]
    lines += ['', '## Unsupported test fault types', '',
              'The model was selected for ANY_FAULT. Type-level scores only cover heads learnable from training; the following positive types lack a head in the corresponding family. They remain positive for ANY_FAULT.', '',
              '| Family | Type | Eligible positive minutes |','|---|---|---:|']
    for r in unsupported.itertuples():lines.append(f'| {r.family} | {r.target} | {r.positive_minutes} |')
    rec=pd.read_csv(out/'test_reconstruction.csv').groupby('variant')[['nmse','skill_vs_train_mean']].mean()
    lines += ['', '## Reconstruction', '',
              'Equal-family mean error over the same train-scaled input features. Lower NMSE is better. The task encoder is judged against un-clipped targets even though its neural input is clipped. Reconstruction and fault classification answer different questions.', '',
              '| Expanded encoder | NMSE | Skill versus training mean |','|---|---:|---:|']
    for name,r in rec.iterrows():lines.append(f"| {name} | {r['nmse']:.3f} | {r['skill_vs_train_mean']:.3f} |")
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ordered=sorted(table,key=lambda r:r['fresh_test_ap'])
    fig,ax=plt.subplots(figsize=(12,8))
    positions=np.arange(len(ordered));colors=['#dd6b20' if r['candidate']==selected else '#356b8c' for r in ordered]
    ax.barh(positions,[r['fresh_test_ap'] for r in ordered],color=colors,height=.65)
    ax.scatter([r['development_ap'] for r in ordered],positions,marker='|',s=90,color='#111827',label='Development AP')
    ax.set_yticks(positions,[r['candidate'] for r in ordered]);ax.set_xlim(0,1)
    ax.set_xlabel('Average precision — equal weight per engineering family')
    ax.set_title('Fresh simulated recordings: fault classification\nOrange = candidate selected before fresh test; black marks = development')
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='lower right')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    print(pd.DataFrame(table)[['candidate','development_ap','fresh_test_ap','recall','false_positive_rate']].to_string(index=False))


if __name__=='__main__':main()
