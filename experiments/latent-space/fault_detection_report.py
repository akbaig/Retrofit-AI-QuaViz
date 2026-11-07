"""Render the limited but honest fault-detection comparison."""
import os,tempfile
os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),'latent-space-matplotlib'))
import json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import ROOT
from fault_detection import ANY,READOUTS
NAMES={'pca':'PCA','kernel_pca':'Kernel PCA','autoencoder':'Autoencoder','denoising_ae':'Denoising AE','temporal_ae':'Temporal AE','label_informed':'Label-informed AE'}
OUT=ROOT/'results/fault_detection'


def format_table(ranking):
    return '\n'.join(['| Encoder | Validation AP |','|---|---:|']+[f'| {NAMES[r["model"]]} | {r["macro_ap"]:.3f} |' for r in ranking])


def main():
    selection=json.loads((OUT/'selection.json').read_text());decisions=selection['decisions'];metrics=pd.read_csv(OUT/'metrics.csv');coverage=pd.read_csv(OUT/'label_coverage.csv')
    primary=decisions['any_fault_logistic'];winner=primary['winner'];ranking=primary['ranking']
    test=metrics[metrics.split.eq('test')&metrics.readout.eq('logistic')&metrics.target.eq(ANY)&metrics.asset_id.isin(['PUMP-01','PUMP-09'])]
    primary_val=metrics[metrics.split.eq('validation')&metrics.readout.eq('logistic')&metrics.target.eq(ANY)&metrics.asset_id.isin(['PUMP-01','PUMP-09'])]
    winner_test=test[test.model.eq(winner)].set_index('asset_id')
    detail=['| Pump | Encoder | Validation AP | Test recall | Test false-positive rate |','|---|---|---:|---:|---|']
    for r in primary_val.itertuples():
        te=test[test.asset_id.eq(r.asset_id)&test.model.eq(r.model)].iloc[0]
        detail.append(f'| {r.asset_id} | {NAMES[r.model]} | {r.ap:.3f} | {te.recall:.1%} | Unavailable: no negative examples |')
    text=f'''# Which encoding detects faults best?

**There is no confirmed overall winner.** The provisional validation leader for “fault present vs no labeled event” is **{NAMES[winner]} + logistic regression**, with mean average precision **{ranking[0]['macro_ap']:.3f}**. This comparison is supported by only **PUMP-01 and PUMP-09**. In the test period, that detector recognizes **{winner_test.loc['PUMP-01','recall']:.1%}** of the annotated faulty minutes on PUMP-01 and **{winner_test.loc['PUMP-09','recall']:.1%}** on PUMP-09. Neither test period has negative examples, so test false-alarm rates and a meaningful test ranking cannot be established.

The 0–1 AP score is a ranking measure, not classification accuracy or “percentage of faults detected.” Higher is better. The recall values above describe faulty **minutes**, not independent events or advance warning.

## What was tested

All existing 32D encoders were frozen. Each available pump/encoder received identical downstream classifiers trained only on known labels in its first 60% interval:

- **Primary readout:** logistic regression, standardized using the classifier's training rows, C=1, balanced class weights, max_iter=2000, seed 42.
- **Sensitivity check:** a 15-nearest-neighbor classifier, uniform voting, with the same training-only standardization.
- **Primary target:** ANY_FAULT, whether at least one observable, detectable fault is annotated.
- **Secondary targets:** individual fault types that appear in the training vocabulary and have both positive and negative training examples.

The 60/20/20 chronological split and 120-minute heldout warmup exclusions are unchanged. Pre-detectable/unobservable overlaps and unlabeled minutes are excluded. `NO_LABELED_EVENT` is a negative for this annotation-based experiment, not a guarantee of physical health. No synthetic healthy examples were created. No different pump's labels or coordinates were pooled into a model.

For every trained target, the validation F1-maximizing threshold was saved before testing. If validation has only one class or no rows, a fixed 0.5 threshold is recorded as an unevaluated fallback. Encoder ranking uses validation AP, not thresholded F1. Every candidate sees exactly the same evaluable validation pump/target cases; target scores are averaged within pump and then equally across pumps. A durable `selection.json` was written before calculating test predictions and metrics. Training vocabularies exclude future-only faults; the full vocabulary is used only to report coverage.

In total **192 downstream classifiers** were fitted without changing any of the 67 encoders. No convergence warnings were recorded.

## Main result: any fault present

{format_table(ranking)}

Only PUMP-01 and PUMP-09 have both classes in training and scored validation for this task across all six encoders. The other pumps do not meet that requirement. The mean validation no-skill AP baseline, equal to mean positive prevalence on these two pumps, is {primary_val.groupby('asset_id').prevalence.first().mean():.3f}.

The lead over the next candidate is small: {ranking[0]['macro_ap']-ranking[1]['macro_ap']:.3f} AP. Adjacent minutes are highly correlated and only two pumps contribute, so this is not a robust estimate of future performance. Different encoder types win on the two individual pumps; a single mean hides that variation.

{chr(10).join(detail)}

High recall alone cannot establish a good detector: always predicting “fault” would achieve 100% recall in these all-positive test periods. Conversely, the label-informed model's low PUMP-01 test recall is direct evidence that its validation lead does not imply reliable chronological generalization.

## The result depends on the detection question and classifier

For ANY_FAULT with the 15-neighbor classifier, the leader is **{NAMES[decisions['any_fault_knn']['winner']]}**, with AP {decisions['any_fault_knn']['ranking'][0]['macro_ap']:.3f}.

{format_table(decisions['any_fault_knn']['ranking'])}

For individual known fault types with logistic regression, **{NAMES[decisions['fault_types_logistic']['winner']]}** leads with AP {decisions['fault_types_logistic']['ranking'][0]['macro_ap']:.3f}. That is a different target and a different subset: PUMP-01 strainer clogging, PUMP-02 sensor drift, and PUMP-06 cavitation/strainer clogging. There are four pump/type cases over three pumps, weighted equally by pump.

{format_table(decisions['fault_types_logistic']['ranking'])}

With the nearest-neighbor readout on those type-specific cases, **{NAMES[decisions['fault_types_knn']['winner']]}** leads with AP {decisions['fault_types_knn']['ranking'][0]['macro_ap']:.3f}. Therefore the data do not support calling any encoder universally best for fault detection.

## Why test data cannot confirm the ranking

Every test target for which a classifier could actually be trained has only positive labels or only negative labels, after ambiguity and warmup exclusions. We deliberately mark AP/AUROC **not evaluable for ranking** for those cases. A library can return AP=1 for all-positive labels, but that would not demonstrate the ability to separate faulty from non-faulty observations. We report recall when positives exist and false-positive rate when negatives exist, without combining them across incompatible cases into a misleading headline.

PUMP-02 does have both faulty and no-event test minutes, but its ANY_FAULT training labels are all positive, so no supervised two-class detector can be learned there under the fixed within-pump protocol. PUMP-05 has no unambiguous heldout scoring minutes. PUMP-07 has no positive training examples. PUMP-10–12 have no public labels. Several later fault types never appear in their pump's training labels. These are data-coverage limitations, not missing model runs.

## Interpretation

- **Compression:** the preceding experiment favored PCA on 11 pumps and the ordinary AE on PUMP-08.
- **Binary fault detection:** the label-informed AE is the provisional leader on the prespecified logistic validation comparison; test behavior is inconsistent and false-alarm performance is unmeasurable on those test windows.
- **Known fault recognition:** PCA leads the separate logistic validation comparison, with sparse coverage and no mixed-class test confirmation.

A defensible final winner needs later labeled periods containing both fault-free and faulty behavior, ideally multiple independent incidents/pumps. Randomly shuffling adjacent minutes into train/test would disguise the current limitations rather than resolve them. We did not change the split or fit new encoders to improve the measured ranking.

## Artifacts and boundaries

`results/fault_detection/` contains saved readout models and thresholds, per-case validation/test metrics, full heldout predictions, label coverage, input hashes, selection and verification. `fault_detection.html` provides a compact browser view. `fault_detection.py` reruns the fixed experiment; `verify_fault_detection.py` checks the saved outputs; `fault_detection_report.py` rebuilds this report and charts.

This is a diagnostic on synthetic data and minute annotations, not the official hackathon event-alert/lead-time score, real-world fault detection, or an unseen-pump benchmark. Existing neural encoder checkpoints had already been selected using validation reconstruction. These test intervals were explored earlier, so this is not a new untouched final benchmark. The detector settings and evaluation policy were fixed before obtaining classifier scores and are recorded in [the recorded protocol](../../docs/superpowers/plans/2026-11-07-fault-detection-benchmark.md).

Definitions: [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) and [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
'''
    (ROOT/'FAULT_DETECTION_REPORT.md').write_text(text)
    fig,ax=plt.subplots(figsize=(10,5));values=[r['macro_ap'] for r in ranking];labels=[NAMES[r['model']] for r in ranking]
    ax.barh(labels,values,color=['#b06c3a']+['#428ca3']*5);ax.invert_yaxis();ax.set_xlim(0,1);ax.spines[['top','right']].set_visible(False)
    for i,value in enumerate(values):ax.text(value+.012,i,f'{value:.3f}',va='center')
    ax.set_xlabel('Validation average precision · higher is better · not accuracy')
    fig.suptitle('Fault detection: a provisional leader',x=.06,ha='left',fontsize=19,fontweight='bold')
    fig.text(.06,.9,'Same logistic classifier on each frozen encoding · PUMP-01 and PUMP-09 only',fontsize=11,color='#566574')
    fig.text(.06,.025,'Test periods lack negative examples. This ranking is not confirmed on test data.',fontsize=10,color='#9a5d2d')
    fig.tight_layout(rect=[0,.06,1,.88]);fig.savefig(OUT/'validation_comparison.png',dpi=170);plt.close(fig)
    def htmlrows(rows):return ''.join(f'<tr><td>{html.escape(NAMES[r["model"]])}</td><td>{r["macro_ap"]:.3f}</td></tr>' for r in rows)
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fault detection comparison</title><style>
body{{font-family:system-ui;background:#f1f4f6;color:#183043;margin:0}}main{{max-width:1100px;padding:32px;margin:auto}}h1{{font-size:34px;letter-spacing:-1px;margin-bottom:12px}}h2{{font-size:20px}}p{{line-height:1.6;color:#526675}}.card{{background:white;border:1px solid #d4dfe6;border-radius:12px;padding:24px;margin:18px 0}}.warning{{border-left:5px solid #c18642;background:#fff8ed}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #e5eaef;text-align:left;padding:10px}}th:last-child,td:last-child{{text-align:right}}img{{width:100%;height:auto}}a{{color:#247b9c}}.pill{{font-size:12px;background:#e1edf2;padding:6px 12px;border-radius:20px}}@media(max-width:760px){{main{{padding:16px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><span class="pill">Frozen 32D encoders · 192 detector fits</span><h1>Which encoding detects faults best?</h1><div class="card warning"><h2>No confirmed overall winner yet.</h2><p><strong>{NAMES[winner]} + a simple linear classifier</strong> leads the main validation comparison. But only two pumps support that comparison, and their test periods contain no negative examples. We cannot establish the test false-alarm rate.</p><p>On PUMP-01, this detector caught only <strong>{winner_test.loc['PUMP-01','recall']:.1%}</strong> of faulty test minutes. On PUMP-09 it caught <strong>{winner_test.loc['PUMP-09','recall']:.1%}</strong>. High recall alone would not show a good detector when every test minute is faulty.</p></div>
<div class="card"><h2>“Is any fault present?”</h2><img src="fault_detection/validation_comparison.png" alt="Six-encoder validation average-precision comparison"><p>Training uses earlier observations. Validation compares faulty with no-event minutes on PUMP-01 and PUMP-09. Each model gets the same classifier and examples. Higher AP means better ranking; 0.77 does not mean 77% accuracy.</p></div>
<div class="grid"><section class="card"><h2>Main comparison · linear classifier</h2><table><thead><tr><th>Encoding</th><th>Validation AP</th></tr></thead><tbody>{htmlrows(ranking)}</tbody></table><p>Two pumps, equal weight. The label-informed lead is small and does not establish stability across new data.</p></section><section class="card"><h2>Check · nearest neighbors</h2><table><thead><tr><th>Encoding</th><th>Validation AP</th></tr></thead><tbody>{htmlrows(decisions['any_fault_knn']['ranking'])}</tbody></table><p>The winner changes when the classifier changes. There is no universal winning representation.</p></section></div>
<section class="card"><h2>“Which known fault is present?”</h2><table><thead><tr><th>Encoding</th><th>Validation AP</th></tr></thead><tbody>{htmlrows(decisions['fault_types_logistic']['ranking'])}</tbody></table><p>This separate test uses four pump/fault cases over three pumps. PCA leads here. It does not measure the same target or use the same cases as the binary fault comparison.</p></section>
<section class="card"><h2>What we can say</h2><p>PCA was the strongest reconstruction baseline. The label-informed AE is a promising candidate for binary fault detection with a linear classifier, but its test performance is inconsistent. More labeled test periods with both faulty and no-event behavior are needed to confirm a winner.</p><p>No ambiguous labels were treated as normal, no future labels trained the classifiers, and no encoder was refitted. The original chronological split remains intact.</p><a href="explorer.html">Back to embedding explorer</a> · <a href="fault_detection/metrics.csv">All detector scores</a> · <a href="fault_detection/label_coverage.csv">Label coverage</a> · <a href="fault_detection/selection.json">Frozen validation decision</a></section></main></body></html>'''
    (ROOT/'results/fault_detection.html').write_text(page)
    print('Wrote fault detection report, chart and browser page',flush=True)
if __name__=='__main__':main()
