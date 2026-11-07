"""Independent-recording protocol; event annotations never become input features."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'latent-space'
sys.path.insert(0, str(LEGACY))
from fault_detection import target_vector, training_targets

FAMILIES = [f'PUMP-{i:02d}' for i in range(1,8)]
NEW_SEEDS = [202611071,202611073,202611079,202611083,202611089]
TRAIN_ROLES = ['train','train_balanced','train_crossed','train_sensor_crossed']


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def dump_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')


def fit_indices(meta, stride=5):
    return np.flatnonzero(meta.role.eq('train') & meta.minute_index.ge(120) & meta.minute_index.mod(stride).eq(0))


def check_roles(meta):
    if meta.groupby('recording').role.nunique().gt(1).any():
        raise ValueError('A recording crosses roles')
    seeds = meta.dropna(subset=['seed'])
    if seeds.groupby('seed').role.nunique().gt(1).any():
        raise ValueError('A seed crosses roles')


def make_targets(labels, names=None):
    names = training_targets(labels) if names is None else list(names)
    return np.column_stack([target_vector(labels,n) for n in names]), names


def rank_common(rows, candidates):
    valid = rows[rows.ap.notna() & rows.candidate.isin(candidates)].copy()
    keys = ['family','recording']
    if 'target' in valid: keys.append('target')
    counts = valid.groupby(keys).candidate.nunique()
    cases = counts[counts.eq(len(candidates))].index
    valid = valid[pd.MultiIndex.from_frame(valid[keys]).isin(cases)]
    ranking = []
    for name,g in valid.groupby('candidate'):
        per_recording = g.groupby(['family','recording']).ap.mean()
        per_family = per_recording.groupby(level='family').mean()
        ranking.append(dict(candidate=name,macro_ap=float(per_family.mean()),
                            families=len(per_family),recordings=len(per_recording),cases=len(g)))
    return sorted(ranking,key=lambda r:(-r['macro_ap'],r['candidate']))


def source_paths(root):
    return [root/n for n in ['assets.csv','features_1min.parquet','operating_context.csv','vibration_snapshots.parquet']]


def prepare_sources(sources, out):
    """sources: (path, role, original_bool); cache split by template family."""
    from data import candidate_features
    from features import build_features
    from label_informed import annotate_labels
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    paths = [Path(__file__), LEGACY/'data.py', LEGACY/'label_informed.py', *sorted((LEGACY/'vendor').glob('*.py'))]
    for path,role,original in sources:
        paths += source_paths(path)
        paths += [path/('events_train.csv' if original else 'events.csv')]
        if not original: paths.append(path/'manifest.json')
    hashes = {str(p.resolve()):sha256(p) for p in paths}
    signature = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    marker = out/'preparation.json'
    if marker.exists():
        if json.loads(marker.read_text())['signature'] != signature:
            raise ValueError('Preparation sources changed; use a fresh output directory')
        return
    frames = {family:[] for family in FAMILIES}; metadata = {family:[] for family in FAMILIES}
    for path,role,original in sources:
        assets = pd.read_csv(path/'assets.csv')
        if original: assets = assets[assets.asset_id.isin(FAMILIES)]
        minutes = pd.read_parquet(path/'features_1min.parquet')
        context = pd.read_csv(path/'operating_context.csv')
        shots = pd.read_parquet(path/'vibration_snapshots.parquet')
        events = pd.read_csv(path/('events_train.csv' if original else 'events.csv'))
        seeds = {} if original else {r['asset_id']:r['seed'] for r in json.loads((path/'manifest.json').read_text())['runs']}
        for _, asset in assets.iterrows():
            family = asset.asset_id if original else asset.template_asset_id
            if family not in frames: raise ValueError(f'Unexpected family {family}')
            meta,frame = build_features(minutes,asset,context,shots)
            # Preserve the original within-pump training boundary; later original rows are unused.
            if original:
                n = int(len(meta)*.6); meta=meta.iloc[:n].copy(); frame=frame.iloc[:n].copy()
            labels = annotate_labels(meta,events,True)
            meta['fault_label'] = labels.fault_label.to_numpy()
            meta['label_eligible'] = labels.label_eligible.to_numpy()
            meta['family'] = family; meta['recording'] = asset.asset_id; meta['role'] = role
            meta['minute_index'] = np.arange(len(meta)); meta['seed'] = seeds.get(asset.asset_id,np.nan)
            meta['source_role'] = path.name
            frames[family].append(frame[candidate_features(frame)])
            metadata[family].append(meta)
            print('prepared',role,asset.asset_id,len(meta),flush=True)
    all_meta = []
    for family in FAMILIES:
        meta = pd.concat(metadata[family],ignore_index=True)
        check_roles(meta); all_meta.append(meta)
        pd.concat(frames[family],ignore_index=True).to_parquet(out/f'{family}_features.parquet',index=False)
        meta.to_parquet(out/f'{family}_metadata.parquet',index=False)
    meta = pd.concat(all_meta,ignore_index=True); check_roles(meta)
    coverage = meta.assign(scored=meta.minute_index.ge(120) & meta.label_eligible,
                           positive=meta.fault_label.ne('NO_LABELED_EVENT') & meta.label_eligible)
    summary = coverage.groupby(['role','family','recording','source_role'],dropna=False).agg(
        rows=('recording','size'),eligible=('scored','sum'),positive=('positive','sum')).reset_index()
    summary.to_csv(out/'coverage.csv',index=False)
    dump_json(marker,dict(signature=signature,source_hashes=hashes,rows=len(meta),
                          recordings=int(meta.recording.nunique()),features_contain_labels=False))


def load_family(cache,family):
    cache=Path(cache)
    return pd.read_parquet(cache/f'{family}_metadata.parquet'),pd.read_parquet(cache/f'{family}_features.parquet')
