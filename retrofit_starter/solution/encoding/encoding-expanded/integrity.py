"""Byte-level receipts for prepared observations and frozen fresh-test generation."""
import json
from pathlib import Path
import pandas as pd
from protocol import FAMILIES,sha256,dump_json


def verify_data_seal(path):
    seal=json.loads(Path(path).read_text())
    changed=[p for p,h in seal['hashes'].items() if sha256(p)!=h]
    if changed:raise ValueError(f'Prepared data changed: {changed}')
    return seal


def seal_prepared_data(cache,destination):
    cache=Path(cache);marker=json.loads((cache/'preparation.json').read_text())
    # Recheck original observation and target bytes before sealing this run's generated cache.
    for p,h in marker['source_hashes'].items():
        if Path(p).suffix in {'.csv','.parquet','.json'} and sha256(p)!=h:
            raise ValueError(f'Original data changed: {p}')
    role_assignments=[]
    for family in FAMILIES:
        meta=pd.read_parquet(cache/f'{family}_metadata.parquet')
        frame=pd.read_parquet(cache/f'{family}_features.parquet')
        if len(meta)!=len(frame):raise ValueError('Prepared row alignment changed')
        role_assignments += meta[['recording','role','source_role','seed']].drop_duplicates().fillna('').to_dict('records')
    files=[cache/'preparation.json',*sorted(cache.glob('*.parquet'))]
    dump_json(destination,dict(hashes={str(p.resolve()):sha256(p) for p in files},
        role_assignments=role_assignments,scope='Original sources rechecked; prepared outputs sealed before fresh-test generation'))


def validate_test_dataset(root,freeze_path,old_meta,generator,config,families=FAMILIES):
    root=Path(root);frozen=json.loads(Path(freeze_path).read_text());manifest=json.loads((root/'manifest.json').read_text())
    seeds=frozen['test_seeds']
    if manifest['seeds']!=seeds:raise ValueError('Unexpected test seeds')
    if manifest['frozen_pipeline_manifest_sha256']!=sha256(freeze_path):raise ValueError('Test freeze digest mismatch')
    required={'assets.csv','features_1min.parquet','operating_context.csv','vibration_snapshots.parquet','events.csv'}
    if set(manifest['artifacts'])!=required:raise ValueError('Unexpected artifact inventory')
    for name,digest in manifest['artifacts'].items():
        if sha256(root/name)!=digest:raise ValueError(f'Test artifact changed: {name}')
    for p,key in [(generator,'generator_sha256'),(config,'source_config_sha256')]:
        p=Path(p)
        if manifest[key]!=sha256(p) or frozen['hashes'].get(str(p.resolve()))!=sha256(p):
            raise ValueError(f'Test generator/config differs from frozen source: {p}')
    runs=pd.DataFrame(manifest['runs'])
    expected={(seed,family) for seed in seeds for family in families}
    actual=list(zip(runs.seed,runs.template_asset_id))
    if set(actual)!=expected or len(actual)!=len(expected) or not runs.asset_id.is_unique:
        raise ValueError('Test seed/family coverage is incomplete or duplicated')
    if set(runs.seed)&set(old_meta.seed.dropna()):raise ValueError('Test/train seed overlap')
    if set(runs.asset_id)&set(old_meta.recording):raise ValueError('Test/train recording overlap')
    assets=pd.read_csv(root/'assets.csv')
    if not assets.asset_id.is_unique or set(assets.asset_id)!=set(runs.asset_id):raise ValueError('Test asset inventory mismatch')
    expected_family=dict(zip(runs.asset_id,runs.template_asset_id))
    if not assets.apply(lambda r:expected_family[r.asset_id]==r.template_asset_id,axis=1).all():
        raise ValueError('Test family assignment mismatch')
    return manifest
