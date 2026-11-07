from pathlib import Path
import json
import pandas as pd
import pytest
from protocol import sha256,dump_json
from integrity import validate_test_dataset,verify_data_seal


def test_role_assignment_changes_cache_identity(tmp_path):
    from protocol import preparation_signature
    data={'same_input':'same_digest'}
    a=preparation_signature(data,[(tmp_path,'train',False)])
    b=preparation_signature(data,[(tmp_path,'development',False)])
    assert a!=b


def test_prepared_cache_mutation_is_rejected(tmp_path):
    from protocol import verify_prepared_cache
    p=tmp_path/'prepared.parquet';p.write_bytes(b'original')
    dump_json(tmp_path/'preparation.json',{'output_hashes':{'prepared.parquet':sha256(p)}})
    verify_prepared_cache(tmp_path)
    p.write_bytes(b'modified')
    with pytest.raises(ValueError,match='changed'):verify_prepared_cache(tmp_path)


def test_freeze_reentry_is_rejected_before_any_mutation(tmp_path):
    from argparse import Namespace
    from run_expanded import freeze
    (tmp_path/'frozen.json').write_text('existing freeze')
    (tmp_path/'selection.json').write_text('existing selection')
    with pytest.raises(ValueError,match='Already frozen'):
        freeze(Namespace(output=tmp_path,cache=tmp_path/'missing'))
    assert (tmp_path/'selection.json').read_text()=='existing selection'


def test_changed_prepared_data_is_rejected(tmp_path):
    p=tmp_path/'features.parquet';p.write_bytes(b'original observations')
    seal=tmp_path/'seal.json';dump_json(seal,{'hashes':{str(p):sha256(p)}})
    verify_data_seal(seal)
    p.write_bytes(b'changed observations')
    with pytest.raises(ValueError,match='changed'):verify_data_seal(seal)


def fixture_data(tmp_path):
    generator=tmp_path/'generator.py';generator.write_text('generator')
    config=tmp_path/'config.yaml';config.write_text('config')
    freeze=tmp_path/'frozen.json';dump_json(freeze,{'hashes':{str(generator):sha256(generator),str(config):sha256(config)},'test_seeds':[11]})
    assets=pd.DataFrame([{'asset_id':'new-1','template_asset_id':'p1'}]);assets.to_csv(tmp_path/'assets.csv',index=False)
    for n in ['features_1min.parquet','operating_context.csv','vibration_snapshots.parquet','events.csv']:(tmp_path/n).write_text('data')
    manifest={'seeds':[11],'runs':[{'asset_id':'new-1','template_asset_id':'p1','seed':11}],
              'frozen_pipeline_manifest_sha256':sha256(freeze),'generator_sha256':sha256(generator),'source_config_sha256':sha256(config),
              'artifacts':{n:sha256(tmp_path/n) for n in ['assets.csv','features_1min.parquet','operating_context.csv','vibration_snapshots.parquet','events.csv']}}
    dump_json(tmp_path/'manifest.json',manifest)
    old=pd.DataFrame([{'recording':'old','seed':3}])
    return freeze,generator,config,old


def test_manifest_does_not_hide_changed_observation_bytes(tmp_path):
    freeze,generator,config,old=fixture_data(tmp_path)
    validate_test_dataset(tmp_path,freeze,old,generator,config,families=['p1'])
    (tmp_path/'features_1min.parquet').write_text('tampered')
    with pytest.raises(ValueError,match='artifact'):
        validate_test_dataset(tmp_path,freeze,old,generator,config,families=['p1'])


def test_fresh_generation_cannot_reuse_training_seed(tmp_path):
    freeze,generator,config,old=fixture_data(tmp_path);old['seed']=11
    with pytest.raises(ValueError,match='seed overlap'):
        validate_test_dataset(tmp_path,freeze,old,generator,config,families=['p1'])


def test_missing_seed_family_pair_is_rejected(tmp_path):
    freeze,generator,config,old=fixture_data(tmp_path)
    with pytest.raises(ValueError,match='coverage'):
        validate_test_dataset(tmp_path,freeze,old,generator,config,families=['p1','p2'])
