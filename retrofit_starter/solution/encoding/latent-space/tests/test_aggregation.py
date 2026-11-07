import json
import numpy as np
import pandas as pd
import run


def test_partial_rerun_keeps_other_completed_pumps(tmp_path):
    for asset,value in [('PUMP-01',1.),('PUMP-02',2.)]:
        dest=tmp_path/asset;dest.mkdir()
        pd.DataFrame([{'asset_id':asset,'model':'pca','validation_nmse':value}]).to_csv(dest/'comparison.csv',index=False)
        (dest/'selection.json').write_text(json.dumps({'selected_model':'pca'}))
        frame=pd.DataFrame({'asset_id':[asset],**{f'z{i:02d}':[value] for i in range(32)}})
        frame.to_parquet(dest/'selected_embeddings.parquet',index=False)
    run.aggregate_results(tmp_path)
    assert set(pd.read_csv(tmp_path/'selected_models.csv').asset_id)=={'PUMP-01','PUMP-02'}
    assert np.load(tmp_path/'selected_embeddings.npy').shape==(2,32)
    # Updating just one local result must retain the other pump in global files.
    pd.DataFrame([{'asset_id':'PUMP-02','model':'pca','validation_nmse':.5}]).to_csv(tmp_path/'PUMP-02'/'comparison.csv',index=False)
    run.aggregate_results(tmp_path)
    result=pd.read_csv(tmp_path/'model_comparison.csv').set_index('asset_id')
    assert len(result)==2 and result.loc['PUMP-02','validation_nmse']==.5
