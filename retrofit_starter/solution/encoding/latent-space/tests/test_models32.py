import numpy as np
import models


def test_pca_32_reloads_and_reconstructs_without_test_fit():
    rng=np.random.default_rng(3)
    x=rng.normal(size=(120,48)).astype('float32');m=np.ones_like(x,dtype=bool)
    train=np.arange(80);validation=np.arange(80,100)
    bundle=models.fit_encoder('pca',x,m,train,validation,latent=32)
    z,recon=models.encode_reconstruct(bundle,x,m)
    assert z.shape==(120,32) and recon.shape==x.shape
    changed=x.copy();changed[100:]*=1000
    other=models.fit_encoder('pca',changed,m,train,validation,latent=32)
    np.testing.assert_allclose(bundle['pca'].components_,other['pca'].components_)


def test_sequences_do_not_cross_pump_boundary_or_look_forward():
    seq=models.sequence_indices(np.array(['A']*3+['B']*3),length=3,step=1)
    assert seq.tolist()==[[0,0,0],[0,0,1],[0,1,2],[3,3,3],[3,3,4],[3,4,5]]
