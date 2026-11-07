import numpy as np
import torch
import models


def test_masked_loss_ignores_missing_targets():
    prediction=torch.tensor([[2.,1000.]])
    target=torch.tensor([[1.,-1000.]])
    mask=torch.tensor([[True,False]])
    assert models.masked_loss(prediction,target,mask).item()==1.


def test_encoder_reload_and_batch_independence(tmp_path):
    torch.manual_seed(3)
    net=models.Autoencoder(4,latent=3,temporal=True,length=3).eval()
    x=torch.randn(6,3,8)
    with torch.no_grad(): expected=net.encode(x).numpy()
    torch.save(net.state_dict(),tmp_path/'weights.pt')
    other=models.Autoencoder(4,latent=3,temporal=True,length=3).eval()
    other.load_state_dict(torch.load(tmp_path/'weights.pt',weights_only=True))
    with torch.no_grad(): actual=other.encode(x[:2]).numpy()
    np.testing.assert_allclose(actual,expected[:2],atol=1e-6)
