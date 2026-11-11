import numpy as np
import torch

from forecaster import OnnxForecaster
from model import UNet, export_onnx, predict_mm


def test_onnx_export_matches_torch_in_opencv(tmp_path):
    torch.manual_seed(0)
    net = UNet(base=8, depth=3, norm="batch")
    # give BatchNorm non-trivial running stats, as after training
    net.train()
    with torch.no_grad():
        for _ in range(3):
            net(torch.rand(4, 12, 64, 64))
    net.eval()
    path = export_onnx(net, tmp_path / "m.onnx", grid=64)
    inputs = (np.random.default_rng(0).random((12, 64, 64)) * 5).astype(np.float32)
    ref = predict_mm(net, inputs, torch.device("cpu"))
    out = OnnxForecaster(path).predict(inputs)
    assert out.shape == ref.shape == (12, 64, 64)
    assert np.allclose(out, ref, rtol=1e-3, atol=1e-4)
