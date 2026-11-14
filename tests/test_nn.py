import numpy as np
import pytest

from nn import NeuralNet


@pytest.mark.parametrize("mode", ["sigmoid", "softmax"])
def test_gradients(mode):
    rng = np.random.default_rng(123)
    net = NeuralNet(4, 3, 5, 2, output=mode)
    net.weights = [rng.normal(0, 0.3, w.shape) for w in net.weights]
    net.biases = [rng.normal(0, 0.1, b.shape) for b in net.biases]
    x = rng.normal(size=(6, 4))
    y = np.array([0, 1, 2, 0, 1, 2])
    acts, zs = net.forward(x)
    gw, gb = net.backprop(acts, zs, y)
    for param, grad in zip(net.weights + net.biases, gw + gb):
        numeric = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + 1e-5
            plus = net.loss(x, y)
            param[idx] = original - 1e-5
            minus = net.loss(x, y)
            param[idx] = original
            numeric[idx] = (plus - minus) / 2e-5
        relative = np.linalg.norm(numeric - grad) / (np.linalg.norm(numeric) + np.linalg.norm(grad))
        assert relative < 1e-5


def test_softmax_stable():
    net = NeuralNet(4, 3, 5, 2, output="softmax")
    for w in net.weights:
        w *= 1e6
    for x in (np.full((6, 4), 1e6), np.full(4, 1e6)):
        out = net.forward(x)[0][-1]
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out.sum(axis=-1), 1)


@pytest.mark.parametrize("mode", ["sigmoid", "softmax"])
def test_roundtrip(tmp_path, mode):
    net = NeuralNet(4, 3, 5, 2, output=mode)
    x = np.ones((2, 4))
    path = tmp_path / "weights.npz"
    net.save(path)
    loaded = NeuralNet.load(path)
    assert loaded.output == mode
    for a, b in zip(net.weights + net.biases, loaded.weights + loaded.biases):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(net.forward(x)[0][-1], loaded.forward(x)[0][-1])


def test_legacy(tmp_path):
    net = NeuralNet(4, 3, 5, 2)
    arrays = {"dims": np.array([4, 3, 5, 2])}
    for i, (w, b) in enumerate(zip(net.weights, net.biases)):
        arrays[f"w{i}"] = w
        arrays[f"b{i}"] = b
    path = tmp_path / "legacy.npz"
    np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]
    assert NeuralNet.load(path).output == "sigmoid"


def test_invalid_output():
    with pytest.raises(ValueError):
        NeuralNet(4, 3, 5, 2, output="invalid")


def test_softmax_loss_extreme_logits():
    # One layer, no hidden: logits are exactly w @ x, here [0, 100].
    net = NeuralNet(1, 2, 1, 0, output="softmax")
    net.weights = [np.array([[0.0], [100.0]])]
    net.biases = [np.zeros(2)]
    x = np.ones((1, 1))
    assert net.loss(x, [0]) == pytest.approx(100.0)

    # The loss slope w.r.t. the target's logit weight must match backprop (-1).
    eps = 1e-4
    net.weights[0][0, 0] = eps
    up = net.loss(x, [0])
    net.weights[0][0, 0] = -eps
    down = net.loss(x, [0])
    net.weights[0][0, 0] = 0.0
    numeric = (up - down) / (2 * eps)
    activations, zs = net.forward(x)
    step_w, _ = net.backprop(activations, zs, [0])
    assert numeric == pytest.approx(step_w[0][0, 0], abs=1e-6)
