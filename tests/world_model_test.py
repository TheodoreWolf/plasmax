"""Tests for the JAX KSTAR fusion_lstm forward pass (plasmax.models.world_model).

Fast tests use vendored weights only (golden values + shape/jit/vmap contract).
The ``integration``-marked parity test compares against the original NeoRL2
ONNX models; it is skipped unless those weights are reachable (set
``NEORL2_WEIGHTS`` to the ``.../fusion_lstm/weights`` dir, or have the public
``neorl2`` package installed). Golden values were captured from the JAX forward
pass after it was validated to match onnxruntime to float32 precision (~1e-6).
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plasmax.models import world_model as wm

# Fixed inputs (deterministic) used for the golden-value checks.
_X_LSTM = np.arange(10 * 21, dtype=np.float32).reshape(10, 21) / 210.0
_X_NN = np.arange(17, dtype=np.float32) / 17.0
_X_BPW = np.arange(8, dtype=np.float32) / 8.0

_GOLDEN = {
    "lstm": np.array([1.0582094, 5.8657203, 2.4299777, 1.0157008], np.float32),
    # nn uses member 0 only (NeoRL2 instantiates it with n_models=1).
    "nn": np.array([0.514441, 6.668106, 4.181331, 0.843372], np.float32),
    "bpw": np.array([1.9993268e-01, 8.0815151e03], np.float32),
}


class WorldModelForwardTest:
    def setup_class(self):
        self.bundle = wm.load_bundle()

    def test_output_shapes(self):
        assert wm.predict_lstm(self.bundle, jnp.asarray(_X_LSTM)).shape == (4,)
        assert wm.predict_nn(self.bundle, jnp.asarray(_X_NN)).shape == (4,)
        assert wm.predict_bpw(self.bundle, jnp.asarray(_X_BPW)).shape == (2,)

    def test_golden_values(self):
        # Pins numerical correctness without needing the ONNX reference.
        np.testing.assert_allclose(
            wm.predict_lstm(self.bundle, jnp.asarray(_X_LSTM)),
            _GOLDEN["lstm"],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            wm.predict_nn(self.bundle, jnp.asarray(_X_NN)),
            _GOLDEN["nn"],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            wm.predict_bpw(self.bundle, jnp.asarray(_X_BPW)),
            _GOLDEN["bpw"],
            rtol=1e-5,
            atol=1.0,  # wmhd ~ 8e3
        )

    def test_jit_and_vmap(self):
        # The whole point of the JAX port: jit + vmap clean (for scan/PPO).
        batch = jnp.asarray(np.stack([_X_LSTM, _X_LSTM + 0.1]))
        out = jax.jit(jax.vmap(lambda x: wm.predict_lstm(self.bundle, x)))(batch)
        assert out.shape == (2, 4)
        assert jnp.all(jnp.isfinite(out))


def _onnx_weights_dir() -> Path | None:
    env = os.environ.get("NEORL2_WEIGHTS")
    if env and Path(env).is_dir():
        return Path(env)
    try:
        import neorl2  # noqa: F401

        p = Path(neorl2.__file__).parent / "envs" / "data" / "fusion_lstm" / "weights"
        return p if p.is_dir() else None
    except ImportError:
        return None


@pytest.mark.integration
class WorldModelOnnxParityTest:
    """Parity vs the original NeoRL2 ONNX ensemble (the fidelity gate)."""

    def setup_class(self):
        pytest.importorskip("onnxruntime", reason="install the 'onnx' extra")
        self.weights_dir = _onnx_weights_dir()
        if self.weights_dir is None:
            pytest.skip("NeoRL2 ONNX weights not found (set NEORL2_WEIGHTS)")
        self.bundle = wm.load_bundle()

    def _onnx(self, tag, x, n_members=10):
        # nn uses member 0 only (NeoRL2 n_models=1); lstm/bpw use all 10.
        import onnxruntime as ort

        refs = []
        for i in range(n_members):
            s = ort.InferenceSession(
                str(self.weights_dir / tag / f"best_model{i}.onnx")
            )
            iname, oname = s.get_inputs()[0].name, s.get_outputs()[0].name
            refs.append(s.run([oname], {iname: x[None].astype(np.float32)})[0][0])
        ymean = np.asarray(self.bundle[tag]["ymean"])
        ystd = np.asarray(self.bundle[tag]["ystd"])
        return np.mean([r * ystd + ymean for r in refs], axis=0)

    def test_lstm_parity(self):
        x = np.random.default_rng(0).standard_normal((10, 21)).astype(np.float32)
        got = np.asarray(wm.predict_lstm(self.bundle, jnp.asarray(x)))
        np.testing.assert_allclose(got, self._onnx("lstm", x), rtol=1e-4, atol=1e-3)

    def test_nn_parity(self):
        x = np.random.default_rng(1).standard_normal(17).astype(np.float32)
        got = np.asarray(wm.predict_nn(self.bundle, jnp.asarray(x)))
        np.testing.assert_allclose(
            got, self._onnx("nn", x, n_members=1), rtol=1e-4, atol=1e-3
        )

    def test_bpw_parity(self):
        x = np.random.default_rng(2).standard_normal(8).astype(np.float32)
        got = np.asarray(wm.predict_bpw(self.bundle, jnp.asarray(x)))
        # wmhd channel ~ 1e4, so use a relative tolerance there.
        np.testing.assert_allclose(got, self._onnx("bpw", x), rtol=1e-4, atol=1.0)
