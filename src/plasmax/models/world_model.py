"""Pure-JAX forward pass for the NeoRL2 KSTAR fusion_lstm model ensemble.

Reimplements (in JAX, for vmap/scan/jit compatibility) the three TF-exported
ONNX nets that NeoRL2's KSTAR env uses, from weights converted by
``tools/artifacts/convert_kstar_onnx.py`` (packaged under
``configs/data/kstar_lstm``):

* ``lstm`` — dynamics. Input ``(10, 21)`` -> 4 = [βn, q95, q0, li].
* ``nn``   — steady-state init. Input ``(17,)`` -> 4.
* ``bpw``  — β-power head. Input ``(8,)`` -> 2 = [βp, wmhd].

BatchNorm is folded to an affine ``x*mul + sub`` (inference mode) in the npz.
Dense layers use sigmoid activations except the linear output layer. The LSTM
layers are standard Keras cells (gate order ``[i, f, c, o]``,
``recurrent_activation=sigmoid``, ``activation=tanh``, zero initial state).

Each net is a 10-member ensemble; predictions denormalise (``y*ystd + ymean``)
and average over members. Validated against onnxruntime in
``tests/world_model_test.py``.
"""

from __future__ import annotations

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "data"
    / "kstar_lstm"
    / "weights.npz"
)
_TAGS = ("lstm", "nn", "bpw")

# A bundle is {tag: {role: jnp.ndarray}}; per-role arrays carry a leading
# ensemble axis of length 10. ymean/ystd have no ensemble axis.
Bundle = dict[str, dict[str, jax.Array]]


def load_bundle(path: str | Path = _DEFAULT_WEIGHTS) -> Bundle:
    """Load the converted KSTAR weights npz into a nested {tag: {role: array}}."""
    raw = np.load(path)
    bundle: Bundle = {tag: {} for tag in _TAGS}
    for key in raw.files:
        tag, role = key.split("/", 1)
        bundle[tag][role] = jnp.asarray(raw[key])
    return bundle


# --- primitive ops ---------------------------------------------------------


def _affine(x: jax.Array, mul: jax.Array, sub: jax.Array) -> jax.Array:
    """Folded BatchNorm at inference: ``x * mul + sub`` (broadcast over rows)."""
    return x * mul + sub


def _dense(x: jax.Array, W: jax.Array, b: jax.Array) -> jax.Array:
    return x @ W + b


def _lstm_layer(
    x_seq: jax.Array, W: jax.Array, R: jax.Array, b: jax.Array, return_sequences: bool
) -> jax.Array:
    """Keras LSTM over a single sequence ``x_seq`` of shape ``(T, in)``.

    Kernel ``W (in, 4u)``, recurrent ``R (u, 4u)``, bias ``b (4u,)`` with gate
    order ``[i, f, c, o]``. Returns ``(T, u)`` if ``return_sequences`` else the
    last hidden state ``(u,)``.
    """
    u = W.shape[1] // 4

    def step(carry, x_t):
        h, c = carry
        z = x_t @ W + h @ R + b
        i = jax.nn.sigmoid(z[:u])
        f = jax.nn.sigmoid(z[u : 2 * u])
        g = jnp.tanh(z[2 * u : 3 * u])
        o = jax.nn.sigmoid(z[3 * u :])
        c = f * c + i * g
        h = o * jnp.tanh(c)
        return (h, c), h

    zeros = jnp.zeros((u,), x_seq.dtype)
    (h_last, _), h_seq = jax.lax.scan(step, (zeros, zeros), x_seq)
    return h_seq if return_sequences else h_last


# --- single-model forward passes -------------------------------------------


def lstm_forward(w: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """One LSTM member. ``x``: ``(10, 21)`` -> normalised output ``(4,)``."""
    h = _affine(x, w["bn0_mul"], w["bn0_sub"])
    h = _lstm_layer(h, w["W0"], w["R0"], w["b0"], return_sequences=True)
    h = _affine(h, w["bn1_mul"], w["bn1_sub"])
    h = _lstm_layer(h, w["W1"], w["R1"], w["b1"], return_sequences=False)
    h = _affine(h, w["bn2_mul"], w["bn2_sub"])
    h = jax.nn.sigmoid(_dense(h, w["Wd"], w["bd"]))
    h = _affine(h, w["bn3_mul"], w["bn3_sub"])
    return _dense(h, w["Wout"], w["bout"])


def mlp_forward(w: dict[str, jax.Array], x: jax.Array, n_dense: int) -> jax.Array:
    """One feed-forward member: BN -> (Dense->sigmoid->BN)... -> Dense(linear)."""
    h = _affine(x, w["bn1_mul"], w["bn1_sub"])
    for k in range(1, n_dense + 1):
        h = _dense(h, w[f"W{k}"], w[f"b{k}"])
        if k < n_dense:
            h = jax.nn.sigmoid(h)
            h = _affine(h, w[f"bn{k + 1}_mul"], w[f"bn{k + 1}_sub"])
    return h


# --- ensemble predictions (denorm + average over members) ------------------


def _ensemble(single_fn, weights: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """vmap ``single_fn`` over the 10 members, denormalise, average."""
    members = {k: v for k, v in weights.items() if k not in ("ymean", "ystd")}
    raw = jax.vmap(lambda w: single_fn(w, x))(members)  # (10, out)
    return (raw * weights["ystd"] + weights["ymean"]).mean(axis=0)


def predict_lstm(bundle: Bundle, x: jax.Array) -> jax.Array:
    """Ensemble LSTM dynamics. ``x``: ``(10, 21)`` -> ``(4,)`` [βn, q95, q0, li]."""
    return _ensemble(lstm_forward, bundle["lstm"], x)


def predict_nn(bundle: Bundle, x: jax.Array) -> jax.Array:
    """Steady-state init, **member 0 only**. ``x``: ``(17,)`` -> ``(4,)``.

    NeoRL2 instantiates the ``nn`` with ``n_models=1`` (the ``lstm`` and ``bpw``
    nets use the full 10-member ensemble); we match that so the reset matches.
    """
    w0 = {k: v[0] for k, v in bundle["nn"].items() if k not in ("ymean", "ystd")}
    raw = mlp_forward(w0, x, n_dense=4)
    return raw * bundle["nn"]["ystd"] + bundle["nn"]["ymean"]


def predict_bpw(bundle: Bundle, x: jax.Array) -> jax.Array:
    """Ensemble β-power head. ``x``: ``(8,)`` -> ``(2,)`` [βp, wmhd]."""
    return _ensemble(functools.partial(mlp_forward, n_dense=3), bundle["bpw"], x)
