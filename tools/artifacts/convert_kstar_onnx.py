"""Convert the NeoRL2 KSTAR fusion_lstm ONNX weights to a JAX-friendly npz.

The NeoRL2 `fusion_lstm` env (KSTAR, Seo et al. Nucl. Fusion 2021) ships three
TF-exported ONNX model ensembles (10 members each):

* ``lstm`` — the dynamics model. Input ``(10, 21)``, output 4 = [βn, q95, q0, li].
  Architecture (BatchNorm folded to affine ``x*mul + sub`` at inference):
      BN(21) → LSTM0(200, seq) → BN(200) → LSTM1(200, last) → BN(200)
             → Dense(200) sigmoid → BN(200) → Dense(4) linear → denorm
  Keras LSTM kernels: W ``(in, 4*200)``, R ``(200, 4*200)``, b ``(4*200,)``,
  gate order ``[i, f, c, o]`` (input, forget, cell, output).
* ``nn`` — steady-state init. Input ``(17,)``, output 4. 4×(Dense+sigmoid/BN),
  last Dense linear.
* ``bpw`` — β-power head. Input ``(8,)``, output 2 = [βp, wmhd]. 3 Dense layers.

Output denorm (``y*ystd + ymean``) stats live in NeoRL2's ``fusion_utils.py``
(not the ONNX graph) and are copied below.

This is a one-off dev-time tool (needs ``onnx``); the runtime only needs the
emitted npz + JAX. Run with the NeoRL2 repo checked out somewhere:

    uv run python tools/artifacts/convert_kstar_onnx.py \
        --src /path/to/NeoRL2/neorl2/envs/data/fusion_lstm/weights \
        --out src/plasmax/configs/data/kstar_lstm/weights.npz

All arrays are stacked with a leading ensemble axis of length 10 so the JAX
forward pass can ``vmap`` over members.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from scripts.project_paths import CONFIGS_DIR

N_MODELS = 10

# Denorm stats (from NeoRL2 neorl2/envs/data/fusion_lstm/fusion_utils.py).
YMEAN = {
    "lstm": np.array([1.30934765, 5.20082444, 1.47538417, 1.14439883], np.float32),
    "nn": np.array([1.22379703, 5.2361062, 1.64438005, 1.12040048], np.float32),
    "bpw": np.array([1.02158800e00, 1.87408512e05], np.float32),
}
YSTD = {
    "lstm": np.array([0.74135689, 1.44731883, 0.56747578, 0.23018484], np.float32),
    "nn": np.array([0.72255576, 1.5622809, 0.96563557, 0.23868018], np.float32),
    "bpw": np.array([6.43390272e-01, 1.22543529e05], np.float32),
}


def _sq(a: np.ndarray) -> np.ndarray:
    """Drop singleton dims (BN params export as (1, 1, C)); keep matrices as-is."""
    a = np.asarray(a, np.float32)
    return a.reshape(a.shape[-1]) if a.ndim != 2 else a


def _ordered_weights(path: Path) -> list[tuple[str, np.ndarray]]:
    """Walk the graph topologically and pull weight tensors in node order.

    Robust to the ONNX initializer renaming across ensemble members (only
    model 0 keeps the canonical ``sequential/...`` names). The graph *structure*
    is identical across members, so node order + op type identifies each weight.

    Emits ``(role, array)`` for the relevant ops, where role is one of
    ``mul`` / ``add`` / ``matmul`` (BatchNorm folds to ``Mul``+``Add``; Dense is
    ``MatMul``+``Add``) or ``lstm`` (a Keras LSTM ``Loop`` -> W, R, b in order).
    Loop control scalars are skipped by keeping only >=1-D initializer inputs.
    """
    g = onnx.load(str(path)).graph
    inits = {init.name: numpy_helper.to_array(init) for init in g.initializer}
    out: list[tuple[str, np.ndarray]] = []
    for node in g.node:
        big = [inits[i] for i in node.input if i in inits and inits[i].ndim >= 1]
        if node.op_type in ("Mul", "Add", "MatMul") and big:
            out.append((node.op_type.lower(), _sq(big[0])))
        elif node.op_type == "Loop":  # W, R, b (in input order)
            for a in big:
                out.append(("lstm", a.astype(np.float32)))
    return out


def _assign(
    ops: list[tuple[str, np.ndarray]], roles: list[str]
) -> dict[str, np.ndarray]:
    """Positionally map the ordered (op, array) list onto semantic role names."""
    if len(ops) != len(roles):
        got = [o for o, _ in ops]
        raise ValueError(f"expected {len(roles)} weight ops, got {len(ops)}: {got}")
    return {role: arr for role, (_, arr) in zip(roles, ops, strict=True)}


# Role order matching the LSTM graph's topological node order.
_LSTM_ROLES = [
    "bn0_mul",
    "bn0_sub",  # leading BN over the 21 input features
    "W0",
    "R0",
    "b0",  # LSTM layer 0 (return_sequences)
    "bn1_mul",
    "bn1_sub",
    "W1",
    "R1",
    "b1",  # LSTM layer 1 (last timestep)
    "bn2_mul",
    "bn2_sub",
    "Wd",
    "bd",  # Dense(200) + sigmoid
    "bn3_mul",
    "bn3_sub",
    "Wout",
    "bout",  # Dense(4) linear
]


def _mlp_roles(n_dense: int) -> list[str]:
    """BN -> (Dense -> sigmoid -> BN) ... -> Dense: n_dense blocks of bn,W,b."""
    roles: list[str] = []
    for k in range(1, n_dense + 1):
        roles += [f"bn{k}_mul", f"bn{k}_sub", f"W{k}", f"b{k}"]
    return roles


def _convert(src: Path, tag: str, roles: list[str]) -> dict[str, np.ndarray]:
    stacks: dict[str, list[np.ndarray]] = {r: [] for r in roles}
    for i in range(N_MODELS):
        weights = _assign(_ordered_weights(src / f"best_model{i}.onnx"), roles)
        for r in roles:
            stacks[r].append(weights[r])
    out = {f"{tag}/{r}": np.stack(v) for r, v in stacks.items()}
    out[f"{tag}/ymean"] = YMEAN[tag]
    out[f"{tag}/ystd"] = YSTD[tag]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="NeoRL2 .../fusion_lstm/weights dir (with lstm/ nn/ bpw/)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=CONFIGS_DIR / "data" / "kstar_lstm" / "weights.npz",
    )
    args = ap.parse_args()

    bundle: dict[str, np.ndarray] = {}
    bundle.update(_convert(args.src / "lstm", "lstm", _LSTM_ROLES))
    bundle.update(_convert(args.src / "nn", "nn", _mlp_roles(4)))
    bundle.update(_convert(args.src / "bpw", "bpw", _mlp_roles(3)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **bundle)
    total = sum(a.nbytes for a in bundle.values())
    print(f"Wrote {len(bundle)} arrays ({total / 1e6:.1f} MB raw) to {args.out}")
    for k in sorted(bundle):
        print(f"  {k}: {bundle[k].shape}")


if __name__ == "__main__":
    main()
