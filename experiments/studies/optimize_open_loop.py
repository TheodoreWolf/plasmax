"""Gradient-based open-loop actuator trajectory optimization through TORAX.

TORAX is end-to-end differentiable (its time-stepping ``while_loop`` and the
Newton-Raphson PDE solver carry custom reverse-mode rules), so we can backprop a
cumulative-reward objective all the way to the actuator settings at every step
and optimize the *whole open-loop schedule* by gradient ascent — the "optimal
trajectory the differentiable simulator can produce". TORAX itself ships no such
optimizer; this builds one on top of the plasmax env.

The decision variable is a per-step normalized action sequence ``a[t] ∈ [-1,1]^A``
(squashed through ``tanh`` from unconstrained ``theta``). It is rolled out through
the env with ``lax.scan``; the objective is the first-episode cumulative reward.
The scan freezes after its first termination or truncation and masks its fixed-size
padding. ``optax.adam`` updates ``theta``.

Notes:
  * The env's per-step ``max_action_delta`` rate-limit clip is inside the
    differentiated graph, so the optimized schedule is automatically rate-feasible
    (the optimizer gets zero gradient anywhere it tries to slew faster than the
    actuator allows). We initialize at the reset setpoint so step 0's clip starts
    as identity and gradients flow from the first step.
  * Reverse-mode AD stores per-step residuals; ``--remat`` checkpoints each step
    to bound memory at the cost of a forward recompute in the backward pass.
    Long phase horizons are still very expensive in reverse mode — start with a
    modest ``--num-steps`` (tens to low hundreds).

Usage::

    uv run python experiments/studies/optimize_open_loop.py  # defaults
    uv run python experiments/studies/optimize_open_loop.py --num-steps 60 --iters 40
    uv run python experiments/studies/optimize_open_loop.py \
        --env iter/hybrid/flattop --num-steps 100 --reward Q_fusion
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from plasmax.environment import factory as sc
from plasmax.wrappers import unwrap_to_env_state
from scripts.project_paths import wandb_dir


@dataclasses.dataclass
class Args:
    env: str = "iter/hybrid/flattop"
    backend: str = "cgm"
    reward: str | None = None
    # 'realistic' activates sensor effects and samples the env YAML's physics
    # parameters independently at every transition. Each random seed therefore
    # defines one deterministic stream of transition functions.
    variant: Literal["oracle", "realistic"] = "realistic"
    # Terminal reward on a disrupting/unphysical step (same semantics as
    # train_ppo.py's --env.disruption_penalty), for objective parity with PPO.
    # Non-differentiable (behind jnp.where), so it does not shape gradients
    # here -- use q_weight/gw_weight for that.
    disruption_penalty: float | None = None
    # Log per-iteration metrics to wandb.
    wandb: bool = False

    wandb_project: str = "plasmax"

    wandb_entity: str = "flair"

    wandb_group: str = "debug"

    num_steps: int = 50
    # Number of evenly-spaced time-knots for a piecewise-linear schedule (0 =
    # dense per-step controls). Use this to scale to long horizons.
    knots: int = 0

    iters: int = 30

    lr: float = 0.05

    seed: int = 0
    # Disable per-step checkpointing.
    no_remat: bool = False

    out: str = "outputs/open_loop_baseline.npz"
    # Validate the AD gradient against finite differences, then exit (no
    # optimization). Use a small num_steps; FD cost is O(2 * T * A * eps).
    check_grad: bool = False
    # Finite-difference step sizes for check_grad.
    fd_eps: tuple[float, ...] = (1e-2, 1e-3, 1e-4)
    # Disable the smooth q_min/Greenwald disruption barrier (robustness
    # against the disruption-cliff discontinuity).
    no_barrier: bool = False
    # q_min margin where the barrier switches on (hard threshold ~0.8).
    q_safe: float = 0.9

    q_weight: float = 50.0
    # n/n_GW margin where the barrier switches on (hard threshold ~1.1).
    gw_safe: float = 1.05

    gw_weight: float = 50.0
    # Per-step reward for staying alive (counters truncation myopia).
    survival_bonus: float = 0.0
    # Global-norm gradient clip (0 to disable).
    grad_clip: float = 1.0
    # Early-stop after N iters without best-reward improvement (0=off).
    patience: int = 0


def _setpoint_theta_row(env, key) -> jax.Array:
    """Returns the unconstrained ``theta`` row (shape (A,)) whose ``tanh`` is the
    env's reset actuator setpoint.

    Initializing the schedule at the reset setpoint means step 0's
    ``max_action_delta`` clip is the identity (action == prev_action), so
    gradients flow from step 0 instead of dying on a saturated clip.
    """
    state0, _ = env.init(key)
    state0 = unwrap_to_env_state(state0)
    base = env.unwrapped
    low = jnp.asarray(base.action_space.low)
    high = jnp.asarray(base.action_space.high)
    setpoint_norm = 2.0 * (state0.prev_action - low) / (high - low) - 1.0
    return jnp.arctanh(jnp.clip(setpoint_norm, -0.999, 0.999))


def make_parameterization(env, key, num_steps: int, n_knots: int):
    """Builds the decision-variable -> dense-action map.

    Returns ``(to_actions, theta0, label)`` where ``to_actions(theta)`` produces
    the dense normalized action sequence of shape ``(num_steps, A)`` in [-1,1],
    and ``theta0`` is the setpoint-hold initialization.

    Two modes:
      * ``n_knots <= 0`` or ``>= num_steps``: dense per-step controls, ``theta``
        has shape ``(num_steps, A)`` (direct single shooting).
      * ``0 < n_knots < num_steps``: piecewise-linear schedule, ``theta`` holds
        the values at ``n_knots`` evenly-spaced time-knots (shape ``(n_knots, A)``)
        and the per-step sequence is linearly interpolated between them. Far fewer
        parameters and a smoother landscape, so reverse-mode AD over long horizons
        stays cheap and well-conditioned. Interpolation happens in unconstrained
        space (before ``tanh``), keeping every interpolated action in-bounds.
    """
    theta_row = _setpoint_theta_row(env, key)  # (A,)
    act_dim = theta_row.shape[0]

    if 0 < n_knots < num_steps:
        knot_idx = jnp.linspace(0.0, num_steps - 1, n_knots)
        step_idx = jnp.arange(num_steps, dtype=knot_idx.dtype)

        def to_actions(theta):  # theta: (n_knots, A)
            interpolate_actuator = jax.vmap(
                lambda knot_values: jnp.interp(step_idx, knot_idx, knot_values),
                in_axes=1,
                out_axes=1,
            )
            return jnp.tanh(interpolate_actuator(theta))

        theta0 = jnp.broadcast_to(theta_row, (n_knots, act_dim))
        label = f"{n_knots} time-knots (piecewise-linear) x {act_dim} actuators"
    else:

        def to_actions(theta):  # theta: (num_steps, A)
            return jnp.tanh(theta)

        theta0 = jnp.broadcast_to(theta_row, (num_steps, act_dim))
        label = f"{num_steps} per-step controls x {act_dim} actuators"

    return to_actions, theta0, label


def make_barrier(q_safe: float, q_weight: float, gw_safe: float, gw_weight: float):
    """Smooth repulsive barrier on the *margin* to the disruption thresholds.

    The env's hard disruption (``q_min < q_threshold`` or ``n/n_GW > gw_threshold``)
    is a discontinuity: reward rises as you push toward it, then the episode
    terminates and reward falls off a cliff, so a greedy gradient climbs straight
    in. A constant ``disruption_penalty`` does *not* help — it sits behind a
    non-differentiable ``jnp.where(disruption, ...)``, contributing zero gradient.
    This barrier instead grows quadratically as ``q_min`` / Greenwald approach
    their thresholds from the safe side, giving a repulsive gradient *before* the
    cliff. It is exactly zero in normal operation (comfortable margins), so it
    does not distort the objective away from the boundary.
    """

    def barrier(postout) -> jax.Array:
        q_pen = q_weight * jnp.square(jnp.maximum(0.0, q_safe - postout.q_min))
        gw_pen = gw_weight * jnp.square(
            jnp.maximum(0.0, postout.fgw_n_e_line_avg - gw_safe)
        )
        return q_pen + gw_pen

    return barrier


def _make_rollout(env, remat: bool, barrier=None, survival_bonus: float = 0.0):
    """Returns ``rollout(actions_norm, key) -> (objective, aux)``.

    ``objective = sum_t alive_t * (reward_t - barrier_t + survival_bonus)`` is what
    we maximize; the survival bonus rewards reaching the horizon (countering the
    truncation myopia that traps the optimizer once it disrupts early). ``aux``
    carries the *true* physics cumulative reward (``raw_cum``, barrier-free) for
    reporting / best-iterate selection, plus per-step rewards, boundary flags,
    validity, and the ``q_min`` / Greenwald traces for margin diagnostics.
    """

    def step_body(carry, a):
        state, active = carry

        def step_active(_):
            next_state, info = env.step(state, a)
            plasma = unwrap_to_env_state(next_state).plasma
            pen = barrier(plasma) if barrier is not None else 0.0
            shaped = info.reward - pen + survival_bonus
            boundary = info.terminated | info.truncated
            return (next_state, ~boundary), (
                shaped,
                info.reward,
                info.terminated,
                info.truncated,
                jnp.asarray(True),
                plasma.q_min,
                plasma.fgw_n_e_line_avg,
            )

        def pad(_):
            plasma = unwrap_to_env_state(state).plasma
            zero = jnp.zeros_like(plasma.q_min)
            return (state, jnp.asarray(False)), (
                zero,
                zero,
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(False),
                plasma.q_min,
                plasma.fgw_n_e_line_avg,
            )

        return jax.lax.cond(active, step_active, pad, operand=None)

    if remat:
        step_body = jax.checkpoint(step_body)

    def rollout(actions_norm, key):
        state, _ = env.init(key)
        carry = (state, jnp.asarray(True))
        _, (shaped, raw, terminated, truncated, valid, qmin, fgw) = jax.lax.scan(
            step_body, carry, actions_norm
        )
        aux = {
            "raw_cum": jnp.sum(raw),
            "rewards": raw,
            "terminated": terminated,
            "truncated": truncated,
            "valid": valid,
            "qmin": qmin,
            "fgw": fgw,
        }
        return jnp.sum(shaped), aux

    return rollout


def _make_loss(env, remat: bool, to_actions, barrier=None, survival_bonus=0.0):
    """Returns ``loss(theta, key) -> (-objective, aux)`` over the decision var."""
    rollout = _make_rollout(env, remat, barrier, survival_bonus)

    def loss(theta, key):
        objective, aux = rollout(to_actions(theta), key)
        return -objective, aux

    return loss


def check_gradient(
    env, key, to_actions, theta0, eps_list: list[float], remat: bool
) -> None:
    """Compares the AD gradient against central finite differences.

    TORAX is a stiff implicit-solver system, the regime where Suh et al. (ICML
    2022) show first-order (AD) gradients can be biased/high-variance relative to
    a zeroth-order estimate. This is the cheap sanity check: if AD and FD agree
    (cosine ~1, small relative error) and that agreement is *stable across eps*,
    the AD gradient is trustworthy here. If agreement degrades as eps shrinks,
    the solver's Newton-Raphson tolerance is a noise floor swamping the FD probe.
    """
    loss = _make_loss(env, remat, to_actions)
    value_fn = jax.jit(lambda th, k: loss(th, k)[0])
    grad_fn = jax.jit(jax.value_and_grad(lambda th, k: loss(th, k), has_aux=True))

    theta = theta0
    (val, _), g_ad = grad_fn(theta, key)
    g_ad = np.asarray(g_ad).reshape(-1)
    print(f"objective at setpoint = {float(val):+.6f}")
    print(f"AD |grad| = {np.linalg.norm(g_ad):.3e}   ({g_ad.size} params)\n")

    base = np.asarray(theta).reshape(-1)
    shape = theta.shape

    def central_difference(flat_theta, eps):
        basis = jnp.eye(flat_theta.size, dtype=flat_theta.dtype)

        def evaluate_direction(direction):
            delta = eps * direction
            fp = value_fn((flat_theta + delta).reshape(shape), key)
            fm = value_fn((flat_theta - delta).reshape(shape), key)
            return (fp - fm) / (2.0 * eps)

        return jax.vmap(evaluate_direction)(basis)

    base_jax = jnp.asarray(base)
    eps_array = jnp.asarray(eps_list, dtype=base_jax.dtype)
    finite_difference_batch = jax.jit(jax.vmap(central_difference, in_axes=(None, 0)))
    g_fds = np.asarray(finite_difference_batch(base_jax, eps_array))
    print(f"{'eps':>8}  {'cos_sim':>9}  {'rel_L2_err':>11}  {'max_abs_diff':>12}")
    for eps, g_fd in zip(eps_list, g_fds, strict=True):
        denom = np.linalg.norm(g_ad) * np.linalg.norm(g_fd) + 1e-30
        cos = float(g_ad @ g_fd / denom)
        rel = float(np.linalg.norm(g_ad - g_fd) / (np.linalg.norm(g_fd) + 1e-30))
        mad = float(np.max(np.abs(g_ad - g_fd)))
        print(f"{eps:8.0e}  {cos:9.5f}  {rel:11.3e}  {mad:12.3e}")

    # Per-component view at the largest eps (least noise-floor-limited).
    max_eps_idx = int(np.argmax(eps_list))
    eps = eps_list[max_eps_idx]
    g_fd = g_fds[max_eps_idx]
    print(f"\nper-component (eps={eps:.0e}), AD vs FD:")
    for i in range(min(8, base.size)):
        print(f"  [{i:2d}]  AD={g_ad[i]:+.4e}   FD={g_fd[i]:+.4e}")


def optimize(
    env,
    key,
    to_actions,
    theta0,
    num_steps: int,
    iters: int,
    lr: float,
    remat: bool,
    barrier=None,
    survival_bonus: float = 0.0,
    grad_clip: float = 1.0,
    patience: int = 0,
    on_iter=None,
):
    """Runs Adam on the open-loop schedule; returns (best_theta, history, best).

    ``on_iter``, if given, is called with each iteration's history dict as it
    happens (e.g. ``wandb.log``) -- logging only after the loop returns loses
    every iteration under a hard walltime kill (SLURM, etc).

    Tracks the best iterate by *true* cumulative reward (barrier-free) and returns
    that, not the last — the optimizer can still wander off the disruption cliff,
    and we never want to ship the collapsed post-cliff schedule. Gradient clipping
    softens any overshoot off the cliff; optional ``patience`` early-stops when the
    best hasn't improved.
    """
    loss = _make_loss(env, remat, to_actions, barrier, survival_bonus)
    grad_fn = jax.jit(jax.value_and_grad(loss, has_aux=True))

    chain = [optax.zero_nans()]
    if grad_clip and grad_clip > 0:
        chain.append(optax.clip_by_global_norm(grad_clip))
    chain.append(optax.adam(lr))
    opt = optax.chain(*chain)

    theta = theta0
    opt_state = opt.init(theta)
    best = {"raw_cum": -np.inf, "theta": theta, "iter": -1, "alive": 0}
    history: list[dict] = []
    since_improve = 0
    for it in range(iters):
        t0 = time.time()
        (neg_obj, aux), grad = grad_fn(theta, key)  # evaluated AT theta
        raw_cum = float(aux["raw_cum"])
        valid = np.asarray(aux["valid"])
        n_alive = int(valid.sum())
        qmin_min = float(np.min(np.asarray(aux["qmin"])[:n_alive]))
        fgw_max = float(np.max(np.asarray(aux["fgw"])[:n_alive]))
        gnorm = float(jnp.linalg.norm(grad))

        if raw_cum > best["raw_cum"]:
            best = {"raw_cum": raw_cum, "theta": theta, "iter": it, "alive": n_alive}
            since_improve = 0
        else:
            since_improve += 1

        updates, opt_state = opt.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        record = {
            "iter": it,
            "raw_cum": raw_cum,
            "obj": float(-neg_obj),
            "grad_norm": gnorm,
            "alive": n_alive,
            "qmin_min": qmin_min,
            "fgw_max": fgw_max,
        }
        history.append(record)
        if on_iter is not None:
            on_iter(record)
        tag = "compile+" if it == 0 else ""
        print(
            f"[{it:3d}] raw_cum={raw_cum:+.4f}  obj={-float(neg_obj):+.4f}  "
            f"|grad|={gnorm:.2e}  alive={n_alive}/{num_steps}  "
            f"q_min={qmin_min:.3f}  n/n_GW={fgw_max:.3f}  "
            f"({tag}{time.time() - t0:.1f}s)"
        )
        if patience and since_improve >= patience:
            print(f"early stop: no improvement for {patience} iters")
            break

    print(
        f"\nbest iterate: #{best['iter']}  raw_cum={best['raw_cum']:+.4f}  "
        f"alive={best['alive']}/{num_steps}"
    )
    return best["theta"], history, best


def evaluate(env, key, actions_norm: jax.Array):
    """Replays a fixed normalized schedule; returns (raw_cum_reward, traj dict)."""
    rollout = _make_rollout(env, remat=False)  # barrier-free: true physics
    _, aux = jax.jit(rollout)(actions_norm, key)
    return float(aux["raw_cum"]), {
        "rewards": np.asarray(aux["rewards"]),
        "terminated": np.asarray(aux["terminated"]),
        "truncated": np.asarray(aux["truncated"]),
        "valid": np.asarray(aux["valid"]),
        "qmin": np.asarray(aux["qmin"]),
        "fgw": np.asarray(aux["fgw"]),
    }


def main() -> None:
    args = tyro.cli(Args)

    env = sc.make(
        args.env,
        args.backend,
        reward=args.reward,
        max_steps=args.num_steps,
        disruption_penalty=args.disruption_penalty,
        variant=args.variant,
    )
    key = jax.random.key(args.seed)

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            dir=wandb_dir(),
            group=args.wandb_group,
            name=f"backprop-{Path(args.env).stem}-{Path(args.backend).stem}-seed{args.seed}",
            config={**dataclasses.asdict(args), "method": "backprop_open_loop"},
        )

    to_actions, theta0, label = make_parameterization(
        env, key, args.num_steps, args.knots
    )
    print(f"parameterization: {label}")
    barrier = (
        None
        if args.no_barrier
        else make_barrier(args.q_safe, args.q_weight, args.gw_safe, args.gw_weight)
    )
    print(
        "disruption barrier: "
        + (
            "OFF"
            if barrier is None
            else f"q_safe={args.q_safe} (w={args.q_weight}), "
            f"gw_safe={args.gw_safe} (w={args.gw_weight})"
        )
        + f"  |  grad_clip={args.grad_clip}  survival_bonus={args.survival_bonus}\n"
    )

    if args.check_grad:
        print(
            f"Gradient check: {args.env} / {args.backend}, reward={args.reward}, "
            f"T={args.num_steps}\n"
        )
        check_gradient(
            env,
            key,
            to_actions,
            theta0,
            eps_list=sorted(args.fd_eps, reverse=True),
            remat=not args.no_remat,
        )
        return

    # Baseline: hold the reset setpoint for the whole episode.
    setpoint_actions = to_actions(theta0)
    base_reward, _ = evaluate(env, key, setpoint_actions)
    print(f"setpoint-hold baseline cum_reward = {base_reward:+.5f}\n")

    theta, history, best = optimize(
        env,
        key,
        to_actions,
        theta0,
        num_steps=args.num_steps,
        iters=args.iters,
        lr=args.lr,
        remat=not args.no_remat,
        barrier=barrier,
        survival_bonus=args.survival_bonus,
        grad_clip=args.grad_clip,
        patience=args.patience,
        on_iter=(lambda h: wandb.log(h, step=h["iter"])) if args.wandb else None,
    )

    opt_actions = to_actions(theta)
    opt_reward, opt_traj = evaluate(env, key, opt_actions)
    phys_actions = np.asarray(env.to_physical(opt_actions))
    n_alive = int(opt_traj["valid"].sum())

    improvement = opt_reward - base_reward
    print(
        f"\noptimized cum_reward = {opt_reward:+.5f}  "
        f"(baseline {base_reward:+.5f}, Δ = {improvement:+.5f}, "
        f"{100 * improvement / abs(base_reward):+.1f}%)  "
        f"alive={n_alive}/{args.num_steps}"
    )
    print("optimized physical actions [step 0]:", phys_actions[0])
    print("optimized physical actions [last] :", phys_actions[-1])

    if args.wandb:
        wandb.log(
            {
                "final/baseline_reward": base_reward,
                "final/opt_reward": opt_reward,
                "final/improvement": improvement,
                "final/alive_steps": n_alive,
                "final/disrupted": n_alive < args.num_steps,
            }
        )
        wandb.finish()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        actions_norm=np.asarray(opt_actions),
        actions_phys=phys_actions,
        rewards=opt_traj["rewards"],
        terminated=opt_traj["terminated"],
        truncated=opt_traj["truncated"],
        valid=opt_traj["valid"],
        qmin=opt_traj["qmin"],
        fgw=opt_traj["fgw"],
        baseline_cum_reward=base_reward,
        optimized_cum_reward=opt_reward,
        best_iter=best["iter"],
        history_iter=np.array([h["iter"] for h in history]),
        history_raw_cum=np.array([h["raw_cum"] for h in history]),
        history_alive=np.array([h["alive"] for h in history]),
        theta=np.asarray(theta),  # decision variable (knots or dense)
        knots=args.knots,
        env=args.env,
        backend=args.backend,
        reward=args.reward,
    )
    print(f"\nSaved optimized open-loop baseline to {out}")
    print(
        "Replay with: np.load(path)['actions_norm'] -> feed as a "
        "constant-per-step policy."
    )


if __name__ == "__main__":
    main()
