# Agent policy search

This clone-only study uses a language model to write and improve executable
`plasmax` policies. Each proposal is a complete Python source file. A simulator
worker evaluates it, and the next model request contains the incumbent policy,
development returns, and compact trajectory diagnostics. The policy executes
locally without model calls during a rollout.

The initial experiment is sequential: evaluate a caller-supplied seed policy,
generate a proposal, evaluate it on the development seeds, and retain the better
incumbent. Defaults are 10 proposals and two disjoint banks of 64 episodes each.
The development bank is reused across candidates for paired comparisons. The
held-out bank evaluates the final incumbent and is excluded from model feedback
and candidate selection. The explicit episode seeds are recorded with the run.
By default, development seeds are `0..63` and held-out seeds are
`10000..10063`; the CLI exposes the start and episode count for each bank.
The seed must evaluate successfully with finite returns and requested actions.
Invalid proposals and failed evaluations consume a proposal and leave the
incumbent in place; their errors are available to the next proposal. Model API
and tracking errors stop the search with the recorded artifacts preserved.

## Setup and launch

Run from the repository root, using the research dependencies:

```bash
uv sync --no-dev --group research
uv run python -m experiments.agent_policy_search.run --help
```

Configure credentials for the selected model provider and W&B before launching.
Model calls use LiteLLM's provider/model identifiers. The model, task, and seed
policy must be supplied explicitly. The supplied seed is evaluated before model
proposals begin. Online W&B logging is required. Failed
authentication or initialization stops the run, with no offline fallback.

For example, after setting `MODEL` to the desired provider/model identifier:

```bash
uv run python -m experiments.agent_policy_search.run \
    --model.model "$MODEL" \
    --task.env-setup iter/hybrid/flattop \
    --task.backend bohm_gyrobohm \
    --task.variant realistic \
    --seed-policy /absolute/path/to/seed_policy.py
```

Run outputs are saved under `outputs/agent_policy_search/`. Inspect the CLI help
for proposal, episode, model-generation, and evaluation limits before committing
to a run. Simulator work includes the seed policy, each evaluated proposal, and
the final held-out evaluation. New policy source generally requires a new JAX
compilation, which can take minutes for TORAX.

## Policy contract

A policy file defines a pure, stateless JAX-compatible callable:

```python
import jax
import jax.numpy as jnp


def act(obs: jax.Array, rng: jax.Array) -> jax.Array:
    """Return one action in the selected environment's policy action space."""
    del obs, rng
    return jnp.zeros((4,), dtype=jnp.float32)
```

This four-actuator midpoint example matches the normalized continuous action
interface of the ITER hybrid flat-top command above. It demonstrates the calling
convention; it has not been evaluated as a useful controller. Use the recorded
environment interface to choose action shape, bounds, observation indices, and
scaling for another task.

The model must return complete policy source, including its imports and helper
functions. Policies must support JAX tracing and fixed-shape execution inside
the rollout collector. Use JAX operations for data-dependent arithmetic and
control flow. Any randomness must use the supplied key. Python global mutation,
side effects, and retaining Python state between calls do not implement
controller memory under JIT. A controller can use history already included in
the observation; an explicit recurrent controller state is outside this initial
interface.

The selected wrappers define what the policy observes and the units of its
actions. In particular, normalized policy actions and physical actuator values
are different representations. Configured sensor noise, observation history,
action scaling, and rate limits remain part of the selected task variant.

## Feedback and recorded evidence

Each proposal receives development episode returns and compact diagnostics from
the collected trajectories. This bounds the context size while retaining
information about behavior across an episode. Full numerical trajectories are
saved as NPZ artifacts for every evaluated rollout; the model does not receive
every raw timestep by default. Validity masks distinguish real transitions from
padding after an episode ends. Requested physical commands and applied physical
controls must be interpreted separately where actuator limits affect execution.

Every model request is assembled afresh from the task/interface description,
incumbent source and feedback, and at most the latest proposal with its outcome.
There is no accumulated conversation, automatic access to local files, or
trajectory-inspection tool in this first version. Editing the request builder
controls the information supplied to the model.

The run records its configuration and episode seeds, exact model requests and
responses, candidate source and source hash, parent candidate, evaluation
results, acceptance decisions, and errors. These artifacts support inspecting
the information used for each proposal and evaluating saved source again. Model
sampling and numerical execution can still vary across providers, dependency
versions, and hardware; saved source and requests make that variation auditable.

Held-out results are final evaluation evidence. Repeatedly inspecting them and
then adapting policy code turns that bank into development data, so use a new
held-out bank for subsequent claims of generalization.

## Evaluate saved source without generating a proposal

Use the saved run configuration and candidate identifier to reproduce an
evaluation in a new output directory:

```bash
uv run python -m experiments.agent_policy_search.evaluate \
    --run-dir 'outputs/agent_policy_search/<run>' \
    --candidate-id 0001 \
    --split development \
    --output-dir 'outputs/agent_policy_search/<new-evaluation>'
```

Use `--split heldout` for the held-out seed bank. This path makes no model API
requests and still uses online W&B logging. The existing run artifacts remain
available alongside the newly recorded evaluation.
Omit `--candidate-id` to evaluate the saved best candidate.

## Execution limits

Generated code runs in subprocesses so a timeout or worker crash can be reported
without taking down the search controller. This provides fault isolation. It is
not a security sandbox and does not guarantee filesystem or network isolation.
Run only trusted seed source and use a dedicated restricted execution
environment when generated source requires a security boundary.

Episodes are evaluated sequentially and written individually, preserving
completed rollout artifacts if a later episode fails. Full simulator-state
history is retained for only one episode at a time. Numeric exports remain in
memory for the bank's combined feedback and trajectory file. Other backends
reuse one worker for the seed bank; its evaluation timeout covers that entire
bank. TGLFNN uses independent single-seed workers, with the timeout applied to
each episode. Long horizons and many rollouts can still require substantial
compilation time, simulator time, memory, and disk space.
