# Policy-writing agent workspace

This experiment adapts the supplied `grounded_coord_simple-XP-seed0-task (1)`
bundle's workspace, policy contract, evaluation commands, and hypothesis ledger
to a single plasmax controller. A coding agent edits policy files, requests
scores or trajectory windows, and writes exploratory analyses. The existing
`agent_policy_search` completion-based runner remains available separately.

Use your chosen coding-agent host, or the small logged host below. Both
use the same workspace commands, recurrent policy contract and evaluation rules.

## Logged agent

After creating a fresh workspace, run a bounded tool-using agent:

```bash
uv run python -m experiments.agent_policy_workspace.agent \
    --workspace outputs/policy_workspace \
    --model openai/gpt-5.4-mini --proposals 2 --max-model-calls 8
```

To use Codex with its saved ChatGPT login and applicable subscription credits:

```bash
uv run python -m experiments.agent_policy_workspace.agent \
    --workspace outputs/policy_workspace \
    --host codex --model gpt-6.2-sol --proposals 2 --max-model-calls 8
```

`codex login status` must report a ChatGPT login for subscription billing. The
adapter removes API-key environment variables, uses a read-only sandbox with
built-in execution/connectors disabled, and asks Codex for a structured next
action. The trusted host executes the two workspace tools. In this mode,
`max-model-calls` limits CLI decision invocations; Codex can make internal model
requests within an invocation. `max-output-tokens` is an API-only hard limit;
the CLI has a per-invocation time limit but no equivalent exposed output cap.

The host scores the starter, supplies public task documents and that baseline,
then lets the model call `inspect_trajectory` and `evaluate_policy`. The latter
records a model-written hypothesis, archives the complete policy bundle, scores
it, and records the selection decision. Invalid submissions consume proposal
budget. The host restores the best development candidate and evaluates held-out
seeds only after model calls have ended. API or W&B failures stop the run.

`agent_session/` contains:

- Numbered exact `model_request.json` and `model_response.json` payloads, including
  the full replayed conversation and usage returned by the provider.
- `events.jsonl`: timestamps, model calls, tool arguments/results, errors and
  host-owned baseline/final evaluations, distinguished from agent calls.
- `transcript.md`: readable prompts, responses, tool exchanges and any returned
  reasoning summaries. Summaries are requested with `summary=auto`; a model may
  return none. Opaque encrypted reasoning is retained in JSON for replay, not
  displayed as readable reasoning.
- `config.json` and `summary.json`: budgets, usage and baseline/best/final scores.

Codex additionally saves `codex/turn_NNN/` with the exact supplied `prompt.txt`,
CLI command, output schema, native JSONL events, stderr and final action JSON.
The numbered request/response JSON in this mode is the harness's normalized
exchange; the native CLI files are authoritative. These record the prompt we
supply and events the CLI exposes, not its private system context, internal HTTP
requests or raw hidden reasoning. Emitted reasoning summaries are included in
the readable transcript.

The transcript stays local; existing W&B evaluation runs upload policy/config
artifacts and metrics. The model provider receives the task context, policy code
and selected development feedback. API credentials come from the environment;
Codex uses its saved login. Neither is included in recorded model requests.
Use a new workspace for each run;
existing transcript directories are never overwritten.

For FLAIR, keep this host and its API/W&B credentials local. Sync a minimal
checkout to `REMOTE_ROOT/source` using `ftk sync`, then set the trusted worker
command before initialization and agent execution:

```bash
export PLASMAX_POLICY_WORKER_COMMAND='["uv", "run", "--no-cache", "--no-sync", "python", "-m", "experiments.agent_policy_workspace.flair_worker", "--host", "f2", "--remote-root", "/homes/cdt24/theo/plasmax-policy-agent", "--gpu", "1", "--image", "theo_plasmax:jax-0.10.1-20260826-bgb"]'
```

Select the host/GPU from current `ftk find-gpus` output and verify the image on
that node. The adapter rechecks availability before each launch. Each evaluation
uses a non-root, network-disabled Docker container with no credential or home
mounts. Only the source checkout, candidate job directory and compilation cache
are mounted; the source checkout is read-only. Workers are time-bounded and exit
naturally. Their containers and per-job artifacts are retained for inspection;
cleanup follows the cluster's confirmation rules. `remote_execution.json` links
each local evaluation to its node, GPU, container and remote directory.

## Create a workspace

From the repository root, with research dependencies installed:

```bash
uv run python -m experiments.agent_policy_workspace init \
    --workspace outputs/policy_workspace \
    --env.env-setup iter/hybrid/flattop \
    --env.backend bohm_gyrobohm \
    --env.variant realistic
```

Initialization creates task/interface documentation, a valid starter policy,
and an experiment ledger. It does not call a model or collect a rollout. Give
your coding agent the generated workspace and ask it to follow its `AGENTS.md`.
Use a new workspace directory for each independent search.

`context/interface.json` describes the runtime interface and reward;
`context/config.json` records the public task settings. Evaluator-owned settings,
including the seed banks, live under `.policy_eval/` and are not agent-editable
inputs.

The default search lifetime is 7,200 seconds, starting when `init` creates the
workspace; initial environment preparation consumes part of that time. Official
evaluation uses 64 fixed development seeds; final evaluation uses 64 disjoint
held-out seeds. Inspect
`--help` for the configuration accepted by each command. Online W&B logging is
required for scoring, exploratory evaluation, and final evaluation; configure
W&B before starting those commands. Final evaluation must fit within the same
overall lifetime.

## Experiment loop

Set `WORKSPACE` to the generated directory, then establish the baseline. The
initial attempt is already open; conclude it using the hash returned by `score`:

```bash
uv run python -m experiments.agent_policy_workspace validate --workspace "$WORKSPACE"
uv run python -m experiments.agent_policy_workspace score --workspace "$WORKSPACE"
uv run python -m experiments.agent_policy_workspace conclude \
    --workspace "$WORKSPACE" --summary "Established the starting-policy baseline" \
    --decision keep --best-candidate-sha256 HASH --best-reason "Baseline candidate"
```

For a material change, record the hypothesis before editing:

```bash
uv run python -m experiments.agent_policy_workspace start \
    --workspace "$WORKSPACE" \
    --title "Reduce persistent control error" \
    --hypothesis "Accumulated error will improve tracking" \
    --planned-change "Add bounded integral state" \
    --success-criterion "Improve the official development mean return"
```

Edit `policy.py` and, if useful, `custom_primitives.py`. Validate and score again.
Use the reported candidate hash to inspect a bounded trajectory window:

```bash
uv run python -m experiments.agent_policy_workspace trace \
    --workspace "$WORKSPACE" --candidate HASH --episode 0 --start 0 --stop 100
uv run python -m experiments.agent_policy_workspace observe \
    --workspace "$WORKSPACE" --summary "Describe the measured result and failure mode"
uv run python -m experiments.agent_policy_workspace conclude \
    --workspace "$WORKSPACE" --summary "Explain what this attempt established" \
    --decision follow_up --best-candidate-sha256 HASH \
    --best-reason "Explain why this scored candidate remains preferred"
```

Use `explore --workspace "$WORKSPACE" --seeds 7 8` for additional exploratory
episodes. Their results help diagnose behavior; they do not replace official
development scores. Inspect one with `trace --workspace "$WORKSPACE"
--exploration EVALUATION_ID`, using the returned evaluation identifier.
Trace windows default to 16 steps and are limited to 200 steps per request;
choose a window within the recorded horizon. Use `--fields obs requested_action
reward trace_NAME` to include a policy diagnostic named `NAME`.
`rollouts.py` is an editable example for analyzing saved NPZ files without
putting every timestep into the agent's context:

```bash
uv run python "$WORKSPACE/rollouts.py" --path /path/to/trajectories.npz
```

Declare and restore an officially scored candidate before final evaluation:

```bash
uv run python -m experiments.agent_policy_workspace select-best \
    --workspace "$WORKSPACE" --candidate HASH --reason "Explain the evidence"
uv run python -m experiments.agent_policy_workspace restore-best --workspace "$WORKSPACE"
uv run python -m experiments.agent_policy_workspace finish --workspace "$WORKSPACE"
```

`score` automatically promotes an eligible candidate with the highest official
mean development return. `select-best` must select a candidate attaining that
best score; it cannot promote a lower-scoring or exploratory-only candidate.
`restore-best` preserves displaced source files in a backup. `finish` also
restores the selected bundle, freezes further development, independently
rechecks the development returns, and then evaluates the held-out bank.

The objective is development return, unlike the reference bundle's cross-play
objective. Keep held-out results out of subsequent policy decisions. Check
remaining search time with `lifetime --workspace "$WORKSPACE"`.

The workspace lifetime includes final evaluation. The agent measures evaluation
wall time and reserves two further seed banks plus tracking time before starting
another proposal. If the deadline prevents final verification, it preserves the
best scored source and records a partial result instead of a held-out score.
For unattended runs, also wrap the controller in an operating-system timeout so
the limit covers stalled provider or tracking calls. FLAIR workers receive an
absolute deadline and enforce it inside their containers.

## Policy contract

`policy.py` defines `POLICY_API_VERSION = 1`, `init_memory()`, and
`act(obs, reset, memory, rng) -> (action, next_memory, trace)`. Memory is a
fixed-shape JAX pytree, initialized independently for every episode. Trace is a
dictionary of fixed-shape numeric or boolean JAX arrays; it is recorded for
inspection and is not fed back into the next action. See the generated
`context/policy_api.md` for the full contract.

Observation layout, action shape/dtype, normalization, reward, and horizon come
from the selected environment. The starter uses the generated initial command;
it is a baseline, not a demonstrated useful controller. Helpers in
`custom_primitives.py` are part of the candidate bundle and must remain available
when restoring a candidate.

## Execution boundary

Generated policies run in subprocesses with evaluation timeouts. Read-only
context and evaluator-owned paths are workflow rules, not filesystem permission
boundaries. The workspace instructions constrain intended behavior, but are not
an operating-system sandbox: policy code and exploratory scripts may access
host files and network resources available to the launching account. Removing API-key environment
variables does not hide keys stored in files such as `~/.zshrc`.

Run this workspace inside an appropriately restricted agent/runtime environment
when executing untrusted generated code. No claim of secure isolation is made
by the task layout or the policy API.
