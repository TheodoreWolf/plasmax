# plasmax policy-search workspace

Improve the controller in this workspace according to `context/task_spec.md`.
Read `context/policy_api.md`, `context/interface.json`, `context/config.json`,
and the experiment ledger before editing the policy. Edit only `policy.py`,
optional `custom_primitives.py`, and `rollouts.py`; use the provided commands for
evaluation and experiment records. Keep `context/`, `.policy_eval/`, archives,
and evaluator reports unchanged. Do not inspect credentials, unrelated files,
hidden evaluation data, or evaluator internals.

The runtime policy may use only its observation, reset flag, private memory,
supplied PRNG key, and fixed constants. It must not perform host I/O, use the
network, launch processes, use host callbacks, or retain mutable global state.
Keep `POLICY_API_VERSION = 1`, `init_memory()`, and `act()` in `policy.py`.
Put optional policy helpers in `custom_primitives.py`.

The starting-template attempt is already open. Validate and officially score
the starter, then conclude that attempt with its candidate hash before opening
another attempt. The search lifetime starts when the workspace is initialized,
including initial environment preparation. For each material revision:

1. Check the remaining lifetime and open an attempt with a falsifiable
   hypothesis, planned change, and success criterion.
2. Edit, validate, and score the policy.
3. Inspect relevant trajectory windows or write an exploratory analysis.
4. Record evidence and failure modes, then conclude the attempt.
5. Reaffirm or update the preferred officially scored candidate.

Use the evaluator's candidate hash to identify source exactly. Preserve all
candidate archives and evaluation reports. Use ledger commands rather than
editing generated ledger files. Exploratory results must be identified as
exploratory; `score` is the authoritative development measurement.
Scoring automatically promotes the best eligible development return. A manual
best-candidate declaration must attain the highest official development score;
it cannot promote a lower-scoring or exploratory-only candidate.

Commands are available from the repository's Python environment:

```bash
uv run python -m experiments.agent_policy_workspace validate --workspace WORKSPACE
uv run python -m experiments.agent_policy_workspace score --workspace WORKSPACE
uv run python -m experiments.agent_policy_workspace trace --workspace WORKSPACE --candidate HASH --episode 0 --start 0 --stop 100
uv run python -m experiments.agent_policy_workspace explore --workspace WORKSPACE --seeds 7 8
uv run python -m experiments.agent_policy_workspace trace --workspace WORKSPACE --exploration EVALUATION_ID --episode 0 --start 0 --stop 100
uv run python -m experiments.agent_policy_workspace lifetime --workspace WORKSPACE
uv run python -m experiments.agent_policy_workspace start --workspace WORKSPACE --title "..." --hypothesis "..." --planned-change "..." --success-criterion "..."
uv run python -m experiments.agent_policy_workspace observe --workspace WORKSPACE --summary "..."
uv run python -m experiments.agent_policy_workspace conclude --workspace WORKSPACE --summary "..." --decision follow_up --best-candidate-sha256 HASH --best-reason "..."
uv run python -m experiments.agent_policy_workspace select-best --workspace WORKSPACE --candidate HASH --reason "..."
uv run python -m experiments.agent_policy_workspace restore-best --workspace WORKSPACE
uv run python -m experiments.agent_policy_workspace finish --workspace WORKSPACE
```

Replace `WORKSPACE` with this workspace's path. Before finishing, declare the
candidate supported by the official development results and restore it. Make no
further policy edits before final evaluation. Never use held-out results to
choose or revise a policy. Online W&B logging is required for evaluated runs;
do not bypass a tracking failure.

Every attempt conclusion must reaffirm or change the preferred officially
scored candidate, including rejected attempts and unchanged preferences. Record
the candidate hash and the reason for that preference.

These are development instructions, not proof of filesystem or network
isolation. Read-only paths here are workflow rules, not operating-system
permission boundaries. The agent host is responsible for any security sandbox.
