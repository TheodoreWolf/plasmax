# Policy interface

Keep `POLICY_API_VERSION = 1` in `policy.py` and implement:

```python
init_memory() -> memory
act(obs, reset, memory, rng) -> (action, next_memory, trace)
```

`init_memory()` returns a private fixed-shape JAX pytree. Each episode starts
with its own initialized memory. `next_memory` must preserve its pytree
structure, leaf shapes, and leaf dtypes. Honor the scalar boolean reset flag;
do not use Python global mutation or shared mutable objects as controller state.

`obs` is the environment's current observation, including the selected sensor
and history wrappers. Use its documented layout and scaling. The action must be
a finite JAX array with the exact shape and dtype in the generated interface.
Respect its bounds and units: normalized policy actions and physical actuator
values are different representations. Use only `obs`, `reset`, `memory`, the
supplied `rng`, and fixed constants; simulator state and reward are not policy
inputs.

`trace` is a dictionary whose identifier-style string keys map to fixed-shape
numeric or boolean JAX arrays. Keys start with a letter or underscore, followed
by word characters. Return `{}` if no diagnostics are needed. Keep keys, shapes,
and dtypes fixed across calls. Useful fields include controller mode, tracking
error, integral state, and saturation flags. Traces are recorded for inspection;
they do not affect reward and are not passed into the next call. Actual actions
are recorded separately. Saved fields are named `trace_NAME` for a diagnostic
key `NAME` and can be requested through the trace command's `--fields` option.

The policy must support JAX tracing and fixed-shape control flow. Use
`jax.numpy`, `jax.lax.cond`, `jax.lax.switch`, or other JAX operations when
branching on array values. Any randomness must use the supplied key. Runtime
file I/O, networking, subprocesses, host callbacks, and mutable global state are
outside the policy contract.

You may define helpers in `policy.py` or an optional `custom_primitives.py`;
import helper modules at the top of `policy.py`.
Both files form the candidate bundle and must be preserved together. Host-side
analysis belongs in `rollouts.py`, not in the runtime policy. Validation checks
the executable interface; official scoring evaluates behavior on the fixed
development seed bank.
