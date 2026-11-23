# plasmax policy-search task

Maximize expected episode return for the configured plasmax task. There is one
controller acting on the environment's action vector. Use the generated
environment/interface description for observation meanings, action bounds,
normalization, reward definition, and horizon. An omitted reward override
inherits the task's configured reward.

Official development evaluations use a fixed seed bank so candidate comparisons
are paired. A successful candidate must satisfy the policy interface and have
finite returns and requested actions. Inspect episode lengths and termination
reasons alongside return when diagnosing behavior. Termination is a control
outcome; do not remove failing episodes or change the task to improve a score.

The goal is high mean development return, followed by an independent final
evaluation. Prefer an officially scored candidate on that evidence. Additional
seeds, shorter exploratory runs, or custom summaries can guide hypotheses but
are not official replacement scores. Keep the simulator, reward, seed bank, and
recorded evaluation settings unchanged during search.

Use selected trajectories and controller diagnostics to investigate a specific
question, rather than copying every timestep into context. Record what changed,
what the evidence showed, and what remains uncertain. Previous failed attempts
are useful evidence and should remain in the experiment history.

The held-out bank is used only for final evaluation. Do not inspect its outcomes
or use them to select or revise policies. After final evaluation, further
development requires a fresh held-out bank for a new generalization assessment.
