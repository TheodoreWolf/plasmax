# Development guidance

Establish the starter's score before changing it. Read the task's reward and
interface before selecting observation indices or actuator values. Work from
one falsifiable hypothesis at a time and record unsuccessful attempts as well
as improvements.

Ask focused questions of trajectories: when did tracking error grow, which
requested controls saturated, and what preceded termination? Use policy traces
for internal state that observations alone cannot explain. Keep requested
commands, applied physical controls, and observation scaling distinct.

Use recurrent memory for quantities that truly depend on history, such as an
integral error or an estimated mode. Keep memory and trace schemas fixed under
JAX. Report numerical failures rather than hiding them with modified rewards or
discarded episodes.

The default objective is mean return on the official development bank. Preserve
that evaluation protocol while using exploratory analyses to decide what to
try next. Final held-out results are evaluation evidence, not search feedback.
