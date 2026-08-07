"""Generate an environment-throughput plot for TORAX backends on L40S.

Related measurements: outputs/env_throughput_l40s.md
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.project_paths import PLOTS_DIR

n_envs = np.array([64, 256, 1024, 2048, 4096])
iter_simple_sps = np.array([13058, 44159, 128142, 194004, 247429])
iter_cgm_sps = np.array([13174, 41661, 127783, 189208, 241220])

hf_n_envs = np.array([64, 256])
hf_sps = np.array([75.1, 64.8])

plt.style.use(Path(__file__).resolve().parent / "paper.mplstyle")

fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(n_envs, iter_simple_sps, marker="o", label="Constant")
ax.plot(n_envs, iter_cgm_sps, marker="s", label="CGM")
ax.plot(hf_n_envs, hf_sps, marker="^", label="QLKNN")

ideal = iter_simple_sps[0] * (n_envs / n_envs[0])
ax.plot(n_envs, ideal, linestyle="--", color="#999999", label="linear scaling")

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(n_envs)
ax.set_xticklabels([str(n) for n in n_envs])
ax.set_xlabel("Parallel environments")
ax.set_ylabel("Steps/s")
ax.grid(True, which="both")
ax.legend()

fig.tight_layout()
out = PLOTS_DIR / "env_throughput_l40s.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}")
