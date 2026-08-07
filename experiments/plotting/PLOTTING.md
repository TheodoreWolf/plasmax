# Paper plotting style

One style for every figure in the paper. Apply it with the matplotlib style
sheet — no per-script styling:

```python
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use(Path(__file__).resolve().parent / "paper.mplstyle")
```

The sheet lives at `experiments/plotting/paper.mplstyle`.

## Output location and filenames

- Write every generated figure under `plots/`, or a task-specific
  subdirectory of `plots/`. This applies to PDF, PNG, and SVG files.
- Never write figures under `output/`, `outputs/`, `output/pdf/`, or the
  repository root. `outputs/` is reserved for numeric results and other data
  artifacts; `logs/` is reserved for run logs.
- Use concise, stable filenames in the form `<topic>_<metric>.<ext>`, for
  example `plots/backend_agreement_mse.pdf`. Do not encode the full environment,
  hardware list, or implementation details in the filename when the figure
  itself already supplies that context.

## Colors

Series colors are sampled from matplotlib's **plasma** colormap, restricted to
the blue/purple/magenta half (the yellow end is low-contrast on white and is
never used). The darkest sample is brightened slightly so all three pass
colorblind-safety and lightness checks on a white surface
(validated: CVD ΔE ≥ 9.6, normal-vision ΔE ≥ 18.5, contrast ≥ 3:1).

| Role | Hex | Origin |
|---|---|---|
| Series 1 | `#5316d6` | plasma ≈ 0.10, lightened into band |
| Series 2 | `#a11b9b` | plasma 0.35 |
| Series 3 | `#d6556d` | plasma 0.55 |
| Reference / ideal lines | `#999999` | neutral gray, dashed |

Rules:

- Assign colors in this fixed order; never cycle back or generate new hues.
  More than 3 series → rethink the figure (small multiples, or fold into
  "other" in gray).
- Color follows the entity: the same quantity keeps the same color across all
  figures in the paper.
- Reference lines (ideal scaling, targets, baselines) are always gray dashed,
  never a series color.
- Sequential data (e.g. a sweep over one parameter) uses the same plasma
  segment as a ramp: `cm.plasma(np.linspace(0.05, 0.6, n))`.

## Background & chrome

- Pure white figure and axes background (`white`, not off-white/grey) — blends
  into the LaTeX page.
- No top/right spines; remaining spines and ticks in dark gray `#333333`.
- Grid: thin solid light-gray lines behind the data, subtle.
- Legend: no frame.

## Marks & text

- Lines 2 pt, full opacity (no alpha — muddy on white). Markers ~6 pt with a
  thin white edge so overlapping points stay separable; use distinct marker
  shapes (`o`, `s`, `^`) per series as a colorblind fallback.
- DejaVu Sans text (matplotlib default), base size 10 pt; make figures at
  final column width (~3.4 in single column) instead of shrinking big ones.
- Save at 300 dpi, `bbox_inches="tight"` (both set in the style sheet); prefer
  PDF output for the paper, PNG for quick viewing.
