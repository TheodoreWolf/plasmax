"""Profile-vs-rho visualization for TORAX profiles over an episode.

Builds a single static Plotly figure with one panel per profile (T_e, T_i,
n_e, q). Each panel overlays the radial profile at several episode times, the
lines coloured along a shared time colorscale — a clean "profiles over the
episode" view that reads at a glance and survives ``wandb.Plotly`` (no
animation frames to strip).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

__all__ = [
    "PROFILE_LABELS",
    "make_profile_rho_figure",
]

# Ordered profile name -> y-axis label. Order defines subplot placement
# (row-major in a 2x2 grid).
PROFILE_LABELS: dict[str, str] = {
    "T_e": "T_e [keV]",
    "T_i": "T_i [keV]",
    "n_e": "n_e [m⁻³]",
    "q": "q",
}

_COLORSCALE = "Viridis"
_N_ROWS = 2
_N_COLS = 2


def make_profile_rho_figure(
    rho: np.ndarray,
    profiles: dict[str, np.ndarray],
    frame_times: np.ndarray,
) -> go.Figure:
    """Overlay each profile vs rho at successive episode times.

    Args:
      rho: Normalized radial coordinate on the cell grid, shape ``(n_rho,)``.
      profiles: Mapping of profile name -> array ``(n_frames, n_rho)``. Keys
        must be a subset of :data:`PROFILE_LABELS`; subplot order follows it.
      frame_times: Time [s] of each frame, shape ``(n_frames,)`` — sets the
        line colors along the time colorscale.

    Returns:
      A static Plotly ``Figure`` (no animation frames).
    """
    names = [p for p in PROFILE_LABELS if p in profiles]
    if not names:
        raise ValueError(
            f"No known profiles in {list(profiles)}; "
            f"expected a subset of {list(PROFILE_LABELS)}."
        )
    rho = np.asarray(rho)
    times = np.asarray(frame_times, dtype=float)
    n_frames = times.shape[0]
    tmin, tmax = float(times.min()), float(times.max())
    norm = (times - tmin) / (tmax - tmin) if tmax > tmin else np.full(n_frames, 0.5)
    colors = sample_colorscale(_COLORSCALE, norm.tolist())

    fig = make_subplots(
        rows=_N_ROWS,
        cols=_N_COLS,
        subplot_titles=[PROFILE_LABELS[n] for n in names],
        horizontal_spacing=0.1,
        vertical_spacing=0.13,
    )
    for p_i, name in enumerate(names):
        row, col = p_i // _N_COLS + 1, p_i % _N_COLS + 1
        arr = np.asarray(profiles[name])  # (n_frames, n_rho)
        for f_i in range(n_frames):
            fig.add_trace(
                go.Scatter(
                    x=rho,
                    y=arr[f_i],
                    mode="lines",
                    line=dict(color=colors[f_i], width=1.5),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="ρ_norm", row=row, col=col)
        fig.update_yaxes(title_text=PROFILE_LABELS[name], row=row, col=col)

    # Single shared colorbar mapping line color -> episode time.
    fig.add_trace(
        go.Scatter(
            x=[rho[0]],
            y=[np.asarray(profiles[names[0]])[0, 0]],
            mode="markers",
            marker=dict(
                colorscale=_COLORSCALE,
                cmin=tmin,
                cmax=tmax,
                color=[tmin],
                size=0.1,
                showscale=True,
                colorbar=dict(title=dict(text="t [s]", side="right"), thickness=12),
            ),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        height=680,
        width=980,
        title_text="Profiles vs ρ over the episode",
        margin=dict(t=70),
    )
    return fig
