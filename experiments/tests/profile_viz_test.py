"""Tests for the profile-vs-rho figure builder.

These use synthetic rho/profile arrays (no TORAX sim) so they run fast and
exercise the Plotly figure structure and time-color mapping.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from experiments.plotting.viz import (
    PROFILE_LABELS,
    make_profile_rho_figure,
)

N_RHO = 12
N_FRAMES = 8


def _make_profiles(n_profiles: int | None = None) -> dict[str, np.ndarray]:
    """Per-profile (n_frames, n_rho) arrays that ramp over time."""
    names = list(PROFILE_LABELS)
    if n_profiles is not None:
        names = names[:n_profiles]
    rng = np.random.default_rng(0)
    profiles = {}
    for k, name in enumerate(names):
        base = np.linspace(1.0, 0.1, N_RHO) ** 2
        ramp = np.linspace(0.2, 1.0, N_FRAMES)[:, None]
        noise = 0.01 * rng.standard_normal((N_FRAMES, N_RHO))
        profiles[name] = (1.0 + k) * ramp * base[None, :] + noise
    return profiles


class ProfileVizTest:
    def setup_method(self):
        self.rho = np.linspace(0.02, 0.98, N_RHO)
        self.profiles = _make_profiles()
        self.t = np.arange(N_FRAMES) * 0.5

    def test_returns_figure_with_panels_and_lines(self):
        fig = make_profile_rho_figure(self.rho, self.profiles, self.t)
        assert isinstance(fig, go.Figure)
        # One line per (profile, frame), plus one colorbar marker trace.
        assert len(fig.data) == len(PROFILE_LABELS) * N_FRAMES + 1
        titles = [a.text for a in fig.layout.annotations]
        for label in PROFILE_LABELS.values():
            assert label in titles
        # Static figure: no animation frames.
        assert not fig.frames

    def test_lines_colored_by_time(self):
        fig = make_profile_rho_figure(self.rho, self.profiles, self.t)
        # First profile's first vs last frame get different colors.
        line_colors = [d.line.color for d in fig.data if d.mode == "lines"]
        assert line_colors[0] != line_colors[N_FRAMES - 1]

    def test_subset_of_profiles(self):
        subset = _make_profiles(n_profiles=2)
        fig = make_profile_rho_figure(self.rho, subset, self.t)
        titles = [a.text for a in fig.layout.annotations]
        assert len(titles) == 2

    def test_unknown_profiles_raises(self):
        with pytest.raises(ValueError):
            make_profile_rho_figure(
                self.rho, {"bogus": np.ones((N_FRAMES, N_RHO))}, self.t
            )
