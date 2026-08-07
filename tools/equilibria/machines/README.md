# Vendored FreeGSnke machine descriptions

Tokamak machine descriptions (active PF coils, passive structures, and the shared
limiter/wall contour) consumed by `tools/equilibria/generate_equilibrium.py` via
`freegsnke.build_machine.tokamak`.

## Provenance

Copied verbatim from the [FreeGSnke](https://github.com/FusionComputingLab/freegsnke)
repository, `machine_configs/`, at commit
`fa51e33c77d5d7f9dad64dacf3dbb2df9dbc9425`.

| Dir | Source | Files | Retained limiter SHA-256 |
|---|---|---|---|
| `ITER/` | FreeGSnke (from FUSE.jl) | `ITER_{active_coils,passive_coils,limiter}.pickle` | `8c182b5cfd19142ba5e573960b52172cf8d67a870332a82bd5d0ee67439a3d64` |
| `SPARC/` | FreeGSnke (from SPARCPublic) | `SPARC_{active_coils,passive_coils,limiter}.pickle` | `2d345bc8b67dbf77b4e0715461ccf1c185d9fd8319d931836d9f57faa1d9126c` |

In the upstream machine descriptions, each machine's `wall` and `limiter`
pickle was byte-for-byte identical. plasmax retains only the limiter copy and
passes that same path to both FreeGSnke arguments.

These are Python pickles loaded by FreeGSnke at equilibrium-generation time only;
they are not imported during RL training.
