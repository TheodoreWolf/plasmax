# Equilibrium generation venv (FreeGSnke)

`tools/equilibria/generate_equilibrium.py` produces the per-scenario EQDSK
geometry files in `src/plasmax/configs/data/` using
[FreeGSnke](https://github.com/FusionComputingLab/freegsnke), a free-boundary
Grad–Shafranov solver.

**Why a separate venv?** FreeGSnke pins `numpy<2` while TORAX requires `numpy>2`
— they cannot coexist. Installing FreeGSnke into the main project venv breaks
TORAX (and makes the project unresolvable). So FreeGSnke lives here, isolated.
The generated `.eqdsk` files are committed, so training never needs this venv.

## Create the venv

```bash
uv venv tools/equilibria/.venv --python 3.12
VIRTUAL_ENV=tools/equilibria/.venv uv pip install "freegsnke[freegs4e]" tyro
```

(The `[freegs4e]` extra is required — it provides the GS solver backend and the
`freegs4e.geqdsk` EQDSK writer.)

## Generate equilibria

```bash
# all scenarios -> src/plasmax/configs/data/{iter_baseline,iter_hybrid,iter_advanced,sparc_prd,sparc_reduced_field}_ip*.eqdsk
tools/equilibria/.venv/bin/python tools/equilibria/generate_equilibrium.py

# a single scenario
tools/equilibria/.venv/bin/python tools/equilibria/generate_equilibrium.py --scenario iter_hybrid
```

## Validate (in the main venv)

```bash
uv run pytest tests/geometry_eqdsk_test.py -v
```

This reads each EQDSK back through TORAX and checks physical R0/a/B0/Ip/q — it
needs only TORAX, not FreeGSnke.

## Notes

- Machine descriptions (PF coils, passive structures, and limiter/wall) are
  vendored under `tools/equilibria/machines/{ITER,SPARC}` — see that directory's
  README. The upstream limiter and wall files were byte-identical for both
  machines, so the generator reuses the retained limiter artifact for both
  FreeGSnke inputs.
- `.venv/` here is git-ignored; recreate it with the commands above.
