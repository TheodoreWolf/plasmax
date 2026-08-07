# Plotting scripts

This directory contains the repository's plotting and figure-generation
entrypoints together with their shared `paper.mplstyle`.

Run scripts from the repository root:

```bash
uv run python experiments/plotting/<name>.py --help
```

The scripts read experiment data from `outputs/` or the packaged configuration
files and write rendered figures under `plots/`.
