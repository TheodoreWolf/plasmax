# Physical reset source data

This directory contains the compact, backend-independent source material used
to construct plasmax reset states. Runtime configurations remain in
`configs/envs/`; these files provide reproducible provenance and independent
validation data.

- `iter_baseline_450s_profiles_25.csv` is derived from ITPA TC-33's 450 s IMAS
  file. The source record does not state redistribution terms, so the netCDF is
  checksum-downloaded during regeneration and is not vendored.
- `iter_advanced_slide25_profiles_25.csv` is a deterministic colour-curve
  digitization of page 25 of David Campbell's ITER Physics summer-school deck.
- `sparc_h8_figure16_profiles_25.csv` is a deterministic colour-envelope
  digitization of Figure 16 in Muraca et al. (2025). It stores the pointwise
  10th percentile, median, and 90th percentile of the visible ensemble.
- `sparc_prd_transp_20221013.txt` and
  `sparc_prd_freegs_20221013.eqdsk` are unmodified files from the CFS
  `SPARCPublic` Primary Reference Discharge at commit
  `5b913b8216d05346b22e2e1e87c2ff100761613f`.

The authoritative source URLs, hashes, extraction transforms, locked config
paths, and declared local projections live in `configs/references.yaml`.
