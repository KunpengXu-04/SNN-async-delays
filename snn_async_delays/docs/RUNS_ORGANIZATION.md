# Runs organization and archival policy

`runs/` is an immutable evidence store, not a source-of-truth directory tree.
It is intentionally not physically reorganized during cleanup: many historical
scripts, configs, result paths, and paper references use its current paths.
Moving 1.32GB of artifacts would break provenance while adding no scientific
information.

## Logical states

- `canonical`: promoted only after protocol-compliant re-evaluation.
- `exploratory`: usable for hypothesis generation, not for a final claim.
- `invalid`: smoke, debug, incomplete, or protocol-invalid artifact.
- `archived`: historical/pre-clean material retained for provenance.

The generated registry is `docs/generated/experiment_registry.csv`; regenerate
it with `python -m scripts.build_experiment_registry`.

Registry schema v0.2 recognizes two complete exploratory artifact contracts:
task runs with `eval_results.json`, and diagnostic unit cells with
`metrics.json: complete`, a config, scalar/model state, runtime NPZ and runtime
diagnostic panel. A mechanism cell is not incomplete merely because it has no
task accuracy file. `runs/smoke/` remains invalid for claims even when its
debugging artifacts are technically complete.

## Future paths

New protocol-compliant results belong under
`runs/canonical/<protocol-id>/<experiment-id>/`.  Historical folders remain in
place.  A run is promoted through metadata and the claims ledger, never by
moving or renaming its old directory.

## Historical groups

- `single_op_(step1)`: includes incompatible pre-clean and canonical baselines.
- `NAND_serial_slots_(step2_planA)`: early alignment diagnostic.
- `NAND_simul_channels_(step2_planC)`: simultaneous-channel control.
- `NAND_*planD*`, `NAND_compress_*`, `XOR_compress_*`: sequential Plan-D
  exploratory sweeps; pooled metrics and unequal resources require re-audit.
- `smoke_*`, `analysis_examples`, and `all_results_plots*`: non-confirmatory
  utility artifacts.
- `_archive_*`: explicitly archived historical failures or pre-clean results.
