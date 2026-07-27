# Results: temporal WAD scaffold withdrawal W0

Status: complete; the centroid scaffold replicates on all three new seeds and
W1 retention withdrawal is authorized.

W0 independently trains seeds 3657/3669/3691 from the declared all-delay-3
initialization for 400 updates. Weights/readout receive task BCE only; the five
query-tied delays receive arrival-centroid Huber credit only. Every cell is
finite, delay-legal, artifact-complete and uses a fresh optimizer.

The preregistered accuracy-first source selection chooses
`descriptive_best_model.pt` at updates 380/320/400. At those exact source
checkpoints, worst-query balanced accuracy is 1.000/1.000/.99951,
exact-trial accuracy is 1.000/1.000/.99951, minimum output-window activity is
1.0 in every seed, and maximum integer-oracle schedule error is
.50034/.50143/.50000 steps. Thus all seeds jointly pass the function, activity
and <=1-step schedule gate.

The runner initially omitted selected-update metadata from the source sidecar.
This did not affect training, checkpoint selection, or model bytes. Before W1,
the update was deterministically reconstructed by replaying the preregistered
strict comparison over immutable `validation_log.csv`; source hashes were then
recorded. Reconstructed updates are 380/320/400. This provenance correction is
documented rather than hidden.

W0 establishes only that the explicitly supervised scaffold is reproducible.
It says nothing yet about task-only retention. W1 must use the three frozen
`joint_source_model.pt` files and reset optimizer state.

Machine-readable decision:
`docs/generated/mixedop_temporal_wad_scaffold_withdrawal_v1/w0_decision.json`.
