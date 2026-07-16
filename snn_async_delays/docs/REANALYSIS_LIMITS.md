# Historical reanalysis limits

`scripts/audit_existing_results.py` can recover a worst-query estimate only
where an old `eval_results.json` saved `per_query_acc`.  It cannot reconstruct
exact-trial correctness, balanced accuracy, confidence intervals, paired
stochastic encodings, or a sealed test protocol from aggregate files alone.

Consequently, the generated historical audit is a triage artifact.  It can
invalidate pooled-accuracy claims, but it cannot upgrade a historical result
to confirmatory evidence.  New evaluations must save the metrics specified in
`METRICS_AND_COST.md`; later protocol versions should additionally save
per-sample predictions in a separate, versioned artifact.
