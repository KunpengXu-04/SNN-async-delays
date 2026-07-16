# Preregistration: WAD optimization audit v1

This validation-only audit tests whether the completed negative WAD result was
caused by a poor neuronal firing regime or ineffective delay optimization. It
does not reopen the completed superiority claim and does not inspect test data.

Stage A runs threshold `{0.2,0.3,0.5}` × scalar/WAD × seeds `{0,1,42}`. The
selection rule uses activity, gradient, delay movement, saturation, and a broad
non-floor/non-ceiling accuracy gate. Accuracy does not break ties. Stage B is
locked until the selected threshold and Stage-A evidence are recorded in the
experiment log. It uses seven compact, matched scalar/WAD variants covering
delay range, effective delay learning rate, initialization, warm-up and
alternating optimization.

Before Stage B launch, its initially ambiguous phrase "mean improvement in
2/3 seeds" was resolved as follows: improvement is measured against the fresh
Stage-B WAD baseline; the mean paired gain must be at least .03 and at least
2/3 individual gains must each be at least .03. A retained variant must also
have mean WAD-minus-matched-scalar worst-query >= -.01. If multiple variants
pass, mean WAD worst-query selects them, with a predeclared simplicity order
for differences <= .01. If none passes, baseline is retained and the rescue
attempt is recorded as failed. This clarification predates every Stage-B run.

Every run stores validation reliability, resource ledger, epoch-level weight/
delay gradient diagnostics, effective delay movement, saturation, NPZ, and the
diagnostic panel. Formal test data remain sealed.
