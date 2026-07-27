# Mixed-op temporal WAD repair v3

Status: complete. Oracle and arrival-centroid Huber pass 3/3 new seeds; the old
arrival-mass CE fails 0/3. See `RESULTS_MIXEDOP_TEMPORAL_WAD_REPAIR_V3.md`.

## Diagnosis motivating this protocol

V2 did not fail merely because of insufficient updates. Its decoupled
arrival-mass CE arm reached perfect classification on all three seeds, but its
best learned schedules were approximately `[1.0, 7.5, 11.5, 15.3, 18.2]`
rather than `[3,7,11,15,19]`. The mass-ratio objective rewards keeping the
first packet away from later windows, so its mathematical optimum moves q0
earlier than the registered oracle. The training objective and acceptance gate
were therefore inconsistent.

## Frozen repair

V3 keeps the event8/N120/w4/T30 interface, per-query tying, delay support
`[0,23]`, delay-3 initialization, LR 0.01 and 400 updates. Only the explicit
routing objective changes. For query q, define the event-weighted mean synaptic
arrival time

\[
\bar a_q =
\frac{\sum_{t,i,h} x_{tqi}(t+1+d_q)}{\sum_{t,i,h}x_{tqi}},
\]

and the center of output window q as

\[
c_q = 10 + 4q + 2.
\]

The delay-only loss is

\[
L_{route}=\frac{1}{K}\sum_q
\operatorname{Huber}_{0.25}\left(\frac{\bar a_q-c_q}{4}\right).
\]

Weights and the MLP readout receive only task BCE gradients; the five delay
parameters receive only this explicit routing loss. This is supervised routing,
not autonomous WAD.

## Stages and gates

1. Zero-update preflight on seeds 3507/3513/3533: all five raw-delay gradients
   must point toward their window centers.
2. Three invalid smoke cells on seed 3549: oracle, old arrival-mass CE, and new
   centroid-Huber.
3. If smoke is technically clean, nine new-seed formal cells on
   3557/3569/3591 with the same three arms.

For a learned arm to pass, one single saved checkpoint per seed must jointly
satisfy worst-query balanced accuracy >= .85, exact-trial accuracy >= .70,
minimum window activity fraction >= .10, and maximum delay-schedule error <= 1
step. The accuracy-best or schedule-best checkpoint may qualify, but metrics
may not be combined across checkpoints. Sealed test and publication claims stay
locked.
