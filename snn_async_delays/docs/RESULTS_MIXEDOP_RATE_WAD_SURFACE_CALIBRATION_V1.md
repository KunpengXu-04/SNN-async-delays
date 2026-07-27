# Results: Mixed-Operation Rate-WAD Surface Calibration v1

Status: smoke passed; all ten pilot cells completed; the pilot failed its
feasibility, performance and routing gates. The 30-cell surface is not
authorized and was not run. The sealed test split remains closed.

## Design and evidence boundary

This single-seed validation-only pilot replaced the K=5 parent experiment's
fixed four-event one-hot micro-burst with a controlled 4K-channel one-hot rate
code. Selected value channels fired at 400 Hz and unselected channels at 10 Hz
for ten input steps. The resulting validation workload averaged about 8.20
input events/query, so this is not an event-matched comparison.

The five points were the four corners and center of the expanded candidate
domain. Each point used a hand-scheduled fixed oracle and the learned WAD
condition. All optimization and model settings remained frozen.

## Primary results

| Point | N | T | Oracle worst BAcc | WAD worst BAcc | Oracle exact | WAD exact |
|---|---:|---:|---:|---:|---:|---:|
| low N, low T | 20 | 20 | .518 | .500 | .098 | .079 |
| low N, high T | 20 | 70 | .676 | .500 | .379 | .188 |
| high N, low T | 240 | 20 | .565 | .500 | .142 | .117 |
| high N, high T | 240 | 70 | .769 | .500 | .668 | .193 |
| center | 120 | 40 | .632 | .500 | .364 | .178 |

The technical gradient gate passes, but the oracle feasibility gate fails at
all five points: no oracle reaches the registered `.85` worst-query floor.
WAD also fails the performance gate: every point remains exactly `.50`, the
center is below `.60`, and zero of five points exceed `.55`.

## Mechanism diagnosis

At the center, WAD receives nonzero delay gradients on every query in every
update; the median total norm is `.106`. This rules out a literal vanished-
gradient explanation for the continued chance floor. Gradient existence is
not useful temporal credit.

Center per-query balanced accuracies are
`[.960,.947,.501,.500,.500]`. Hidden-activity fractions in the five output
windows are `[1,.991,.123,0,0]`. Rate coding creates some third-window activity
that was absent in the matched burst center, but it does not make query 2
reliable and does not reach windows 3 or 4.

The center query-mean delays are `[2.97,4.41,3.00,3.02,4.14]` steps rather than
the oracle `[0,6,12,18,24]`. Their query-index Spearman correlation is `.40`
and their total spread is only `.239` output windows. Every WAD pilot point has
the same structural pattern: only its first two queries exceed chance, no
activity reaches the last two windows, and query delays remain strongly
overlapped. The routing gate therefore fails.

## Rate-versus-burst center comparison

At the exactly matched `N=120,T=40,w=6` WAD point, the burst parent had worst
BAcc `.50`, mean BAcc `.696`, exact-trial `.200`, and window activity
`[1,.993,0,0,0]`. Rate coding has `.50`, `.682`, `.178`, and
`[1,.991,.123,0,0]`. Thus rate input modestly extends activity into window 2
but worsens the secondary mean and exact metrics and does not rescue the
primary endpoint.

Measured input-to-hidden synaptic events rise from `2400` to `4917` per trial
at this center point (2.05x), while dense MAC and neuron-update proxies remain
`96000` and `4800`. The extra event cost produced no primary reliability gain.

## Decision

The expanded 30-cell WAD surface must not run. A full surface would only map a
known `.50` floor under a rate interface whose own fixed-oracle control has not
met feasibility. The current evidence rejects the proposed rate-code change as
a direct rescue under this frozen ten-step, 400/10 Hz recipe. It does not prove
that all possible rate codes fail; testing another rate, duration or event
budget would be a new calibration question and must not be selected from this
pilot post hoc.

Machine-readable decisions and plots are in
`docs/generated/mixedop_rate_wad_surface_calibration_v1/`. Every pilot cell
contains its runtime NPZ, diagnostic panel, predictions and resource ledger.
