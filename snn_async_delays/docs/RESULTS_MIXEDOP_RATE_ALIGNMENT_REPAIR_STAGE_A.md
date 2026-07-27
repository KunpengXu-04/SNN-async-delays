# Results: Mixed-Operation Rate Alignment Repair v1, Stage A

Status: all 48 fixed-oracle cells are complete. Eight candidates pass and the
preregistered resource rule selects `event8_N120_w4`. Stage B is authorized;
Stages C and sealed test remain locked.

## Main result

Analytically aligning the late-rate packet repairs the fixed-oracle interface.
The four-step packet occupies input steps 6--9 and the schedule
`d_q=3+q*w` puts every event inside its declared output window.

The result depends sharply on event redundancy:

- all eight event4 candidates fail because exact-trial accuracy is only
  `.663--.736`, although worst-query balanced accuracy is `.890--.917` and all
  windows are active;
- all eight event8 candidates pass all three seeds. Their minimum worst-query
  balanced accuracy is at least `.99967`, minimum exact-trial accuracy is at
  least `.99951`, and every window is active in every validation trial.

Thus the previous rate-oracle failure was substantially caused by temporal
misalignment, but four expected events/query remain too noisy for reliable
five-query conjunction. Increasing N or w does not repair event4. Near-
deterministic event8 saturates the tested oracle grid, so Stage A supplies no
width-versus-time law.

## Registered selection

All passing event8 candidates have the same mean measured input event count,
approximately `7.997` events/query. The next selection key is the
`N*T` neuron-update proxy. The selected interface is therefore:

| field | selected value |
|---|---:|
| event code | event8, 990/10 Hz over steps 6--9 |
| total hidden neurons | 120 |
| output-window length | 4 |
| total latency | 30 |
| delay schedule | `[3,7,11,15,19]` |
| minimum worst-query BAcc | 1.0 |
| minimum exact-trial | 1.0 |
| minimum window activity | 1.0 |
| `N*T` proxy | 3600 |

This encoding is a very high-rate, short packet: selected channels fire with
probability `.99` per step. It is closer to a stochastic micro-burst than to a
long low-rate code. Its success must not be generalized to arbitrary rate
coding or treated as an energy advantage; it uses about twice the input events
of event4.

## Scientific decision

Stage B may now test whether five query-tied delays learn the schedule on three
new seeds. It must compare fixed oracle, task-only and explicitly routing-
assisted learning. A routing-assisted success is supervised schedule learning,
not autonomous temporal multiplexing. Per-synapse WAD and any N/T surface
remain locked.

Machine-readable evidence is in
`docs/generated/mixedop_rate_alignment_repair_v1/stage_a_decision.json` and
`stage_a_cells.csv`. All 48 cells contain checkpoint, predictions, runtime NPZ,
diagnostic panel and resource ledger.
