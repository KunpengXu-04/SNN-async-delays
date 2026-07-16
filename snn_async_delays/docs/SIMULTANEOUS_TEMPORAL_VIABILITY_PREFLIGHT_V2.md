# Preregistration: simultaneous temporal viability preflight v2

> Completion note: all mechanism-valid gates passed separately in both held-
> out seeds. This establishes viability only; WAD worst-window balanced
> accuracy remained `.5`. See
> `RESULTS_SIMULTANEOUS_TEMPORAL_VIABILITY_PREFLIGHT_V2.md`.

## Why v2 is necessary

Preflight v1 formally failed because WAD emitted no hidden spikes during output
window 3. The frozen-checkpoint audit showed that this was the wrong proxy for
arrival support: earlier hidden spikes traverse `syn_ho` delays and produce
nonzero third-window current in 70.3% of exhaustive trials. V2 therefore gates
the quantity received by the output layer, not the time at which its
presynaptic hidden spike was emitted.

The audit also identified an objective mismatch. Training pooled ordinary BCE
over XOR/NAND/NOR, whose positive prevalences are `.50/.75/.25`, while the
primary evaluation is macro balanced. V2 freezes a window/class-balanced BCE
before seeing its held-out seeds. This is an interface correction, not a WAD-
specific rescue; every v2 condition uses it.

## Design and data separation

Seed 0 was used by v1 and the mechanism audit and is excluded. V2 uses held-out
execution seeds `{1,42}` and three conditions, producing six cells:

1. full-support frozen heterogeneous `[0,30]`;
2. temporal scaffold positive execution control;
3. WAD with the unchanged frozen threshold `.3`, `dmax=30`, delay LR `.001`,
   constant initialization and joint optimization.

All cells use one shared opponent output pair, output threshold `.2`, 100
epochs, the same 20-epoch membrane warm-up, auxiliary weight `.2`, final
checkpoint, burst encoding and identical data budgets. Accuracy cannot select
a condition or pass this preflight.

## Mechanism-valid gates

For every held-out seed separately:

- scaffold and fixed-full must have output synaptic-current support in at least
  50% of exhaustive trials and output spikes in at least 10% in every window;
- fixed-full must retain nonzero window-specific output-weight gradients;
- WAD must meet the same current/output activity gates;
- at least 5% of normalized arrivals produced by actually observed hidden
  spikes must land in each WAD output window;
- every WAD window must have nonzero output-weight and total delay gradients;
- WAD final mean delay movement must be at least `.05` and saturation below
  `.95`.

The current-support statistic is computed from `I_o` after delayed `syn_ho`;
the realized-arrival statistic uses actual hidden spike times and interpolated
hidden-output delays. Hidden emission in the same window remains descriptive.

The `.50/.10/.05` thresholds are explicitly audit-informed using seed 0. They
are tested only on seeds 1 and 42, and both seeds must pass without averaging.

## Prohibited interpretations

- Passing means the interface has support and gradients; it does not mean WAD
  is accurate or superior.
- The scaffold cannot support a routing claim.
- A failed seed cannot be removed or rescued by mean performance.
- No threshold, initialization, delay range or gate may be changed after v2
  begins.
- Test remains sealed.

## Commands

```powershell
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_simultaneous_temporal_viability_preflight_v2 --dry-run
D:\anaconda3\envs\snn_async\python.exe -m scripts.run_simultaneous_temporal_viability_preflight_v2 --device cuda
D:\anaconda3\envs\snn_async\python.exe -m scripts.summarize_simultaneous_temporal_viability_preflight_v2
```
