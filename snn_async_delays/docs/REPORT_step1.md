# REPORT - Step 1: Single-op Solvability Baseline

> Generated: 2026-03-07 21:22:21

## Key Notes

- Delay semantics: continuous delay with floor/ceil interpolation (not integer-only hard rounding).
- `weights_only` baseline uses fixed configured delay (recommended and configured as 0 ms for synchronous baseline).

## Mode Summary

| mode | n | acc_mean | acc_std | acc_ci95 | spikes_mean |
|---|---:|---:|---:|---:|---:|
| delays_only | 24 | 0.7370 | 0.1018 | 0.0407 | 1.48 |
| weights_and_delays | 24 | 0.9536 | 0.0389 | 0.0156 | 17.13 |
| weights_only | 24 | 0.7239 | 0.0954 | 0.0381 | 12.24 |

## Grouped Results (op, mode, hidden)

| op | mode | hidden | n_seeds | acc_mean | acc_ci95 | k/spk_mean | density_mean |
|---|---|---:|---:|---:|---:|---:|---:|
| AND | delays_only | 10 | 1 | 0.7840 | 0.0000 | 3.6496 | 0.002000 |
| AND | delays_only | 20 | 1 | 0.8810 | 0.0000 | 0.6549 | 0.001000 |
| AND | delays_only | 50 | 1 | 0.8600 | 0.0000 | 0.4704 | 0.000400 |
| AND | weights_and_delays | 10 | 1 | 0.9620 | 0.0000 | 0.2266 | 0.002000 |
| AND | weights_and_delays | 20 | 1 | 0.9850 | 0.0000 | 0.0836 | 0.001000 |
| AND | weights_and_delays | 50 | 1 | 0.9880 | 0.0000 | 0.0428 | 0.000400 |
| AND | weights_only | 10 | 1 | 0.7810 | 0.0000 | 0.1072 | 0.002000 |
| AND | weights_only | 20 | 1 | 0.7900 | 0.0000 | 0.0621 | 0.001000 |
| AND | weights_only | 50 | 1 | 0.8000 | 0.0000 | 0.0336 | 0.000400 |
| A_IMP_B | delays_only | 10 | 1 | 0.7650 | 0.0000 | 3.4130 | 0.002000 |
| A_IMP_B | delays_only | 20 | 1 | 0.7600 | 0.0000 | 0.5672 | 0.001000 |
| A_IMP_B | delays_only | 50 | 1 | 0.7680 | 0.0000 | 0.3794 | 0.000400 |
| A_IMP_B | weights_and_delays | 10 | 1 | 0.9500 | 0.0000 | 0.1274 | 0.002000 |
| A_IMP_B | weights_and_delays | 20 | 1 | 0.9850 | 0.0000 | 0.0588 | 0.001000 |
| A_IMP_B | weights_and_delays | 50 | 1 | 0.9850 | 0.0000 | 0.0391 | 0.000400 |
| A_IMP_B | weights_only | 10 | 1 | 0.7770 | 0.0000 | 0.0977 | 0.002000 |
| A_IMP_B | weights_only | 20 | 1 | 0.7810 | 0.0000 | 0.0628 | 0.001000 |
| A_IMP_B | weights_only | 50 | 1 | 0.7840 | 0.0000 | 0.0321 | 0.000400 |
| B_IMP_A | delays_only | 10 | 1 | 0.7510 | 0.0000 | 3.5336 | 0.002000 |
| B_IMP_A | delays_only | 20 | 1 | 0.7690 | 0.0000 | 0.5473 | 0.001000 |
| B_IMP_A | delays_only | 50 | 1 | 0.7790 | 0.0000 | 0.3750 | 0.000400 |
| B_IMP_A | weights_and_delays | 10 | 1 | 0.9330 | 0.0000 | 0.0936 | 0.002000 |
| B_IMP_A | weights_and_delays | 20 | 1 | 0.9640 | 0.0000 | 0.0701 | 0.001000 |
| B_IMP_A | weights_and_delays | 50 | 1 | 0.9780 | 0.0000 | 0.0311 | 0.000400 |
| B_IMP_A | weights_only | 10 | 1 | 0.7810 | 0.0000 | 0.0846 | 0.002000 |
| B_IMP_A | weights_only | 20 | 1 | 0.8110 | 0.0000 | 0.0536 | 0.001000 |
| B_IMP_A | weights_only | 50 | 1 | 0.8030 | 0.0000 | 0.0241 | 0.000400 |
| NAND | delays_only | 10 | 1 | 0.7860 | 0.0000 | 4.0486 | 0.002000 |
| NAND | delays_only | 20 | 1 | 0.8570 | 0.0000 | 0.6456 | 0.001000 |
| NAND | delays_only | 50 | 1 | 0.8920 | 0.0000 | 0.4866 | 0.000400 |
| NAND | weights_and_delays | 10 | 1 | 0.9360 | 0.0000 | 0.2014 | 0.002000 |
| NAND | weights_and_delays | 20 | 1 | 0.9650 | 0.0000 | 0.1188 | 0.001000 |
| NAND | weights_and_delays | 50 | 1 | 0.9890 | 0.0000 | 0.0484 | 0.000400 |
| NAND | weights_only | 10 | 1 | 0.7590 | 0.0000 | 0.2591 | 0.002000 |
| NAND | weights_only | 20 | 1 | 0.7850 | 0.0000 | 0.0745 | 0.001000 |
| NAND | weights_only | 50 | 1 | 0.7910 | 0.0000 | 0.0314 | 0.000400 |
| NOR | delays_only | 10 | 1 | 0.7550 | 0.0000 | 3.4247 | 0.002000 |
| NOR | delays_only | 20 | 1 | 0.7560 | 0.0000 | 0.5804 | 0.001000 |
| NOR | delays_only | 50 | 1 | 0.7560 | 0.0000 | 0.3858 | 0.000400 |
| NOR | weights_and_delays | 10 | 1 | 0.9410 | 0.0000 | 0.1032 | 0.002000 |
| NOR | weights_and_delays | 20 | 1 | 0.9860 | 0.0000 | 0.0384 | 0.001000 |
| NOR | weights_and_delays | 50 | 1 | 0.9850 | 0.0000 | 0.0252 | 0.000400 |
| NOR | weights_only | 10 | 1 | 0.7550 | 0.0000 | 1.6835 | 0.002000 |
| NOR | weights_only | 20 | 1 | 0.7560 | 0.0000 | 0.5379 | 0.001000 |
| NOR | weights_only | 50 | 1 | 0.7560 | 0.0000 | 0.3473 | 0.000400 |
| OR | delays_only | 10 | 1 | 0.7560 | 0.0000 | 3.5336 | 0.002000 |
| OR | delays_only | 20 | 1 | 0.7560 | 0.0000 | 0.5426 | 0.001000 |
| OR | delays_only | 50 | 1 | 0.7560 | 0.0000 | 0.3987 | 0.000400 |
| OR | weights_and_delays | 10 | 1 | 0.9650 | 0.0000 | 0.0532 | 0.002000 |
| OR | weights_and_delays | 20 | 1 | 0.9780 | 0.0000 | 0.0375 | 0.001000 |
| OR | weights_and_delays | 50 | 1 | 0.9860 | 0.0000 | 0.0243 | 0.000400 |
| OR | weights_only | 10 | 1 | 0.7560 | 0.0000 | 3.0864 | 0.002000 |
| OR | weights_only | 20 | 1 | 0.7560 | 0.0000 | 0.3220 | 0.001000 |
| OR | weights_only | 50 | 1 | 0.7560 | 0.0000 | 0.3370 | 0.000400 |
| XNOR | delays_only | 10 | 1 | 0.5170 | 0.0000 | 3.1847 | 0.002000 |
| XNOR | delays_only | 20 | 1 | 0.6170 | 0.0000 | 0.6349 | 0.001000 |
| XNOR | delays_only | 50 | 1 | 0.6070 | 0.0000 | 0.3873 | 0.000400 |
| XNOR | weights_and_delays | 10 | 1 | 0.8460 | 0.0000 | 0.2081 | 0.002000 |
| XNOR | weights_and_delays | 20 | 1 | 0.9330 | 0.0000 | 0.0914 | 0.001000 |
| XNOR | weights_and_delays | 50 | 1 | 0.9520 | 0.0000 | 0.0510 | 0.000400 |
| XNOR | weights_only | 10 | 1 | 0.5270 | 0.0000 | 1.1377 | 0.002000 |
| XNOR | weights_only | 20 | 1 | 0.5800 | 0.0000 | 0.1324 | 0.001000 |
| XNOR | weights_only | 50 | 1 | 0.5920 | 0.0000 | 0.0632 | 0.000400 |
| XOR | delays_only | 10 | 1 | 0.5340 | 0.0000 | 3.8168 | 0.002000 |
| XOR | delays_only | 20 | 1 | 0.6240 | 0.0000 | 0.6211 | 0.001000 |
| XOR | delays_only | 50 | 1 | 0.6020 | 0.0000 | 0.3873 | 0.000400 |
| XOR | weights_and_delays | 10 | 1 | 0.8600 | 0.0000 | 0.2006 | 0.002000 |
| XOR | weights_and_delays | 20 | 1 | 0.9030 | 0.0000 | 0.1284 | 0.001000 |
| XOR | weights_and_delays | 50 | 1 | 0.9320 | 0.0000 | 0.0507 | 0.000400 |
| XOR | weights_only | 10 | 1 | 0.5550 | 0.0000 | 0.2349 | 0.002000 |
| XOR | weights_only | 20 | 1 | 0.5470 | 0.0000 | 0.2051 | 0.001000 |
| XOR | weights_only | 50 | 1 | 0.5950 | 0.0000 | 0.0661 | 0.000400 |