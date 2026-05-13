"""Collect multi-seed Plan D results and print mean ± std tables."""
import json, os, glob, numpy as np

runs_dir = "runs"
planD_dir = "runs/step2_planD"

def find_eval(h, K, seed):
    name = f"step2_seq_NAND_w_and_d_continuous_h{h}_K{K}_sw10_seed{seed}"
    for base in [runs_dir, planD_dir]:
        p = os.path.join(base, name, "eval_results.json")
        if os.path.exists(p):
            return p
    return None

def collect(h, K_list, seeds):
    results = {}
    for K in K_list:
        accs = []
        for seed in seeds:
            p = find_eval(h, K, seed)
            if p is None:
                print(f"  MISSING: h={h} K={K} seed={seed}")
                continue
            with open(p) as f:
                d = json.load(f)
            accs.append(d["accuracy"])
        results[K] = accs
    return results

print("=" * 60)
print("Plan D multi-seed results (w_and_d_continuous, NAND)")
print("Seeds: 0, 1, 2, 42")
print("=" * 60)

for h, K_list in [(20, [1, 2, 3]), (50, [2, 3])]:
    seeds = [0, 1, 2, 42]
    res = collect(h, K_list, seeds)
    print(f"\nh={h}:")
    print(f"  {'K':>3}  {'seeds':>30}  {'mean':>7}  {'std':>6}")
    print(f"  {'-'*3}  {'-'*30}  {'-'*7}  {'-'*6}")
    for K, accs in res.items():
        arr = np.array(accs)
        seed_str = "  ".join(f"{a:.4f}" for a in accs)
        print(f"  {K:>3}  {seed_str:>30}  {arr.mean():.4f}  {arr.std():.4f}")

# Also check if there's a seed=42 K=1 h=50 run
print("\n--- h=50 K=1 seed=42 check ---")
p = os.path.join(runs_dir, "step2_seq_NAND_w_and_d_continuous_h50_K1_sw10_seed42", "eval_results.json")
if os.path.exists(p):
    with open(p) as f: d = json.load(f)
    print(f"  acc={d['accuracy']:.4f}")
else:
    print("  not found (expected)")
