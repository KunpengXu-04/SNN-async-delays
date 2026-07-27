"""Registered SLAYER S0 technical verification (scientifically invalid smoke)."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from snn.backend_adapter import ModelBackendAdapter
from snn.slayer_backend import (
    SlayerNativeWindowedModel, btc_to_nct, configure_slayer_portable_shift,
    nct_to_btc,
)


BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    str(BASE / "runs" / "smoke" / "spatial_temporal_capacity_slayer_v1" / "_torch_extensions"),
)
# PyTorch's extension loader searches PATH rather than importing the Python
# ``ninja`` package. Conda-launched scripts do not always prepend Scripts/.
_env_scripts = str(Path(sys.executable).resolve().parent / "Scripts")
if _env_scripts not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _env_scripts + os.pathsep + os.environ.get("PATH", "")


def _delay_probe(initial_delay: float, *, device: torch.device) -> dict[str, Any]:
    from lava.lib.dl import slayer
    shift_backend = configure_slayer_portable_shift()

    module = slayer.axon.Delay(max_delay=8).to(device)
    module.delay = torch.nn.Parameter(torch.tensor([initial_delay], device=device))
    module.init = True
    spike = torch.zeros(1, 1, 10, device=device)
    spike[0, 0, 2] = 1.0
    delayed = module(spike)
    times = torch.arange(10, device=device, dtype=delayed.dtype).reshape(1, 1, -1)
    centroid = (delayed * times).sum() / delayed.sum()
    loss = (centroid - 5.0).square()
    loss.backward()
    return {
        "initial_delay": initial_delay,
        "arrival_step": int(delayed.argmax(dim=-1).item()),
        "centroid_step": float(centroid.detach().cpu()),
        "delay_gradient": float(module.delay.grad.detach().cpu().item()),
        "event_count": float(delayed.detach().sum().cpu()),
        "train": delayed.detach().cpu().flatten().tolist(),
        "shift_backend": shift_backend,
    }


def _environment_record() -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "lava_dl": importlib.metadata.version("lava-dl"),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pip_freeze": freeze,
    }


def _checkpoint_probe(seed: int, output: Path) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = SlayerNativeWindowedModel(
        n_queries=1, n_hidden=3, win_len=10, read_len=8, d_max=7,
        delay_method="task_only_slayer", delay_init=3.0,
    )
    spikes = torch.zeros(2, 18, 4)
    spikes[:, 6:10, 0] = 1.0
    spikes[:, 6:10, 2] = 1.0
    logits_before, info_before, ledger = ModelBackendAdapter(model).forward(
        spikes, record=True
    )
    checkpoint = output / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "seed": seed}, checkpoint)

    torch.manual_seed(seed + 1)
    restored = SlayerNativeWindowedModel(
        n_queries=1, n_hidden=3, win_len=10, read_len=8, d_max=7,
        delay_method="task_only_slayer", delay_init=0.0,
    )
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    logits_after, info_after = restored(spikes, record=True)
    checks = {
        "logits_exact": torch.equal(logits_before, logits_after),
        "hidden_spikes_exact": torch.equal(
            info_before["hidden_spike_train"], info_after["hidden_spike_train"]
        ),
        "delayed_inputs_exact": torch.equal(
            info_before["delayed_input_spike_train"],
            info_after["delayed_input_spike_train"],
        ),
        "delays_exact": torch.equal(
            model.get_delays()["input_axons"], restored.get_delays()["input_axons"]
        ),
    }
    (output / "resource_ledger.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8"
    )
    return {"checks": checks, "checkpoint": str(checkpoint)}


def run_one(seed: int, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    environment = _environment_record()
    early = _delay_probe(1.0, device=torch.device("cpu"))
    late = _delay_probe(4.0, device=torch.device("cpu"))
    fixed = _delay_probe(3.0, device=torch.device("cpu"))
    direction_checks = {
        "early_gradient_increases_delay": early["delay_gradient"] < 0.0,
        "late_gradient_decreases_delay": late["delay_gradient"] > 0.0,
        "fixed_arrival_matches_declared": fixed["arrival_step"] == 5,
        "fixed_event_preserved": fixed["event_count"] == 1.0,
    }

    tensor = torch.zeros(2, 7, 5)
    tensor[0, 3, 4] = 1.0
    conversion_checks = {
        "btc_nct_roundtrip_exact": torch.equal(nct_to_btc(btc_to_nct(tensor)), tensor),
        "event_count_preserved": float(btc_to_nct(tensor).sum()) == float(tensor.sum()),
    }
    checkpoint = _checkpoint_probe(seed, output)

    cuda_probe: dict[str, Any]
    if torch.cuda.is_available():
        cpu = _delay_probe(3.0, device=torch.device("cpu"))
        gpu = _delay_probe(3.0, device=torch.device("cuda"))
        cuda_probe = {
            "available": True,
            "forward_exact": cpu["train"] == gpu["train"],
            "gradient_close": abs(cpu["delay_gradient"] - gpu["delay_gradient"]) < 1e-6,
        }
    else:
        cuda_probe = {
            "available": False, "forward_exact": None, "gradient_close": None,
            "reason": "isolated Lava-DL environment currently has a CPU-only torch build",
        }

    implemented_checks_pass = all(direction_checks.values()) and all(
        conversion_checks.values()
    ) and all(checkpoint["checks"].values())
    complete_gate_pass = implemented_checks_pass and bool(cuda_probe["available"]) and bool(
        cuda_probe["forward_exact"] and cuda_probe["gradient_close"]
    )
    result = {
        "protocol_id": "spatial_temporal_capacity_slayer_v1",
        "stage": "s0",
        "claim_status": "invalid_smoke",
        "seed": seed,
        "environment": environment,
        "early_probe": early,
        "late_probe": late,
        "fixed_probe": fixed,
        "direction_checks": direction_checks,
        "conversion_checks": conversion_checks,
        "checkpoint_probe": checkpoint,
        "cpu_cuda_probe": cuda_probe,
        "implemented_checks_pass": implemented_checks_pass,
        "complete_s0_gate_pass": complete_gate_pass,
    }
    (output / "s0_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output / "environment_lock.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    return result


def run_registered_s0(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    root = BASE / protocol["execution"]["smoke_root"] / "s0"
    results = []
    for seed in protocol["gates"]["s0"]["seeds"]:
        results.append(run_one(int(seed), root / f"seed{int(seed)}"))
    summarize_s0(protocol)
    print(json.dumps(results, indent=2))
    return results


def summarize_s0(protocol: dict[str, Any]) -> dict[str, Any]:
    root = BASE / protocol["execution"]["smoke_root"] / "s0"
    rows = []
    for seed in protocol["gates"]["s0"]["seeds"]:
        path = root / f"seed{int(seed)}" / "s0_result.json"
        if not path.exists():
            rows.append({"seed": int(seed), "complete": False, "passed": False})
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        supplement_path = path.parent / "full_model_cpu_cuda_supplement.json"
        supplement = (
            json.loads(supplement_path.read_text(encoding="utf-8"))
            if supplement_path.exists() else {"passed": False}
        )
        rows.append({
            "seed": int(seed), "complete": True,
            "passed": bool(result["complete_s0_gate_pass"] and supplement["passed"]),
            "cpu_cuda_forward_exact": result["cpu_cuda_probe"]["forward_exact"],
            "cpu_cuda_gradient_close": result["cpu_cuda_probe"]["gradient_close"],
            "full_model_cpu_cuda_passed": supplement["passed"],
        })
    passed = len(rows) == len(protocol["gates"]["s0"]["seeds"]) and all(
        row["complete"] and row["passed"] for row in rows
    )
    decision = {
        "protocol_id": "spatial_temporal_capacity_slayer_v1",
        "stage": "s0", "invalid_for_scientific_claims": True,
        "rows": rows, "passed": passed,
        "s1_calibration_authorized_by_results": passed,
        "s1_calibration_launch_still_requires_yaml_unlock": not bool(
            protocol["authorization"]["s1_calibration_launch"]
        ),
    }
    output = BASE / protocol["execution"]["generated_root"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "s0_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return decision


def add_full_model_cuda_supplements(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Add, never overwrite, full CUBA/LIF CPU-CUDA parity to completed S0 cells."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered supplement")
    root = BASE / protocol["execution"]["smoke_root"] / "s0"
    rows = []
    for seed in protocol["gates"]["s0"]["seeds"]:
        directory = root / f"seed{int(seed)}"
        output = directory / "full_model_cpu_cuda_supplement.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        checkpoint = torch.load(
            directory / "checkpoint.pt", map_location="cpu", weights_only=True
        )["model"]
        cpu = SlayerNativeWindowedModel(
            n_queries=1, n_hidden=3, win_len=10, read_len=8, d_max=7,
            delay_method="task_only_slayer", delay_init=0.0,
        )
        gpu = SlayerNativeWindowedModel(
            n_queries=1, n_hidden=3, win_len=10, read_len=8, d_max=7,
            delay_method="task_only_slayer", delay_init=0.0,
        ).cuda()
        cpu.load_state_dict(checkpoint)
        gpu.load_state_dict(checkpoint)
        spikes = torch.zeros(2, 18, 4)
        spikes[:, 6:10, 0] = 1.0
        spikes[:, 6:10, 2] = 1.0
        cpu_logits, cpu_info = cpu(spikes, record=True)
        gpu_logits, gpu_info = gpu(spikes.cuda(), record=True)
        cpu_logits.sum().backward()
        gpu_logits.sum().backward()
        cpu_gradient = cpu.axon_delay.delay.grad.detach().cpu()
        gpu_gradient = gpu.axon_delay.delay.grad.detach().cpu()
        row = {
            "seed": int(seed),
            "portable_backend": "slayer_portable_torch_delay_and_cuba_dynamics",
            "logits_exact": torch.equal(cpu_logits, gpu_logits.cpu()),
            "hidden_spikes_exact": torch.equal(
                cpu_info["hidden_spike_train"], gpu_info["hidden_spike_train"]
            ),
            "delay_gradients_close_atol_1e-6": torch.allclose(
                cpu_gradient, gpu_gradient, rtol=1e-4, atol=1e-6
            ),
            "maximum_delay_gradient_absolute_difference": float(
                (cpu_gradient - gpu_gradient).abs().max()
            ),
        }
        row["passed"] = bool(
            row["logits_exact"] and row["hidden_spikes_exact"]
            and row["delay_gradients_close_atol_1e-6"]
        )
        output.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
    summarize_s0(protocol)
    return rows


if __name__ == "__main__":
    from scripts.run_spatial_temporal_capacity_slayer import load_protocol

    run_registered_s0(load_protocol())
