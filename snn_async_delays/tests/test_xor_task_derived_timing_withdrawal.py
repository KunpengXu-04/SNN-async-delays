from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_xor_delay_granularity_level1b as level1b
from scripts import run_xor_task_derived_timing_withdrawal as withdrawal


def test_registered_cell_counts_and_fresh_seeds():
    protocol = withdrawal.load_protocol()
    assert withdrawal.expected_cells(protocol) == {"w0": 5, "w1": 10, "w2": 45}
    assert len(withdrawal.foundation_specs(protocol)) == 5
    assert len(withdrawal.branch_specs(protocol, stage="w1")) == 10
    assert len(withdrawal.branch_specs(protocol, stage="w2")) == 45
    assert protocol["optimization"]["formal_seeds"] == [2503, 2521, 2539, 2551, 2579]


def test_primary_branch_freezes_weights_and_removes_oracle():
    protocol = withdrawal.load_protocol()
    primary = next(
        spec for spec in withdrawal.branch_specs(protocol, stage="w2")
        if spec["condition"] == "task_delay_only" and spec["functional_delay_override"] == 3.0
    )
    assert primary["trainable_components"] == {"input_hidden_delays"}
    assert primary["arrival_auxiliary_weight"] == 0.0
    assert primary["task_loss_weight"] == 1.0
    assert primary["full_batch_updates"] == 500


def test_compensation_controls_cannot_be_confused_with_primary():
    protocol = withdrawal.load_protocol()
    specs = withdrawal.branch_specs(protocol, stage="w2")
    joint = next(spec for spec in specs if spec["condition"] == "task_joint")
    weight_only = next(spec for spec in specs if spec["condition"] == "task_weight_only")
    assert joint["trainable_components"] == {
        "input_hidden_weights", "hidden_output_weights", "input_hidden_delays"
    }
    assert weight_only["trainable_components"] == {
        "input_hidden_weights", "hidden_output_weights"
    }


def test_functional_override_is_exact_and_no_update_is_supported():
    protocol = withdrawal.load_protocol()
    adapted = withdrawal.level1b_protocol(protocol)
    spec = withdrawal._branch_spec(protocol, 99503, "no_update_damage", 3.0, updates=0)
    base = level1b.build_model(adapted, spec).state_dict()
    _, result = level1b.train_cell(
        adapted,
        spec,
        device="cpu",
        initial_state_dict=base,
        functional_delay_override=3.0,
        trainable_components=set(),
        task_loss_weight=1.0,
    )
    assert np.allclose(result["final_record"]["initial_independent_delays"], 3.0)
    assert np.allclose(result["final_record"]["final_independent_delays"], 3.0)
    assert len(result["history"]["step"]) == 1


def test_missing_foundation_checkpoint_blocks_branch(tmp_path: Path):
    protocol = withdrawal.load_protocol()
    spec = withdrawal._branch_spec(protocol, 99503, "no_update_damage", 3.0, updates=0)
    with pytest.raises(RuntimeError, match="missing matching-seed W0 checkpoint"):
        withdrawal.run_branch(protocol, spec, root=tmp_path, device="cpu")
