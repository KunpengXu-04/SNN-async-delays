"""
Training engine.

Supports:
  - Separate learning rates for weights / delays / readout
  - Spike penalty (energy proxy)
  - Delay L1 penalty
  - Gradient clipping
  - Checkpoint saving (best val accuracy)
"""

from __future__ import annotations
import os
import csv
import json
import time
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from snn.model import SNNModel, SNNSimultaneousModel, SlotBoundaries
from data.encoding import encode_trial, encode_simultaneous_trial


# ---------------------------------------------------------------------------

def window_class_balanced_mean(
    per_sample_window_loss: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Macro-average a [batch, window] loss over windows and label classes."""
    if per_sample_window_loss.shape != labels.shape:
        raise ValueError("per-sample loss and labels must have identical [B,K] shape")
    terms = []
    for q in range(labels.shape[1]):
        for target in (0.0, 1.0):
            mask = labels[:, q] == target
            if bool(mask.any()):
                terms.append(per_sample_window_loss[mask, q].mean())
    if not terms:
        raise ValueError("window-class-balanced loss received no labeled samples")
    return torch.stack(terms).mean()


def window_class_balanced_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return window_class_balanced_mean(losses, labels)


def opponent_target_spike_train(
    labels: torch.Tensor,
    *,
    total_steps: int,
    input_steps: int,
    output_window_len: int,
    timing_mode: str,
    target_offset_steps: int | None = None,
) -> torch.Tensor:
    """Create one class-opponent target spike per query and trial.

    Unlike a zero-spike/one-spike code, both classes receive exactly one target
    event. Centre modes target window centres. Offset modes target a declared
    number of steps after the start of each output window and are used when a
    d0 spatial baseline must remain causally capable of emitting the target.
    """
    if labels.dim() != 2:
        raise ValueError("labels must have shape [B,K]")
    valid_modes = {
        "simultaneous_center", "sequential_centers",
        "simultaneous_offset", "sequential_offsets",
    }
    if timing_mode not in valid_modes:
        raise ValueError("unknown opponent target timing mode")
    batch, queries = labels.shape
    target = torch.zeros(
        batch, total_steps, 2 * queries, device=labels.device, dtype=labels.dtype
    )
    for query in range(queries):
        sequential = timing_mode in {"sequential_centers", "sequential_offsets"}
        window_offset = query * output_window_len if sequential else 0
        within_window = (
            output_window_len // 2
            if timing_mode in {"simultaneous_center", "sequential_centers"}
            else int(target_offset_steps if target_offset_steps is not None else 0)
        )
        if within_window < 0 or within_window >= output_window_len:
            raise ValueError("target offset lies outside its output window")
        time_index = input_steps + window_offset + within_window
        if time_index >= total_steps:
            raise ValueError("target spike lies outside the simulated trial")
        class_index = labels[:, query].long().clamp(0, 1) + 2 * query
        target[torch.arange(batch, device=labels.device), time_index, class_index] = 1.0
    return target


def opponent_target_membrane_loss(
    output_pre_reset: torch.Tensor,
    labels: torch.Tensor,
    *,
    input_steps: int,
    output_window_len: int,
    timing_mode: str,
    target_offset_steps: int | None,
    threshold: float,
    surrogate_beta: float,
) -> torch.Tensor:
    """Balanced class loss on pre-reset voltages at declared target times."""
    batch, _, outputs = output_pre_reset.shape
    queries = labels.shape[1]
    if outputs != 2 * queries:
        raise ValueError("target membrane loss requires two outputs per query")
    per_query = []
    for query in range(queries):
        sequential = timing_mode in {"sequential_centers", "sequential_offsets"}
        window_offset = query * output_window_len if sequential else 0
        within_window = (
            output_window_len // 2
            if timing_mode in {"simultaneous_center", "sequential_centers"}
            else int(target_offset_steps if target_offset_steps is not None else 0)
        )
        time_index = input_steps + window_offset + within_window
        pair = output_pre_reset[:, time_index, 2 * query:2 * query + 2]
        class_targets = torch.stack([1.0 - labels[:, query], labels[:, query]], dim=1)
        logits = surrogate_beta * (pair - threshold)
        per_query.append(
            F.binary_cross_entropy_with_logits(logits, class_targets, reduction="none")
            .mean(dim=1)
        )
    return window_class_balanced_mean(torch.stack(per_query, dim=1), labels)


def filtered_opponent_spike_train_loss(
    output_spikes: torch.Tensor,
    target_spikes: torch.Tensor,
    labels: torch.Tensor,
    *,
    tau_steps: float,
) -> torch.Tensor:
    """Class/window-balanced van-Rossum-inspired filtered spike MSE."""
    if output_spikes.shape != target_spikes.shape:
        raise ValueError("output and target spike trains must have identical shape")
    if tau_steps <= 0:
        raise ValueError("tau_steps must be positive")
    decay = float(torch.exp(torch.tensor(-1.0 / tau_steps)).item())

    def filter_train(values: torch.Tensor) -> torch.Tensor:
        state = torch.zeros_like(values[:, 0])
        filtered = []
        for t in range(values.shape[1]):
            state = decay * state + values[:, t]
            filtered.append(state)
        return torch.stack(filtered, dim=1)

    filtered_output = filter_train(output_spikes)
    filtered_target = filter_train(target_spikes)
    batch, steps, outputs = output_spikes.shape
    queries = labels.shape[1]
    if outputs != 2 * queries:
        raise ValueError("opponent spike loss requires two outputs per query")
    per_query = (
        (filtered_output - filtered_target).pow(2)
        .reshape(batch, steps, queries, 2)
        .mean(dim=(1, 3))
    )
    return window_class_balanced_mean(per_query, labels)


# ---------------------------------------------------------------------------

def build_optimizer(model: SNNModel, cfg) -> torch.optim.Optimizer:
    """
    Build Adam optimizer with separate LR groups for:
      - SNN weights  (lr_w)
      - SNN delays   (lr_d)
      - Readout      (lr_readout)
    """
    param_groups = []

    w_params = model.weight_params()
    d_params  = model.delay_params()
    r_params  = model.readout_params()

    if w_params:
        param_groups.append({"params": w_params, "lr": cfg["lr_w"]})
    if d_params:
        param_groups.append({"params": d_params, "lr": cfg["lr_d"]})
    if r_params:
        param_groups.append({"params": r_params, "lr": cfg["lr_readout"]})

    return torch.optim.Adam(param_groups)


# ---------------------------------------------------------------------------

class Trainer:
    """
    Encapsulates one training run.

    Parameters
    ----------
    model     : SNNModel
    slots     : list of SlotBoundaries (defines trial structure)
    cfg       : dict-like config
    run_dir   : directory for checkpoints / logs
    device    : torch device string
    """

    def __init__(
        self,
        model: SNNModel,
        slots: List[SlotBoundaries],
        cfg: Dict[str, Any],
        run_dir: str,
        device: str = "cpu",
    ):
        self.model   = model.to(device)
        self.slots   = slots
        self.cfg     = cfg
        self.run_dir = run_dir
        self.device  = device

        os.makedirs(run_dir, exist_ok=True)

        self.optimizer  = build_optimizer(model, cfg)
        self.best_val   = 0.0
        self.log_rows: List[Dict] = []

    # ------------------------------------------------------------------
    def _forward_batch(self, A, B, op_ids, labels):
        """Run one forward pass; return loss and accuracy."""
        A       = A.to(self.device)
        B       = B.to(self.device)
        op_ids  = op_ids.to(self.device)
        labels  = labels.to(self.device)        # [B, K]

        n_ops  = max(self.model.n_input - 2, 0)
        spike_input = encode_trial(
            A, B, op_ids, self.slots,
            n_input=self.model.n_input,
            r_on=self.cfg["r_on"],
            r_off=self.cfg["r_off"],
            dt=self.cfg["dt"],
            device=self.device,
        )

        logits, info = self.model(spike_input, self.slots)  # [B, K]

        # Flatten K queries into batch dimension for loss
        loss_reduction = self.cfg.get("loss_reduction", "pooled_bce")
        if loss_reduction == "window_class_balanced":
            spike_loss = window_class_balanced_bce(logits, labels)
        elif loss_reduction == "pooled_bce":
            spike_loss = F.binary_cross_entropy_with_logits(
                logits.reshape(-1), labels.reshape(-1)
            )
        else:
            raise ValueError(f"Unknown simultaneous loss_reduction: {loss_reduction}")
        warmup_epochs = int(self.cfg.get("output_membrane_warmup_epochs", 0))
        class_logits = info.get("output_membrane_class_logits")
        if warmup_epochs > 0 and class_logits is not None:
            class_targets = torch.stack([1.0 - labels, labels], dim=-1)
            if loss_reduction == "window_class_balanced":
                membrane_per_sample = F.binary_cross_entropy_with_logits(
                    class_logits, class_targets, reduction="none"
                ).mean(dim=-1)
                membrane_loss = window_class_balanced_mean(
                    membrane_per_sample, labels
                )
            else:
                membrane_loss = F.binary_cross_entropy_with_logits(
                    class_logits, class_targets
                )
            if self.current_epoch <= warmup_epochs:
                loss = membrane_loss
            else:
                loss = spike_loss + float(self.cfg.get("output_membrane_aux_weight", 0.2)) * membrane_loss
        else:
            loss = spike_loss
            membrane_loss = None

        # Accuracy is expected to remain flat until output neurons cross their
        # firing threshold. Log the curriculum objectives separately so this
        # interval cannot be mistaken for a frozen optimizer.
        info["spike_count_loss"] = spike_loss.detach()
        if membrane_loss is not None:
            info["output_membrane_loss"] = membrane_loss.detach()

        # Optional penalties
        if self.cfg.get("spike_penalty", 0.0) > 0:
            spk_mean = info["total_hidden_spikes"].mean()
            loss = loss + self.cfg["spike_penalty"] * spk_mean

        if self.cfg.get("delay_penalty", 0.0) > 0:
            loss = loss + self.cfg["delay_penalty"] * self.model.delay_regularization()

        # Accuracy
        preds  = (logits.detach() > 0).float()   # [B, K]
        acc    = (preds == labels.detach()).float().mean().item()

        return loss, acc, info

    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0

        for A, B, op_ids, labels in loader:
            # Promote Step-1 tensors [B] → [B, 1]
            if labels.dim() == 1:
                A, B, op_ids, labels = (
                    A.unsqueeze(1), B.unsqueeze(1),
                    op_ids.unsqueeze(1), labels.unsqueeze(1),
                )

            loss, acc, _ = self._forward_batch(A, B, op_ids, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.get("grad_clip", 1.0)
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_acc  += acc
            n += 1

        return {"loss": total_loss / n, "acc": total_acc / n}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_acc, total_spk, n = 0.0, 0.0, 0.0, 0

        for A, B, op_ids, labels in loader:
            if labels.dim() == 1:
                A, B, op_ids, labels = (
                    A.unsqueeze(1), B.unsqueeze(1),
                    op_ids.unsqueeze(1), labels.unsqueeze(1),
                )

            loss, acc, info = self._forward_batch(A, B, op_ids, labels)
            total_loss += loss.item()
            total_acc  += acc
            total_spk  += info["total_hidden_spikes"].mean().item()
            n += 1

        return {
            "loss": total_loss / n,
            "acc":  total_acc  / n,
            "mean_hidden_spikes": total_spk / n,
        }

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int,
        verbose:      bool = True,
    ) -> List[Dict]:
        """Train for `epochs` epochs; return per-epoch log."""
        for epoch in range(1, epochs + 1):
            t0   = time.time()
            tr   = self.train_epoch(train_loader)
            val  = self.eval_epoch(val_loader)
            dt   = time.time() - t0

            row = {
                "epoch":      epoch,
                "train_loss": tr["loss"],
                "train_acc":  tr["acc"],
                "val_loss":   val["loss"],
                "val_acc":    val["acc"],
                "mean_hidden_spikes": val["mean_hidden_spikes"],
                "time_s":     dt,
            }
            self.log_rows.append(row)

            # Checkpoint
            if val["acc"] > self.best_val:
                self.best_val = val["acc"]
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.run_dir, "best_model.pt"),
                )

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"[{epoch:4d}/{epochs}] "
                    f"train acc={tr['acc']:.4f}  "
                    f"val acc={val['acc']:.4f}  "
                    f"spk={val['mean_hidden_spikes']:.1f}  "
                    f"({dt:.1f}s)"
                )

        self._save_log()
        # Calibration protocols may deliberately select the final state (for
        # example after a membrane-to-spike curriculum) rather than the model
        # with the highest pooled validation accuracy.
        torch.save(
            self.model.state_dict(),
            os.path.join(self.run_dir, "last_model.pt"),
        )
        return self.log_rows

    # ------------------------------------------------------------------
    def _save_log(self):
        if not self.log_rows:
            return
        path = os.path.join(self.run_dir, "train_log.csv")
        keys = list(self.log_rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.log_rows)

    def save_config(self, cfg: Dict):
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)


# ---------------------------------------------------------------------------

class SimultaneousTrainer:
    """
    Training engine for SNNSimultaneousModel (true temporal multiplexing).

    Differences from Trainer:
      - Uses encode_simultaneous_trial() instead of encode_trial()
      - Calls model(spike_input) without slots (single readout window)
      - Model produces [B, K] logits from shared hidden activity
    """

    def __init__(
        self,
        model: SNNSimultaneousModel,
        cfg: Dict[str, Any],
        run_dir: str,
        device: str = "cpu",
        encode_fn=None,
    ):
        self.model   = model.to(device)
        self.cfg     = cfg
        self.run_dir = run_dir
        self.device  = device
        # Plan C default: simultaneous 2K-channel encoding.
        # Plan D override: pass encode_sequential_trial.
        self.encode_fn = encode_fn if encode_fn is not None else encode_simultaneous_trial

        os.makedirs(run_dir, exist_ok=True)

        # Reuse the same optimizer builder (param group API is identical)
        self.optimizer  = build_optimizer(model, cfg)
        self.best_val   = 0.0
        self.log_rows: List[Dict] = []
        self.current_epoch = 0
        self.initial_delays = {
            name: value.detach().cpu().clone()
            for name, value in model.get_delays().items()
        }

    # ------------------------------------------------------------------
    def _forward_batch(self, A, B, op_ids, labels):
        A      = A.to(self.device)
        B      = B.to(self.device)
        labels = labels.to(self.device)   # [B, K]

        spike_input = self.encode_fn(
            A, B,
            win_len=self.cfg["win_len"],
            read_len=self.cfg["read_len"],
            r_on=self.cfg["r_on"],
            r_off=self.cfg["r_off"],
            dt=self.cfg["dt"],
            device=self.device,
            op_ids=op_ids.to(self.device),
            n_ops=self.cfg.get("n_ops", 0),
        )

        target_mode = self.cfg.get("opponent_target_timing_mode")
        logits, info = self.model(
            spike_input,
            return_output_spike_train=target_mode is not None,
        )   # [B, K]

        spike_loss = F.binary_cross_entropy_with_logits(
            logits.reshape(-1), labels.reshape(-1)
        )
        timing_loss = None
        if target_mode is not None:
            if not getattr(self.model, "use_output_spikes", False):
                raise ValueError("opponent target spike training requires spiking outputs")
            target_spikes = opponent_target_spike_train(
                labels,
                total_steps=int(self.model.T),
                input_steps=int(self.cfg["win_len"]),
                output_window_len=int(self.cfg["output_window_len"]),
                timing_mode=target_mode,
                target_offset_steps=self.cfg.get("output_target_offset_steps"),
            )
            timing_loss = filtered_opponent_spike_train_loss(
                info["output_spike_train"], target_spikes, labels,
                tau_steps=float(self.cfg.get("target_filter_tau_steps", 5.0)),
            )

        warmup_epochs = int(self.cfg.get("output_membrane_warmup_epochs", 0))
        class_logits = info.get("output_membrane_class_logits")
        if target_mode is not None:
            membrane_loss = opponent_target_membrane_loss(
                info["output_pre_reset_train"], labels,
                input_steps=int(self.cfg["win_len"]),
                output_window_len=int(self.cfg["output_window_len"]),
                timing_mode=target_mode,
                target_offset_steps=self.cfg.get("output_target_offset_steps"),
                threshold=float(self.model.lif_o.v_threshold),
                surrogate_beta=float(self.model.lif_o.surrogate_beta),
            )
        if warmup_epochs > 0 and class_logits is not None:
            if target_mode is None:
                class_targets = torch.stack([1.0 - labels, labels], dim=-1)
                membrane_loss = F.binary_cross_entropy_with_logits(class_logits, class_targets)
            if self.current_epoch <= warmup_epochs:
                loss = membrane_loss
            else:
                loss = spike_loss + float(self.cfg.get("output_membrane_aux_weight", 0.2)) * membrane_loss
        else:
            loss = spike_loss

        if timing_loss is not None and self.current_epoch > warmup_epochs:
            loss = loss + float(self.cfg.get("target_spike_loss_weight", 1.0)) * timing_loss

        # Keep both calibration objectives visible in the epoch log. During
        # membrane warm-up, count accuracy is intentionally still discrete.
        info["spike_count_loss"] = spike_loss.detach()
        if timing_loss is not None:
            info["target_spike_train_loss"] = timing_loss.detach()
        if warmup_epochs > 0 and class_logits is not None:
            info["output_membrane_loss"] = membrane_loss.detach()

        if self.cfg.get("spike_penalty", 0.0) > 0:
            spk_mean = info["total_hidden_spikes"].mean()
            loss = loss + self.cfg["spike_penalty"] * spk_mean

        # Homeostatic firing-rate regulariser: pull EACH hidden neuron's firing rate
        # toward a small target rate. Revives dead neurons (fixes P1/P2 at a high,
        # sparse threshold) AND caps over-firing -> "sparse AND trainable".
        if self.cfg.get("homeo_lambda", 0.0) > 0 and "hidden_rate" in info:
            target = self.cfg.get("homeo_target", 0.02)
            loss = loss + self.cfg["homeo_lambda"] * ((info["hidden_rate"] - target) ** 2).mean()

        if self.cfg.get("delay_penalty", 0.0) > 0:
            loss = loss + self.cfg["delay_penalty"] * self.model.delay_regularization()

        preds = (logits.detach() > 0).float()
        acc   = (preds == labels.detach()).float().mean().item()

        return loss, acc, info

    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        delay_grad_total, weight_grad_total = 0.0, 0.0
        output_weight_grad_total = 0.0
        total_spike_loss, total_membrane_loss, membrane_batches = 0.0, 0.0, 0
        total_target_timing_loss, timing_batches = 0.0, 0

        for A, B, op_ids, labels in loader:
            if labels.dim() == 1:
                A, B, op_ids, labels = (
                    A.unsqueeze(1), B.unsqueeze(1),
                    op_ids.unsqueeze(1), labels.unsqueeze(1),
                )

            loss, acc, info = self._forward_batch(A, B, op_ids, labels)

            self.optimizer.zero_grad()
            loss.backward()
            delay_grad = sum(
                float(p.grad.detach().pow(2).sum().item())
                for p in self.model.delay_params() if p.grad is not None
            ) ** 0.5
            weight_grad = sum(
                float(p.grad.detach().pow(2).sum().item())
                for p in self.model.weight_params() if p.grad is not None
            ) ** 0.5
            output_weight_grad = (
                float(self.model.syn_ho.weight.grad.detach().norm().item())
                if getattr(self.model, "use_output_spikes", False)
                and self.model.syn_ho.weight.grad is not None else 0.0
            )
            schedule = self.cfg.get("optimization_schedule", "joint")
            warmup_epochs = int(self.cfg.get("weight_warmup_epochs", 0))
            if schedule == "weights_warmup_then_joint" and self.current_epoch <= warmup_epochs:
                for parameter in self.model.delay_params():
                    parameter.grad = None
            elif schedule == "alternating_epochs":
                if self.current_epoch % 2 == 1:
                    for parameter in self.model.delay_params():
                        parameter.grad = None
                else:
                    for parameter in self.model.weight_params() + self.model.readout_params():
                        parameter.grad = None
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.get("grad_clip", 1.0)
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_acc  += acc
            total_spike_loss += float(info["spike_count_loss"].item())
            if "target_spike_train_loss" in info:
                total_target_timing_loss += float(info["target_spike_train_loss"].item())
                timing_batches += 1
            if "output_membrane_loss" in info:
                total_membrane_loss += float(info["output_membrane_loss"].item())
                membrane_batches += 1
            delay_grad_total += delay_grad
            weight_grad_total += weight_grad
            output_weight_grad_total += output_weight_grad
            n += 1

        return {
            "loss": total_loss / n, "acc": total_acc / n,
            "spike_count_loss": total_spike_loss / n,
            "target_spike_train_loss": (total_target_timing_loss / timing_batches
                                         if timing_batches else float("nan")),
            "output_membrane_loss": (total_membrane_loss / membrane_batches
                                      if membrane_batches else float("nan")),
            "delay_grad_norm": delay_grad_total / n,
            "weight_grad_norm": weight_grad_total / n,
            "output_weight_grad_norm": output_weight_grad_total / n,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_acc, total_spk, n = 0.0, 0.0, 0.0, 0
        total_spike_loss, total_membrane_loss, membrane_batches = 0.0, 0.0, 0
        total_target_timing_loss, timing_batches = 0.0, 0

        for A, B, op_ids, labels in loader:
            if labels.dim() == 1:
                A, B, op_ids, labels = (
                    A.unsqueeze(1), B.unsqueeze(1),
                    op_ids.unsqueeze(1), labels.unsqueeze(1),
                )

            loss, acc, info = self._forward_batch(A, B, op_ids, labels)
            total_loss += loss.item()
            total_acc  += acc
            total_spk  += info["total_hidden_spikes"].mean().item()
            total_spike_loss += float(info["spike_count_loss"].item())
            if "target_spike_train_loss" in info:
                total_target_timing_loss += float(info["target_spike_train_loss"].item())
                timing_batches += 1
            if "output_membrane_loss" in info:
                total_membrane_loss += float(info["output_membrane_loss"].item())
                membrane_batches += 1
            n += 1

        return {
            "loss": total_loss / n,
            "acc":  total_acc  / n,
            "mean_hidden_spikes": total_spk / n,
            "spike_count_loss": total_spike_loss / n,
            "target_spike_train_loss": (total_target_timing_loss / timing_batches
                                         if timing_batches else float("nan")),
            "output_membrane_loss": (total_membrane_loss / membrane_batches
                                      if membrane_batches else float("nan")),
        }

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int,
        verbose:      bool = True,
    ) -> List[Dict]:
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            t0  = time.time()
            tr  = self.train_epoch(train_loader)
            val = self.eval_epoch(val_loader)
            dt  = time.time() - t0

            row = {
                "epoch":      epoch,
                "train_loss": tr["loss"],
                "train_acc":  tr["acc"],
                "val_loss":   val["loss"],
                "val_acc":    val["acc"],
                "train_spike_count_loss": tr["spike_count_loss"],
                "val_spike_count_loss": val["spike_count_loss"],
                "train_target_spike_train_loss": tr["target_spike_train_loss"],
                "val_target_spike_train_loss": val["target_spike_train_loss"],
                "train_output_membrane_loss": tr["output_membrane_loss"],
                "val_output_membrane_loss": val["output_membrane_loss"],
                "mean_hidden_spikes": val["mean_hidden_spikes"],
                "time_s":     dt,
                "delay_grad_norm": tr.get("delay_grad_norm", 0.0),
                "weight_grad_norm": tr.get("weight_grad_norm", 0.0),
                "output_weight_grad_norm": tr.get("output_weight_grad_norm", 0.0),
            }
            current_delays = self.model.get_delays()
            movements = []
            saturation = []
            for name, delay in current_delays.items():
                initial = self.initial_delays[name].to(delay.device)
                movements.append((delay.detach() - initial).abs().mean())
                saturation.append(
                    ((delay.detach() <= 1e-3) | (delay.detach() >= self.model.d_max - 1e-3))
                    .float().mean()
                )
            row["mean_abs_delay_movement"] = float(torch.stack(movements).mean().item()) if movements else 0.0
            row["delay_saturation_fraction"] = float(torch.stack(saturation).mean().item()) if saturation else 0.0
            self.log_rows.append(row)

            if val["acc"] > self.best_val:
                self.best_val = val["acc"]
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.run_dir, "best_model.pt"),
                )

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"[{epoch:4d}/{epochs}] "
                    f"train acc={tr['acc']:.4f}  "
                    f"val acc={val['acc']:.4f}  "
                    f"spk={val['mean_hidden_spikes']:.1f}  "
                    f"({dt:.1f}s)"
                )

        self._save_log()
        torch.save(
            self.model.state_dict(),
            os.path.join(self.run_dir, "last_model.pt"),
        )
        return self.log_rows

    # ------------------------------------------------------------------
    def _save_log(self):
        if not self.log_rows:
            return
        path = os.path.join(self.run_dir, "train_log.csv")
        keys = list(self.log_rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.log_rows)

    def save_config(self, cfg: Dict):
        path = os.path.join(self.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)
