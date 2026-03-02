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

from snn.model import SNNModel, SlotBoundaries
from data.encoding import encode_trial


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
        loss = F.binary_cross_entropy_with_logits(
            logits.reshape(-1), labels.reshape(-1)
        )

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
