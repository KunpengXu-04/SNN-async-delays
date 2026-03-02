"""
Boolean-operation datasets for Steps 1, 2, and 3.

OPS_LIST (canonical order, used for one-hot op encoding in Step 3):
    AND, OR, XOR, XNOR, NAND, NOR, A_IMP_B, B_IMP_A

Dataset variants
----------------
Step 1  – BooleanDataset      : one (A,B) pair per sample, one op
Step 2  – MultiQueryDataset   : K pairs per sample, same op (same_op=True)
Step 3  – MultiQueryDataset   : K pairs per sample, mixed ops (same_op=False)
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
OPS_LIST: List[str] = ["AND", "OR", "XOR", "XNOR", "NAND", "NOR", "A_IMP_B", "B_IMP_A"]

_OPS = {
    "AND":     lambda a, b: int(bool(a) and bool(b)),
    "OR":      lambda a, b: int(bool(a) or  bool(b)),
    "XOR":     lambda a, b: int(bool(a) != bool(b)),
    "XNOR":    lambda a, b: int(bool(a) == bool(b)),
    "NAND":    lambda a, b: int(not (bool(a) and bool(b))),
    "NOR":     lambda a, b: int(not (bool(a) or  bool(b))),
    "A_IMP_B": lambda a, b: int((not bool(a)) or bool(b)),
    "B_IMP_A": lambda a, b: int(bool(a) or (not bool(b))),
}


def compute_label(op_name: str, A: int, B: int) -> int:
    return _OPS[op_name](A, B)


# ---------------------------------------------------------------------------

class BooleanDataset(Dataset):
    """
    Step 1 dataset: each sample is (A, B, op_id, label).

    For a single op, all 4 input combinations are tiled to reach n_samples.
    For random ops (op_name=None), ops are sampled uniformly.
    """

    def __init__(
        self,
        n_samples: int,
        op_name: Optional[str] = "NAND",
        ops_list: List[str] = OPS_LIST,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        n_ops = len(ops_list)

        A = rng.randint(0, 2, n_samples).astype(np.float32)
        B = rng.randint(0, 2, n_samples).astype(np.float32)

        if op_name is not None:
            op_ids = np.full(n_samples, ops_list.index(op_name), dtype=np.int64)
        else:
            op_ids = rng.randint(0, n_ops, n_samples).astype(np.int64)

        labels = np.array(
            [compute_label(ops_list[op_ids[i]], int(A[i]), int(B[i])) for i in range(n_samples)],
            dtype=np.float32,
        )

        self.A      = torch.from_numpy(A)
        self.B      = torch.from_numpy(B)
        self.op_ids = torch.from_numpy(op_ids)
        self.labels = torch.from_numpy(labels)
        self.ops_list = ops_list

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        return self.A[idx], self.B[idx], self.op_ids[idx], self.labels[idx]


# ---------------------------------------------------------------------------

class MultiQueryDataset(Dataset):
    """
    Step 2 / 3 dataset: each sample contains K queries.

    Returns:
        A       : [K]  float  (0 or 1)
        B       : [K]  float
        op_ids  : [K]  long   (index into ops_list)
        labels  : [K]  float  (0 or 1)
    """

    def __init__(
        self,
        K: int,
        n_samples: int,
        same_op: bool = True,
        op_name: Optional[str] = "NAND",
        ops_list: List[str] = OPS_LIST,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        n_ops = len(ops_list)

        A = rng.randint(0, 2, (n_samples, K)).astype(np.float32)
        B = rng.randint(0, 2, (n_samples, K)).astype(np.float32)

        if same_op:
            assert op_name is not None, "op_name required when same_op=True"
            op_idx  = ops_list.index(op_name)
            op_ids  = np.full((n_samples, K), op_idx, dtype=np.int64)
        else:
            op_ids  = rng.randint(0, n_ops, (n_samples, K)).astype(np.int64)

        labels = np.array(
            [
                [compute_label(ops_list[op_ids[s, k]], int(A[s, k]), int(B[s, k]))
                 for k in range(K)]
                for s in range(n_samples)
            ],
            dtype=np.float32,
        )

        self.A      = torch.from_numpy(A)
        self.B      = torch.from_numpy(B)
        self.op_ids = torch.from_numpy(op_ids)
        self.labels = torch.from_numpy(labels)
        self.K      = K
        self.ops_list = ops_list

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        return self.A[idx], self.B[idx], self.op_ids[idx], self.labels[idx]
