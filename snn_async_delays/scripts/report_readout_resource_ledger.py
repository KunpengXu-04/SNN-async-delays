"""Generate a static resource table for candidate readout interfaces.

No training or evaluation is performed.  The report makes resource changes
visible before a pilot is launched.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from snn.model import SNNSimultaneousModel
from utils.resource_ledger import static_resource_ledger


BASE = Path(__file__).resolve().parents[1]


def build(K: int, h: int, sub_win: int, read_len: int, decoder: str, mode: str):
    spiking = decoder == "spiking_output"
    model = SNNSimultaneousModel(
        n_queries=K,
        n_hidden=h,
        win_len=K * sub_win,
        read_len=read_len,
        d_max=K * sub_win,
        n_input_channels=2,
        train_mode="weights_and_delays",
        readout_type="linear" if spiking else decoder,
        use_output_spikes=spiking,
        n_output_neurons=K if spiking else None,
        observation_mode=mode,
    )
    row = {"K": K, "hidden_size": h, "decoder": decoder, **static_resource_ledger(model)}
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--hidden", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--sub-win", type=int, default=10)
    parser.add_argument("--read-len", type=int, default=10)
    parser.add_argument(
        "--output", default="docs/generated/readout_resource_ledger_v1.csv"
    )
    args = parser.parse_args()

    rows = []
    for K in args.K:
        for h in args.hidden:
            for decoder in ("linear", "mlp"):
                for mode in ("late_window", "all_time", "time_binned"):
                    rows.append(build(K, h, args.sub_win, args.read_len, decoder, mode))
            for mode in ("late_window", "all_time"):
                rows.append(build(K, h, args.sub_win, args.read_len, "spiking_output", mode))

    output = Path(args.output)
    if not output.is_absolute():
        output = BASE / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = {
        "output": str(output),
        "n_rows": len(rows),
        "scope": "static architecture/simulator ledger; no accuracy or measured events",
        "schema_version": "resource_ledger_v1",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
