#!/usr/bin/env python3
"""Freeze an honest exploratory alert protocol, then score cached predictions."""

import argparse
from pathlib import Path

from infectionpulse.alert_evaluation import evaluate, freeze_protocol

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "models/respiratory/runs/65cf524d3927a4b8"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    protocol = freeze_protocol(args.run, args.output / "alert_protocol.json")
    print(
        f"Frozen exploratory protocol: {args.output / 'alert_protocol.json'}",
        flush=True,
    )
    if not args.freeze_only:
        result = evaluate(args.run, args.output, protocol)
        print(
            f"Evaluated {len(result['metrics'])} model/policy pairs; report: {args.output / 'alert_report.md'}"
        )


if __name__ == "__main__":
    main()
