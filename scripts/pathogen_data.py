"""Refresh observed influenza and RSV context from official public RKI files."""

import argparse
import json

from infectionpulse.pathogen_data import refresh_pathogen_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["refresh"])
    parser.add_argument("--data-dir", default="data/pathogens")
    parser.add_argument("--output", default="data/service/pathogen_observations.json")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--as-of")
    args = parser.parse_args()
    result = refresh_pathogen_context(
        args.data_dir, args.output, args.offline, args.as_of
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "observations": len(result["observations"]),
                "missing": len(result["missing"]),
                "sources": result["sources"],
                "errors": result["errors"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
