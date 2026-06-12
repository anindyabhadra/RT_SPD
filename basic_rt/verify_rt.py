#!/usr/bin/env python3
"""Command-line verification for the core RT routines."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basic_rt.rt_core import verify_basic_properties


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=8)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    print(json.dumps(verify_basic_properties(args.p, args.seed, args.atol), indent=2))


if __name__ == "__main__":
    main()
