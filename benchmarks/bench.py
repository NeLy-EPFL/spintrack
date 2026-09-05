"""Entry point: `uv run --group bench python benchmarks/bench.py <command> ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spintrack_bench.cli import main

if __name__ == "__main__":
    sys.exit(main())
