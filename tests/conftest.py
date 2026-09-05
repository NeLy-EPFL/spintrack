from pathlib import Path

import pytest

# Real lab configs, used only when present on this machine.
LAB_CONFIGS = [
    Path(
        "/home/tlam/backup-20250501/fictrac/data/260128_dna01_g8m/Fly3/005/config.txt"
    ),
    Path("/home/tlam/fictrac-upstream/sample/config.txt"),
]


@pytest.fixture
def lab_configs():
    present = [p for p in LAB_CONFIGS if p.exists()]
    if not present:
        pytest.skip("no local FicTrac configs available")
    return present
