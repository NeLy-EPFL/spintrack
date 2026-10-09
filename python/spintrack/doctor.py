"""What `spintrack doctor` reports: what this installation can do.

Each check is a section of findings with a status, as in `octacam doctor`: `ok`,
`info` (a fact, nothing to fix), `warn` (works, but worse than it could) or `error`
(a command will fail). Nothing here downloads a file or opens a camera.
"""

import importlib.metadata
import importlib.util
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

from spintrack import __version__

# The driver's version file: an NVIDIA GPU is in this machine, whatever PyTorch sees.
NVIDIA_DRIVER = Path("/proc/driver/nvidia/version")


@dataclass
class Section:
    """A heading of the report and its findings, (status, text) pairs."""

    title: str
    findings: list[tuple[str, str]] = field(default_factory=list)

    def add(self, status: str, text: str) -> None:
        """Add a finding: `status` is `ok`, `info`, `warn` or `error`."""
        self.findings.append((status, text))


def report() -> list[Section]:
    """Every section of the report, in the order it is shown."""
    return [_system(), _torch(), _sam3(), _video(), _streaming()]


def _system() -> Section:
    from spintrack import _core

    section = Section("System")
    section.add("info", f"spintrack {__version__}")
    section.add(
        "info", f"Python {platform.python_version()} on {platform.platform(terse=True)}"
    )
    section.add("ok", f"compiled core: {Path(_core.__file__).name}")
    return section


def _torch() -> Section:
    section = Section("PyTorch (SAM 3 finds the ball and the animal)")
    try:
        import torch
    except ImportError as exc:
        section.add("error", f"PyTorch does not import: {exc}")
        return section
    cuda = torch.version.cuda
    build = f"CUDA {cuda}" if cuda else "CPU build"
    section.add("info", f"PyTorch {torch.__version__} ({build})")
    reinstall = "reinstall with uv tool install --reinstall --torch-backend auto ..."
    if torch.cuda.is_available():
        section.add("ok", f"SAM 3 runs on the GPU: {torch.cuda.get_device_name(0)}")
    elif os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        section.add("info", "CUDA_VISIBLE_DEVICES hides the GPU: SAM 3 runs on the CPU")
    elif NVIDIA_DRIVER.exists() and not cuda:
        section.add(
            "warn", f"this machine has an NVIDIA GPU, PyTorch no CUDA: {reinstall}"
        )
    elif NVIDIA_DRIVER.exists():
        section.add(
            "warn",
            f"PyTorch's CUDA {cuda} does not run on this machine's driver: {reinstall}",
        )
    else:
        section.add(
            "info",
            "no GPU: SAM 3 runs on the CPU, about 13 s per recording instead of 0.3 s",
        )
    return section


def _sam3() -> Section:
    # transformers, which runs SAM 3, depends on huggingface_hub.
    from huggingface_hub import try_to_load_from_cache

    from spintrack.segment import SOURCES

    section = Section("SAM 3 checkpoint")
    for repo, revision in SOURCES:
        path = try_to_load_from_cache(repo, "model.safetensors", revision=revision)
        if isinstance(path, str):
            section.add("ok", f"cached: {repo} at {revision[:7]}")
            return section
    section.add(
        "info",
        "not cached: the first run that needs it downloads 3.4 GB from Hugging Face, "
        "without a login",
    )
    return section


def _video() -> Section:
    section = Section("Video")
    try:
        import av
    except ImportError as exc:
        section.add("error", f"PyAV does not import: {exc}")
        return section
    section.add("ok", f"PyAV {av.__version__}, FFmpeg {av.ffmpeg_version_info}")
    missing = [c for c in ("h264", "hevc", "mpeg4") if c not in av.codecs_available]
    if missing:
        section.add("warn", f"FFmpeg lacks the codecs {', '.join(missing)}")
    return section


def _streaming() -> Section:
    section = Section("Streaming")
    section.add("ok", "UDP and TCP (--udp, --tcp)")
    if importlib.util.find_spec("serial") is None:
        section.add(
            "info",
            "serial: not installed; --serial needs the serial extra "
            "(spintrack[serial])",
        )
    else:
        version = importlib.metadata.version("pyserial")
        section.add("ok", f"serial: pyserial {version} (--serial)")
    return section
