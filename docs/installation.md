# Installation

spintrack installs as a command-line tool with [uv](https://docs.astral.sh/uv/getting-started/installation/), which also fetches Python 3.14 if needed:

```bash
uv tool install --torch-backend auto git+https://github.com/NeLy-EPFL/spintrack
```

- **GPU.** `--torch-backend auto` picks the PyTorch build that matches the machine's GPU driver (CUDA or ROCm), or the CPU build when there is none. PyTorch runs SAM 3, which finds the ball and the animal: 0.3 s per recording on a GPU, about 13 s on a CPU. Its checkpoint (3.4 GB) downloads on the first run that needs it, without a Hugging Face login.
- **Rust.** Until wheels come with the first release, the install builds spintrack's Rust extension, so it needs a [Rust toolchain](https://rustup.rs).
- **Serial streaming.** `--serial` needs the `serial` extra: install `"spintrack[serial] @ git+https://github.com/NeLy-EPFL/spintrack"` instead.
- **Updating.** Run the same command with `--reinstall`; `uv tool upgrade` would not keep the PyTorch build.

## Check the installation

`spintrack doctor` reports what the installation can do without downloading anything: the compiled core, the PyTorch build and the device SAM 3 runs on, whether SAM 3's checkpoint is cached, PyAV's FFmpeg and the serial extra. `--json` prints the same for scripts, and `--check` makes warnings fail too.

## From source

To work on spintrack, clone the repository and run `uv sync`, which builds the extension with maturin; see `CONTRIBUTING.md` in the repository.
