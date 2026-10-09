<h1 align="center">spintrack</h1>

<p align="center">
  <a href="https://github.com/NeLy-EPFL/spintrack/actions/workflows/ci.yml"><img src="https://github.com/NeLy-EPFL/spintrack/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://nely-epfl.github.io/spintrack/"><img src="https://github.com/NeLy-EPFL/spintrack/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <img src="https://img.shields.io/badge/python-3.14-blue" alt="Python 3.14">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
</p>

spintrack tracks the 3D rotation of a trackball (spherical treadmill) under a tethered animal from one camera's video, and integrates the animal's fictive path. It streams FicTrac's records, so closed-loop rigs keep working, writes them to Parquet with named columns, and on synthetic scenes its per-frame error is about a tenth of FicTrac's.

## Installation

```bash
uv tool install --torch-backend auto git+https://github.com/NeLy-EPFL/spintrack
```

`--torch-backend auto` picks the PyTorch build for the machine's GPU; PyTorch runs SAM 3, which finds the ball (its 3.4 GB checkpoint downloads on first use). Until wheels are released, the install builds a Rust extension and needs a [Rust toolchain](https://rustup.rs). Streaming over a serial port needs the `serial` extra; `spintrack doctor` reports what the installation can do.

## Quickstart

```bash
git clone https://github.com/NeLy-EPFL/spintrack && cd spintrack
spintrack run examples/sample/sample.mp4 --set camera.azimuth_deg=180
spintrack gui examples/sample/sample.mp4
```

The run finds the ball, the field of view and the camera's elevation in the video; which side the camera films the fly from (`180`: behind it) is the one thing to say. It prints a link to a live view of the run, and writes `tracks.parquet`, a summary and the config as run into `examples/sample/sample_spintrack/`. `spintrack gui` tracks a clip over and over while you fix what a run got wrong, and saves a config for the rig.

## Documentation

Full documentation: <https://nely-epfl.github.io/spintrack/>

- [Quickstart](https://nely-epfl.github.io/spintrack/stable/quickstart/): the example run, step by step.
- [User guide](https://nely-epfl.github.io/spintrack/stable/guide/): from a recording to a tracked file, and how to tell whether to trust it.
- [Output files](https://nely-epfl.github.io/spintrack/stable/output/): every column, with its unit, frame and sign.
- [Coming from FicTrac](https://nely-epfl.github.io/spintrack/stable/fictrac/): config keys, records, and what differs.
- [Benchmark](https://nely-epfl.github.io/spintrack/stable/benchmark/): accuracy and speed against FicTrac.
- [Python API](https://nely-epfl.github.io/spintrack/stable/api/): `spintrack.track`, and a `Tracker` to feed frame by frame.

Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md).

<!-- --8<-- [start:citing] -->
## Citation

There is no paper on spintrack yet; if you use it, please cite the software ([`CITATION.cff`](https://github.com/NeLy-EPFL/spintrack/blob/main/CITATION.cff) holds the same reference), and FicTrac, whose output it reproduces: Moore, R. J. D., Taylor, G. J., Paulk, A. C., Pearson, T., van Swinderen, B., & Srinivasan, M. V. (2014). FicTrac: a visual method for tracking spherical motion and generating fictive animal paths. *Journal of Neuroscience Methods*, 225, 106–119. [doi:10.1016/j.jneumeth.2014.01.010](https://doi.org/10.1016/j.jneumeth.2014.01.010)

```bibtex
@software{lam_spintrack_2026,
  author = {Lam, Thomas Ka Chung},
  title  = {spintrack: track the rotation of a spherical treadmill from video},
  year   = {2026},
  url    = {https://github.com/NeLy-EPFL/spintrack},
}
```
<!-- --8<-- [end:citing] -->

## Acknowledgments

spintrack is developed in the [Neuroengineering Laboratory (Ramdya lab)](https://www.epfl.ch/labs/ramdya-lab/) at EPFL. It follows [FicTrac](https://github.com/rjdmoore/fictrac), whose record format and path integration it reproduces, and finds the ball with Meta's [SAM 3](https://github.com/facebookresearch/sam3). The full credits, with the methods it reimplements from the literature and the libraries it runs on, are on the [Citing and credits](https://nely-epfl.github.io/spintrack/stable/citing/) page.

<!-- --8<-- [start:ai] -->
## Use of AI

spintrack was developed with extensive help from AI coding assistants, mainly [Claude Code](https://claude.com/claude-code) (Anthropic). Directed by the author, they wrote much of the code, tests, and documentation, including this README, and ran many of the validation and benchmark experiments behind the accuracy figures in the documentation. The author set the goals, made the design decisions, reviewed the changes, and checked the results against data, and is responsible for the software. As with any tool, validate its output on your own recordings before relying on it.
<!-- --8<-- [end:ai] -->

## License

Apache-2.0; see [LICENSE](LICENSE).
