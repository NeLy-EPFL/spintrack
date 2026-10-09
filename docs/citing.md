# Citing and credits

--8<-- "README.md:citing"

If you publish results obtained with SAM 3, which spintrack uses to find the ball and the animal, acknowledge it as well: its license asks publications to.

## Acknowledgments

spintrack is developed in the [Neuroengineering Laboratory (Ramdya lab)](https://www.epfl.ch/labs/ramdya-lab/) at EPFL. It follows [FicTrac](https://github.com/rjdmoore/fictrac), whose record format and path integration it reproduces, and builds on the following work.

**Methods reimplemented from the literature**

| Work | Where in spintrack |
| --- | --- |
| FicTrac: Moore, R. J. D., Taylor, G. J., Paulk, A. C., Pearson, T., van Swinderen, B., & Srinivasan, M. V. (2014), *Journal of Neuroscience Methods* 225, 106–119, [doi:10.1016/j.jneumeth.2014.01.010](https://doi.org/10.1016/j.jneumeth.2014.01.010) | the 25-field record and its line format, the fictive path's integration scheme (reimplemented in closed form), the surface map's density, and loading FicTrac sphere-map PNGs |
| Baker, S., & Matthews, I. (2004), Lucas-Kanade 20 years on: a unifying framework, *International Journal of Computer Vision* 56, 221–255, [doi:10.1023/B:VISI.0000011205.11775.fd](https://doi.org/10.1023/B:VISI.0000011205.11775.fd) | the Gauss-Newton alignment of each frame against the surface map |
| Huber, P. J. (1964), *The Annals of Mathematical Statistics* 35, 73–101, [doi:10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732); Beaton, A. E., & Tukey, J. W. (1974), *Technometrics* 16, 147–185, [doi:10.1080/00401706.1974.10489171](https://doi.org/10.1080/00401706.1974.10489171) | the robust weights of the per-frame solve and of the rim fit |
| Kåsa, I. (1976), *IEEE Transactions on Instrumentation and Measurement* IM-25, 8–14, [doi:10.1109/TIM.1976.6312298](https://doi.org/10.1109/TIM.1976.6312298) | the algebraic circle fit of the ball's rim |
| Fischler, M. A., & Bolles, R. C. (1981), *Communications of the ACM* 24, 381–395, [doi:10.1145/358669.358692](https://doi.org/10.1145/358669.358692) | the RANSAC start of the rim fit |
| Lepskii, O. V. (1991), *Theory of Probability & Its Applications* 35, 454–466, [doi:10.1137/1135065](https://doi.org/10.1137/1135065) | the ball follower's choice of how many looks to fit (Lepski's rule) |
| Brown, C. (2017), [Bringing pixels front and center in VR video](https://blog.google/products/google-ar-vr/bringing-pixels-front-and-center-vr-video/), Google | the equi-angular cubemap of the surface map |

**Models and libraries**

- [SAM 3](https://github.com/facebookresearch/sam3) (SAM License): Carion, N., Gustafson, L., Hu, Y.-T., Debnath, S., et al. (2025). SAM 3: Segment Anything with Concepts. [arXiv:2511.16719](https://arxiv.org/abs/2511.16719). It finds the ball and the animal, through [Transformers](https://github.com/huggingface/transformers) (Apache-2.0); spintrack downloads [an ungated copy](https://huggingface.co/tkclam/sam3) of Meta's [checkpoint](https://huggingface.co/facebook/sam3), byte-identical to it.
- [PyO3](https://pyo3.rs), [rust-numpy](https://github.com/PyO3/rust-numpy) and [maturin](https://www.maturin.rs), which bind the Rust core to Python.
- [NumPy](https://numpy.org), [OpenCV](https://opencv.org), [PyAV](https://github.com/PyAV-Org/PyAV) (FFmpeg), [Polars](https://pola.rs), [pydantic](https://docs.pydantic.dev), [PyTorch](https://pytorch.org), [Pillow](https://python-pillow.github.io), [FastAPI](https://fastapi.tiangolo.com), [Typer](https://typer.tiangolo.com) and [pySerial](https://github.com/pyserial/pyserial), on which everything runs.

spintrack vendors no third-party files: its web page's scripts and styles are its own. The example clips in `examples/` were recorded in the Ramdya lab.

--8<-- "README.md:ai"
