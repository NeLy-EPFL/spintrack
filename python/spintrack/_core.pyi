import numpy as np
from numpy.typing import NDArray

W_MIN: float
MAX_ITER: int
TOL: float

def core_version() -> str: ...

class SolveResult:
    w: list[float]
    r: list[list[float]]
    cost: float
    inlier_frac: float
    overlap: float
    iters: int
    converged: bool
    last_step: float

class Engine:
    def __init__(
        self,
        surface: NDArray[np.float32],
        index: NDArray[np.int64],
        window_size: int,
        face: int,
        max_pixels: int | None = None,
    ) -> None: ...
    def solve(
        self,
        obs: NDArray[np.float32],
        r_prev: NDArray[np.float64],
        w0: list[float] | tuple[float, float, float],
        use_prev: bool = False,
        levels: int | None = None,
        max_iter: int = 10,
    ) -> SolveResult: ...
    def global_search(
        self, obs: NDArray[np.float32], min_overlap: float
    ) -> SolveResult: ...
    def update(
        self,
        obs: NDArray[np.float32],
        r: NDArray[np.float64],
        forget_outside: bool = False,
        update_main: bool = True,
        weight: NDArray[np.float32] | None = None,
    ) -> None: ...
    def reset(self) -> None: ...
    def map_mean(self) -> NDArray[np.float32]: ...
    def map_weight(self) -> NDArray[np.float32]: ...
    def render(
        self, r: NDArray[np.float64]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
    def set_map(
        self, mean: NDArray[np.float32], weight: NDArray[np.float32]
    ) -> None: ...
