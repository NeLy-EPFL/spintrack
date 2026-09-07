import numpy as np
from numpy.typing import NDArray

def core_version() -> str: ...

class SolveResult:
    w: list[float]
    r: list[list[float]]
    cost: float
    rms: float
    inlier_frac: float
    overlap: float
    iters: int
    converged: bool
    hessian: list[list[float]]
    steps: list[float]

class Engine:
    def __init__(
        self,
        surface: NDArray[np.float32],
        index: NDArray[np.int64],
        window_size: int,
        map_w: int,
        map_h: int,
        levels: int = 3,
        max_pixels: int | None = None,
        projection: str = "equal_area",
    ) -> None: ...
    @property
    def n_valid(self) -> int: ...
    @property
    def levels(self) -> int: ...
    def solve(
        self,
        obs: NDArray[np.float32],
        r_prev: NDArray[np.float64],
        w0: list[float] | tuple[float, float, float],
        use_prev: bool = False,
        levels: int | None = None,
        max_iter: int = 10,
        tol: float = 3e-4,
        huber: float = 1.345,
        tukey: float = 4.685,
        w_min: float = 0.1,
        w_sat: float = 3.0,
        damping: float = 1e-6,
        level_tol_factor: float = 4.0,
        coarse_max_iter: int = 3,
        reweight_iters: int = 1000,
    ) -> SolveResult: ...
    def global_search(
        self,
        obs: NDArray[np.float32],
        n_candidates: int = 2000,
        top_k: int = 5,
        max_iter: int = 10,
        tol: float = 3e-4,
        huber: float = 1.0,
        tukey: float = 3.0,
        w_min: float = 0.1,
        w_sat: float = 3.0,
        min_overlap: float = 0.3,
    ) -> SolveResult: ...
    def cost(
        self,
        obs: NDArray[np.float32],
        r: NDArray[np.float64],
        use_prev: bool = False,
        level: int = 0,
        huber: float = 1.0,
        tukey: float = 3.0,
        w_min: float = 0.5,
        w_sat: float = 3.0,
    ) -> tuple[float, float]: ...
    def update(
        self,
        obs: NDArray[np.float32],
        r: NDArray[np.float64],
        lambda_: float = 1.0,
        w_max: float = 50.0,
        forget_outside: bool = False,
        margin: int = 1,
        update_main: bool = True,
    ) -> None: ...
    def reset(self) -> None: ...
    def map_mean(self) -> NDArray[np.float32]: ...
    def map_weight(self) -> NDArray[np.float32]: ...
    def set_map(
        self, mean: NDArray[np.float32], weight: NDArray[np.float32]
    ) -> None: ...
