
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# ---------------------------------------------------------------------
# Configuration (optional helper)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class BOConfig:
    n_candidates: int = 5000
    kappa: float = 2.0
    random_state: int = 42
    length_scale: float = 0.2
    kernel_amplitude: float = 1.0


# ---------------------------------------------------------------------
# Data loading + validation
# ---------------------------------------------------------------------

def load_xy(func_id: int, data_raw_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (X, y) for a function from:
      data_raw_dir/function_{func_id}/initial_inputs.npy
      data_raw_dir/function_{func_id}/initial_outputs.npy

    Validates:
      - func_id in [1, 8]
      - X is 2D, y is 1D
      - row counts match
      - X in [0, 1)
    """
    if not (1 <= func_id <= 8):
        raise ValueError(f"func_id must be between 1 and 8, got {func_id}.")

    func_dir = data_raw_dir / f"function_{func_id}"
    x_path = func_dir / "initial_inputs.npy"
    y_path = func_dir / "initial_outputs.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)

    # Ensure y is shape (n,)
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"Function {func_id}: X must be 2D (n, d). Got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"Function {func_id}: y must be 1D (n,). Got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Function {func_id}: X and y row counts must match. "
            f"Got X rows={X.shape[0]}, y rows={y.shape[0]}."
        )

    x_min = float(X.min())
    x_max = float(X.max())
    if x_min < 0.0 or x_max >= 1.0:
        raise ValueError(
            f"Function {func_id}: X values must be in [0, 1). Got min={x_min}, max={x_max}."
        )

    return X, y


# ---------------------------------------------------------------------
# Surrogate model (Gaussian Process)
# ---------------------------------------------------------------------
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def fit_gp(
    X,
    y,
    *,
    length_scale: float = 0.2,
    kernel_amplitude: float = 1.0,
    random_state: int = 42,
) -> GaussianProcessRegressor:
    # Add bounds to make hyperparameter search well-behaved
    kernel = C(kernel_amplitude, (1e-3, 1e3)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=1e-6,                 # small noise for numerical stability
        n_restarts_optimizer=2,     # optional: more stable fits
        random_state=random_state,
    )
    gp.fit(X, y)
    return gp



#-------------------------------------------------------------------
# OLD
#-------------------------------------------------------------------
def fit_gp_old(
    X: np.ndarray,
    y: np.ndarray,
    *,
    length_scale: float = 0.2,
    kernel_amplitude: float = 1.0,
    random_state: int = 42,
) -> GaussianProcessRegressor:
    """
    Fit a Gaussian Process regressor as a surrogate model.

    Notes:
      - normalize_y=True improves stability (especially if y scale differs by function)
      - kernel = amplitude * RBF(length_scale)
    """
    kernel = C(kernel_amplitude) * RBF(length_scale=length_scale)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=random_state,
    )
    gp.fit(X, y)
    return gp


# -----------------------------
# Acquisition function (UCB)
# -----------------------------
def acquisition_ucb(mean: np.ndarray, std: np.ndarray, *, kappa: float = 2.0) -> np.ndarray:
    """
    Upper Confidence Bound acquisition score.

    score = mean + kappa * std

    Higher kappa -> more exploration.
    """
    mean = np.asarray(mean).reshape(-1)
    std = np.asarray(std).reshape(-1)
    return mean + kappa * std


# ---------------------------------------------------------------------------------------------
# Candidate generation (random uniform)
# ---------------------------------------------------------------------------------------------
def generate_random_candidates(
    n_candidates: int,
    dim: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate random candidate points uniformly in [0, 1)^dim.

    Parameters
    ----------
    n_candidates : int
        Number of candidate points to generate.
    dim : int
        Dimensionality of the search space.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_candidates, dim) with values in [0, 1).
    """
    rng = np.random.default_rng(seed)

    Xc = rng.uniform(0.0, 1.0, size=(n_candidates, dim))

    # Ensure strictly < 1.0 (portal constraint)
    Xc = np.minimum(Xc, np.nextafter(1.0, 0.0))

    return Xc


# -----------------------------
# Selecting the next point (BO step)
# -----------------------------
def suggest_next_x_ucb(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_candidates: int = 5000,
    kappa: float = 2.0,
    seed: int = 42,
    length_scale: float = 0.2,
    kernel_amplitude: float = 1.0,
) -> np.ndarray:
    """
    Suggest ONE next point x_next using:
      - GP surrogate
      - Random candidate pool
      - UCB acquisition

    Returns:
      x_next: shape (dim,)
    """
    dim = X.shape[1]

    gp = fit_gp(
        X,
        y,
        length_scale=length_scale,
        kernel_amplitude=kernel_amplitude,
        random_state=seed,
    )

    X_candidates = generate_random_candidates(
        n_candidates=n_candidates,
        dim=dim,
        seed=seed,
    )

    mean, std = gp.predict(X_candidates, return_std=True)
    scores = acquisition_ucb(mean, std, kappa=kappa)

    best_idx = int(np.argmax(scores))
    x_next = X_candidates[best_idx]
    return x_next


# -----------------------------
# Portal formatting + validation
# -----------------------------
_PORTAL_PATTERN = re.compile(r"^(0\.\d{6})(-(0\.\d{6}))*$")


def format_submission(x: np.ndarray) -> str:
    """
    Format x into portal string: "0.123456-0.654321-..."
    Enforces:
      - exactly 6 decimals
      - values clipped to [0, 0.999999]
    """
    x = np.asarray(x).reshape(-1)

    # Clip to valid portal range: [0, 0.999999]
    x = np.clip(x, 0.0, 0.999999)

    parts = [f"{v:.6f}" for v in x]
    return "-".join(parts)


def validate_submission_string(s: str, dim: int) -> Tuple[bool, str]:
    """
    Validate portal submission string.
    Rules:
      - hyphen-separated
      - no spaces
      - each value starts with 0.
      - exactly 6 decimals
      - correct number of dimensions
    """
    if " " in s:
        return False, "Submission contains spaces. Remove spaces around hyphens."

    if not _PORTAL_PATTERN.match(s):
        return False, "Format must be like 0.123456-0.654321 with exactly 6 decimals each."

    parts = s.split("-")
    if len(parts) != dim:
        return False, f"Expected {dim} values but got {len(parts)}."

    # Range check
    vals = [float(p) for p in parts]
    if any(v < 0.0 or v >= 1.0 for v in vals):
        return False, "All values must be in [0.000000, 0.999999]."

    return True, "OK"


def suggest_and_format_for_portal(
    func_id: int,
    data_raw_dir: Path,
    *,
    config: BOConfig = BOConfig(),
) -> Tuple[np.ndarray, str]:
    """
    Convenience: load data -> suggest x_next -> format portal string.
    Returns (x_next, submission_string).
    """
    X, y = load_xy(func_id, data_raw_dir)

    x_next = suggest_next_x_ucb(
        X,
        y,
        n_candidates=config.n_candidates,
        kappa=config.kappa,
        seed=config.random_state,
        length_scale=config.length_scale,
        kernel_amplitude=config.kernel_amplitude,
    )

    s = format_submission(x_next)
    ok, msg = validate_submission_string(s, dim=X.shape[1])
    if not ok:
        raise ValueError(f"Generated submission failed validation: {msg}\nGenerated: {s}")

    return x_next, s
