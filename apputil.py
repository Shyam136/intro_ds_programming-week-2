from __future__ import annotations

from typing import Iterable, Iterator, Tuple, Dict
import math
import numpy as np


# ---------- Vanilla Python (numbers / loops / logic) ----------

def sum_of_squares(xs: Iterable[float]) -> float:
    """Return the sum of squares of numbers in xs."""
    total = 0.0
    for x in xs:
        total += x * x
    return total


def fibonacci(n: int) -> Iterator[int]:
    """Yield the first n Fibonacci numbers: 0, 1, 1, 2, 3, ..."""
    a, b = 0, 1
    for _ in range(max(0, n)):
        yield a
        a, b = b, a + b


def is_prime(n: int) -> bool:
    """Return True if n is prime (n >= 2)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def moving_average(seq: Iterable[float], k: int) -> list[float]:
    """Simple moving average (window k) over a Python iterable."""
    if k <= 0:
        raise ValueError("k must be positive")
    buf: list[float] = []
    out: list[float] = []
    s = 0.0
    for x in seq:
        buf.append(float(x))
        s += buf[-1]
        if len(buf) > k:
            s -= buf.pop(0)
        if len(buf) == k:
            out.append(s / k)
    return out


# ---------- NumPy (arrays / vectorization / broadcasting) ----------

def to_2d(a: np.ndarray) -> np.ndarray:
    """Ensure array is 2D (n, m). 1D becomes (n, 1)."""
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def zscore(x: np.ndarray, axis: int | None = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Z-score normalize array along axis (mean=0, std=1).
    Uses broadcasting to avoid loops.
    """
    x = np.asarray(x, dtype=float)
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, ddof=0, keepdims=True)
    return (x - mu) / (sigma + eps)  # broadcasting ✓


def minmax_scale(x: np.ndarray, axis: int | None = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Min-max scale array to [0, 1] along axis using broadcasting.
    """
    x = np.asarray(x, dtype=float)
    mn = np.min(x, axis=axis, keepdims=True)
    mx = np.max(x, axis=axis, keepdims=True)
    return (x - mn) / (mx - mn + eps)


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Return a ⋅ b as a Python float."""
    return float(np.dot(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean distance between vectors a and b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def column_stats(a: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Column-wise stats for a 2D array: mean, std, min, max.
    Returns dict of 1D arrays (shape: [n_cols]).
    """
    a = to_2d(np.asarray(a, dtype=float))
    return {
        "mean": np.mean(a, axis=0),
        "std": np.std(a, axis=0, ddof=0),
        "min": np.min(a, axis=0),
        "max": np.max(a, axis=0),
    }


def top_k_indices(x: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the top-k largest values in 1D array x (descending).
    Uses argpartition for O(n) selection + argsort for final order.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if not (1 <= k <= x.size):
        raise ValueError("k must be between 1 and len(x)")
    part = np.argpartition(x, -k)[-k:]
    return part[np.argsort(x[part])[::-1]]


def reshape_matrix(a: Iterable[float], rows: int, cols: int) -> np.ndarray:
    """Reshape a flat iterable to a (rows, cols) matrix."""
    arr = np.asarray(list(a), dtype=float)
    if arr.size != rows * cols:
        raise ValueError("Number of elements does not match rows*cols")
    return arr.reshape(rows, cols)


# ---------- Quick sanity tests (optional local run) ----------

if __name__ == "__main__":
    assert sum_of_squares([1, 2, 3]) == 14
    assert list(fibonacci(6)) == [0, 1, 1, 2, 3, 5]
    assert is_prime(29) and not is_prime(1) and not is_prime(100)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(moving_average(x, 2), [1.5, 2.5, 3.5])

    A = np.array([[1, 2], [3, 4], [5, 6]])
    zs = zscore(A, axis=0)
    assert np.allclose(np.mean(zs, axis=0), [0, 0], atol=1e-8)

    mm = minmax_scale(A, axis=0)
    assert np.allclose(np.min(mm, axis=0), [0, 0]) and np.allclose(
        np.max(mm, axis=0), [1, 1]
    )

    assert math.isclose(dot_product([1, 2], [3, 4]), 11.0)
    assert math.isclose(euclidean_distance([0, 0], [3, 4]), 5.0)

    stats = column_stats(A)
    assert np.allclose(stats["mean"], [3.0, 4.0])

    idx = top_k_indices(np.array([10, 50, 20, 40]), 2)
    assert list(idx) == [1, 3]

    M = reshape_matrix(range(6), 2, 3)
    assert M.shape == (2, 3)

    print("All sanity tests passed.")