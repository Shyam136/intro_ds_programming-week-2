import numpy as np


def ways(n: int) -> int:
    """
    Return the number of unordered factor pairs of n.
    Example: n=12 -> (1,12),(2,6),(3,4) => 3
    For n <= 0, return 0.
    """
    if n <= 0:
        return 0
    c = 0
    r = int(n**0.5)
    for d in range(1, r + 1):
        if n % d == 0:
            c += 1
    return c


def lowest_score(names, scores):
    """
    Return the name with the lowest score.
    Ties: return the alphabetically-first name.
    """
    if len(names) != len(scores):
        raise ValueError("names and scores must have the same length")
    if not names:
        return None

    scores_arr = np.asarray(scores, dtype=float)
    names_arr = np.asarray(names, dtype=object)

    min_val = float(np.min(scores_arr))
    tied = names_arr[scores_arr == min_val].tolist()
    return min(tied)  # alphabetical tie-break


def sort_names(names, scores):
    """
    Return names sorted by score DESC (highest first).
    Ties: alphabetical ASC by name.
    """
    if len(names) != len(scores):
        raise ValueError("names and scores must have the same length")
    if not names:
        return []

    scores_arr = np.asarray(scores, dtype=float)
    names_arr = np.asarray(names, dtype=object)

    # np.lexsort: last key is primary. Use -scores for DESC, names for tie-break ASC.
    order = np.lexsort((names_arr, -scores_arr))
    return names_arr[order].tolist()

assert ways(12) == 3
assert ways(1) == 1    # (1,1)
assert ways(9) == 2    # (1,9),(3,3)

ns = ["Charlie", "Alice", "Bob"]
ss = [70, 85, 85]
assert sort_names(ns, ss) == ["Alice", "Bob", "Charlie"]  # 85s first, A then B