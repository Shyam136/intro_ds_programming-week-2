import numpy as np


def ways(n: int) -> int:
    """
    Return the number of unordered factor pairs of n.
    Examples:
      12 -> (1,12),(2,6),(3,4) => 3
      20 -> (1,20),(2,10),(4,5) => 3
      1  -> (1,1) => 1
    Conventions:
      ways(0) = 1
      negatives use |n|
    """
    if n == 0:
        return 1
    n = abs(n)
    r = int(n ** 0.5)
    count = 0
    for d in range(1, r + 1):
        if n % d == 0:
            count += 1  # each d≤sqrt(n) represents one unordered pair (d, n//d)
    return count


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

# assert ways(0) == 1
# assert ways(1) == 1
# assert ways(12) == 5          # {1,2,3,4,6}
# assert ways(20) == 5          # {1,2,4,5,10}

# ns = ["Charlie", "Alice", "Bob"]
# ss = [70, 85, 85]
# assert sort_names(ns, ss) == ["Alice", "Bob", "Charlie"]  # 85s first, A then B