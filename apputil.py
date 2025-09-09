import numpy as np


def ways(n: int, coin_types=None) -> int:
    """
    Return the number of ways to make n cents using pennies (1) and nickels (5).
    Optional: can accept an arbitrary list of coin values.
    """
    if n < 0:
        return 0
    if coin_types is None:
        coin_types = [1, 5]

    # Only pennies and nickels case
    if coin_types == [1, 5]:
        # For each possible number of nickels, check if remainder can be filled with pennies
        count = 0
        for k in range(n // 5 + 1):
            if (n - 5 * k) >= 0:
                count += 1
        return count

    # Optional extension for arbitrary coin sets
    # Dynamic programming (classic coin change count)
    dp = [0] * (n + 1)
    dp[0] = 1
    for c in coin_types:
        for amt in range(c, n + 1):
            dp[amt] += dp[amt - c]
    return dp[n]


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
assert ways(20) == 5
assert ways(3) == 1
assert ways(0) == 1
assert ways(100, [25, 10, 5, 1]) == 242

ns = ["Charlie", "Alice", "Bob"]
ss = [70, 85, 85]
assert lowest_score(ns, ss) == "Charlie"  # lowest is 70
assert sort_names(ns, ss) == ["Alice", "Bob", "Charlie"]
