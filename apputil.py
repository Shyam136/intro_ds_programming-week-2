import numpy as np


def ways(n: int) -> int:
    """
    Count the number of ways to reach n using steps of size 1 or 2.
    (Classic staircase count: 0->1 way, 1->1 way, 2->2 ways, 3->3 ways, ...)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    # Iterative Fibonacci: ways(0)=1, ways(1)=1, ways(2)=2 ...
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def lowest_score(names, scores):
    """
    Return the name associated with the *lowest* score.
    If there are ties, return the alphabetically-first name among them.
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
    Return the list of names sorted by their corresponding scores (ascending).
    Ties are broken alphabetically by name.
    """
    if len(names) != len(scores):
        raise ValueError("names and scores must have the same length")
    if not names:
        return []

    scores_arr = np.asarray(scores, dtype=float)
    names_arr = np.asarray(names, dtype=object)

    # In np.lexsort, last key is primary. Sort by score, then by name for ties.
    order = np.lexsort((names_arr, scores_arr))  # ascending score, then name
    return names_arr[order].tolist()


# unit tests:
# ways
assert ways(0) == 1 and ways(1) == 1 and ways(2) == 2 and ways(3) == 3 and ways(4) == 5

# lowest_score / sort_names
ns = ["Zed", "Amy", "Bob", "Amy"]
ss = [90, 70, 70, 85]
assert lowest_score(ns, ss) == "Amy"              # tie at 70 -> alphabetical
assert sort_names(ns, ss) == ["Amy", "Bob", "Amy", "Zed"]  # by score asc; tie by name