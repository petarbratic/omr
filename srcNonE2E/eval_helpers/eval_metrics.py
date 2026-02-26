# Helpers for Levenshtein (edit) distance: token-level and character-level.

from typing import List


def levenshtein_tokens(ref: List[str], hyp: List[str]) -> int:
    # Edit distance between two token lists.
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    
    if m < n:
        ref, hyp = hyp, ref
        n, m = m, n

    prev = list(range(m + 1))
    cur = [0] * (m + 1)

    for i in range(1, n + 1):
        cur[0] = i
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev

    return prev[m]


def levenshtein_chars(ref: str, hyp: str) -> int:
    # Edit distance between two character sequences.
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    if m < n:
        ref, hyp = hyp, ref
        n, m = m, n

    prev = list(range(m + 1))
    cur = [0] * (m + 1)

    for i in range(1, n + 1):
        cur[0] = i
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, cur = cur, prev

    return prev[m]
