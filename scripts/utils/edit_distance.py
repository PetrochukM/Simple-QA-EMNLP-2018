from Levenshtein import distance
from functools import lru_cache


@lru_cache(maxsize=65536)
def edit_token_distance(needle, haystack):
    """
    Calculates the fuzzy match of needle in haystack, using a modified version of the Levenshtein
    distance algorithm.

    `levenshtein_word_aligned_distance` only allows the operations delete, insert and substitute of
    words. The cost of substituting a word is Levenshtein between the two words.

    Returns:
        distance: int value concerning the word aligned edit distance 
        start_index: the start index of the min distance alignment between needle & haystack
        end_index: the end index of the min distance alignment between needle & haystack
    """
    m, n = len(needle), len(haystack)

    if not n:
        return m

    row1 = [0] * (n + 1)
    row1_start_index = list(range(0, n + 1))
    for i in range(0, m):
        row2 = [row1[0] + len(needle[i])]  # insertion
        row2_start_index = [row1_start_index[0]]
        for j in range(0, n):
            paths = [
                (row1[j + 1] + len(needle[i]), row1_start_index[j + 1]),  # deletion
                (row1[j] + distance(needle[i], haystack[j]), row1_start_index[j]),  # substitution
                (row2[j] + len(haystack[j]), row2_start_index[j]),  # insertion
            ]
            min_cost, start_index = min(paths, key=lambda p: p[0])
            row2.append(min_cost)
            row2_start_index.append(start_index)
        row1 = row2
        row1_start_index = row2_start_index

    min_cost = min(row1)
    # NOTE: multiple minimum cost spans
    min_spans = [(cost, row1_start_index[end_index], end_index)
                 for end_index, cost in enumerate(row1) if cost == min_cost]
    # NOTE: pick the smallest span that is farthest to the left
    min_cost, start_index, end_index = max(min_spans, key=lambda s: (s[1] - s[2], s[1]))
    return min_cost, int(start_index), int(end_index)


# Reference:
# http://ginstrom.com/scribbles/2007/12/01/fuzzy-substring-matching-with-levenshtein-distance-in-python/


@lru_cache(maxsize=65536)
def edit_substring_distance(needle, haystack):
    """
    Calculates the fuzzy match of needle in haystack, using a modified version of the Levenshtein
    distance algorithm.
    """
    m, n = len(needle), len(haystack)

    # base cases
    if m == 1:
        return needle not in haystack
    if not n:
        return m

    row1 = [0] * (n + 1)
    for i in range(0, m):
        row2 = [i + 1]
        for j in range(0, n):
            cost = (needle[i] != haystack[j])
            row2.append(
                min(
                    row1[j + 1] + 1,  # deletion
                    row2[j] + 1,  # insertion
                    row1[j] + cost)  # substitution
            )
        row1 = row2
    return min(row1)
