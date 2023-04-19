def trinum(n):
    return int(n * (n + 1) / 2)


"""
                                        [a1, b1, b2, b3]
[a1,a2,a3,a4], [b1,b2,b3,b4,b5,b6] ~~>  [0,  a2, b4, b5] ~~> [a1,b1,b2,b3,a2,b4,b5,a3,b6,a4]
                                        [0,   0, a3, b6]
                                        [0,   0,  0, a4]

Illustration of the process perfomed by below function
"""


def ditri_array(a: list, b: list):
    """
    Given two arrays (a,b) representing the diagonal and strict upper triangle of a
    triangular matrix respectively, returns the array obtained by concatenating
    row-wise and deleting zero entries.

    Intended as a convenient way to represent parameter arrays (h,J) in the Ising model

    Returns: array whose length is the nth triangular number where n = len(a)

    Parameters
      a (1D array): the diagonal
      b (1D array): the upper triangle

    NOTE: the length of b must be the (n-1)th triangular number where a is the length of a
    """
    # ensure diag and triu are correct lengths
    if len(b) != trinum(len(a) - 1):
        raise TypeError(
            "Expected len(b) = {}, got {} instead".format(len(b), trinum(len(a) - 1))
        )

    if (not isinstance(a, list)) or (not isinstance(b, list)):
        raise TypeError("Parameters (a,b) must be lists!")

    import itertools

    # left and right slice for triu.
    # in the upper triangular matrix
    # the contribution of triu to row
    # i is triu[x:x+d]
    x = lambda N, i: trinum(N) - trinum(N - i) - i
    d = lambda N, i: (N - 1) - i

    # fix the length of a
    N = len(a)
    return list(
        itertools.chain.from_iterable(
            [[a[i]] + b[x(N, i) : x(N, i) + d(N, i)] for i in range(N)]
        )
    )
