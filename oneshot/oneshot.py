from itertools import chain, combinations
from more_itertools import powerset
from typing import Callable
from math import prod, ceil
from functools import cache
import numpy as np
from copy import deepcopy


class MLPoly:
    def __init__(self, coeffs=None):
        self.coeffs = coeffs if coeffs else {}
        self.clean()

    def clean(self):
        self.coeffs = {key: value for key, value in self.coeffs.items() if value}

    def get_coeff(self, term: tuple) -> float:
        term = tuple(sorted(term))
        return self.coeffs[term] if term in self.coeffs else 0

    def add_coeff(self, term: tuple, value: float) -> None:
        term = tuple(sorted(term))
        self.coeffs[term] = self.coeffs[term] + value if term in self.coeffs else value
        self.clean()

    def set_coeff(self, term: tuple, value: float) -> None:
        term = tuple(sorted(term))
        self.coeffs[term] = value
        self.clean()

    def num_variables(self):
        return 1 + max([max(key) for key in self.coeffs])

    def degree(self):
        return max([len(key) for key in self.coeffs])

    @cache
    def __call__(self, args: tuple) -> float:
        return sum(
            [
                self.coeffs[key] * prod([args[i] for i in key])
                for key, value in self.coeffs.items()
            ]
        )

    def __str__(self):
        sorted_items = sorted(self.coeffs.items(), key=lambda pair: len(pair[0]))
        terms = [
            [
                " + " if value > 0 else " - ",
                str(abs(value)),
                "".join([f"x_{i}" for i in key]),
            ]
            for key, value in sorted_items
            if value
        ]
        return "".join(list(chain(*terms))[1:])

    def __repr__(self):
        return self.__str__()


def generate_polynomial(f: Callable, n: int) -> MLPoly:
    """
    Generate coefficient dictionary representing a multilinear pseudo-Boolean polynomial equivalent to the given arbitrary function defined on binary boolean tuples.

    Very computationally expensive in n.

    c.f. "Pseudo-Boolean Optimization" by Boros & Hammer, Proposition 2
    """

    poly = MLPoly()
    for S in powerset(range(n)):
        value_on_set = f(tuple([i in S for i in range(n)]))
        sub_terms = sum([poly.get_coeff(T) for T in list(powerset(S))[:-1]])
        poly.add_coeff(S, float(value_on_set - sub_terms))

    return poly


def max_common_key(poly: MLPoly, size: int, criterion=lambda factor, key, value: True):
    """
    Helper function for multi-term reduction methods. Iterates through the terms of the polynomial and attempts to find the term of length size which is the factor of the most terms which satisfy the criterion function. Returns the key for the common term and the list of keys for the monomials that it factorizes.

    Note that this algorithm has been written with simplicity in mind and will scale horribly to polynomials with larger numbers of variables.
    """

    n = poly.num_variables()

    term_table = {
        tuple(sorted(factor)): [
            key
            for key, value in poly.coeffs.items()
            if criterion(factor, key, value)  # monomial satisfies filter criterion
        ]
        for factor in combinations(range(n), size)
    }

    factor = max(term_table, key=lambda i: len(term_table[i]))
    return factor, term_table[factor]


def get_common_key(poly: MLPoly, criterion: Callable):
    """
    Tries to find common keys to extract up to degree ceil(d/2) where poly is degree d. Uses a heuristic to pick the best one.
    """

    d = poly.degree()
    options = [max_common_key(poly, i, criterion) for i in range(ceil(d / 2) + 1)]

    return max(options, key=lambda pair: len(pair[1]) * len(pair[0]))


def PositiveFGBZ(poly: MLPoly, C: tuple, H: tuple) -> tuple[MLPoly, bool]:
    """
    Implements the algorithm to reduce the order of higher-order positive coefficient monomials by extracting one common term represented by the product of variables in C from a set of monomials H, at the cost of one new auxilliary variable.

    c.f "A Graph Cut Algorithm for Higher-order Markov Random Fields" by Fix, Gruber, Boros, Zabih, Theorem 3.1
    """

    # Make a copy of the polynomial to prevent changing the state of the argument
    poly = MLPoly(deepcopy(poly.coeffs))

    n = poly.num_variables()

    # Now, execute the algorithm by extracting C
    sum_alpha_H = sum([poly.get_coeff(key) for key in H])
    poly.add_coeff(tuple(set(C) | {n}), sum_alpha_H)

    for key in H:
        term1 = tuple(set(key) - set(C))
        term2 = tuple(set(key) - set(C) | {n})
        alpha_H = poly.get_coeff(key)

        poly.add_coeff(term1, alpha_H)
        poly.add_coeff(term2, -alpha_H)
        poly.set_coeff(key, 0)

    return poly


def NegativeFGBZ(poly: MLPoly, C: tuple, H: tuple) -> tuple[MLPoly, bool]:
    """
    Similar to the positive case, except this time the minimally complex option is to extract pairs (similar to Rosenberg)
    """

    # Make a copy of the polynomial to prevent changing the state of the argument
    poly = MLPoly(deepcopy(poly.coeffs))

    n = poly.num_variables()

    # Now, execute the algorithm by extracting C
    sum_alpha_H = sum([poly.get_coeff(key) for key in H])
    poly.add_coeff((n,), -sum_alpha_H)
    poly.add_coeff(tuple(set(C) | {n}), sum_alpha_H)

    for key in H:
        term = tuple(set(key) - set(C) | {n})
        alpha_H = poly.get_coeff(key)

        poly.add_coeff(term, alpha_H)
        poly.set_coeff(key, 0)

    return poly


def Rosenberg(poly: MLPoly, C: tuple, H: tuple, M: float) -> MLPoly:
    """
    Old standard pair-reduction algorithm.
    """

    assert len(C) == 2

    poly = MLPoly(deepcopy(poly.coeffs))

    n = poly.num_variables()

    # replace occurances of the pair with a new auxilliary
    for key in H:
        term = tuple(set(key) - set(C) | {n})
        poly.add_coeff(term, poly.get_coeff(key))
        poly.set_coeff(key, 0)

    # add the penalty term M(xy - 2xa - 2ya + 3a)
    poly.add_coeff(C, M)
    poly.add_coeff((C[0], n), -2 * M)
    poly.add_coeff((C[1], n), -2 * M)
    poly.add_coeff((n,), 3 * M)

    return poly


def FreedmanDrineas(poly: MLPoly) -> MLPoly:
    """
    Simple algorithm which reduces a single higher-order term with a negative coefficient to a sum of quadratic and linear terms at the cost of one extra variable. This method will apply the algorithm to every negative-coefficient higher-order term in the given polynomial.

    c.f. "Energy Minimization via Graph Cuts: Settling What is Possible" by Freedman & Drineas, Section 2.4
    """

    # Preserve the argument polynomial by making a copy
    poly = MLPoly(deepcopy(poly.coeffs))

    n = poly.num_variables()

    reducible_terms = [
        (key, value)
        for key, value in poly.coeffs.items()
        if value < 0 and len(key) > 2  # negative coefficient  # higher-order monomial
    ]

    for key, value in reducible_terms:
        order = len(key)
        poly.add_coeff((n,), -value * (order - 1))
        for i in key:
            poly.add_coeff((i, n), value)

        poly.set_coeff(key, 0)

        # Since we added a new auxilliary, the polynomial now has one more variable.
        n += 1

    return poly


def full_Rosenberg(poly: MLPoly) -> MLPoly:
    rosenberg_criterion = lambda factor, key, value: (
        set(factor).issubset(key) and len(factor) == 2 and len(key) > 2
    )

    M = sum([value for key, value in poly.coeffs.items() if value > 0])
    while poly.degree() > 2:
        C, H = get_common_key(poly, rosenberg_criterion)
        if not len(H):
            break

        poly = Rosenberg(poly, C, H, M)

    return poly


def FGBZ_FD(poly: MLPoly) -> MLPoly:
    """
    Uses positive FGBZ to reduce positive monomials until there are no more left. Then eliminates all negative monomials using FD. Returns a quadratic form.
    """

    positive_FGBZ_criterion = lambda factor, key, value: (
        set(factor).issubset(key)
        and len(factor) < len(key) - 1
        and len(key) > 2
        and len(factor) >= 1
        and value > 0
    )

    negative_FGBZ_criterion = lambda factor, key, value: (
        set(factor).issubset(key)
        and len(factor) < len(key) - 1
        and len(key) > 3
        and len(factor) >= 2
        and value < 0
    )

    while True:
        C, H = get_common_key(poly, positive_FGBZ_criterion)
        if not len(H):
            break

        poly = PositiveFGBZ(poly, C, H)
        print(poly)
        print("")

    print("no positives:")
    print(poly)

    while True:
        C, H = get_common_key(poly, negative_FGBZ_criterion)
        if not len(H):
            break

        poly = NegativeFGBZ(poly, C, H)
        print(poly)
        print("")

    print("no large negatives:")
    print(poly)

    return FreedmanDrineas(poly)
