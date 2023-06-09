from itertools import chain, combinations
from more_itertools import powerset
from typing import Callable
from math import prod, ceil
from functools import cache
import numpy as np
from copy import deepcopy


class MLPoly:
    def __init__(self, coeffs=None, clean = False):
        self.coeffs = coeffs if coeffs else {}
        self.threshold = 1e-3
        if clean:
            self.clean()


    def clean(self, threshold=0):
        self.coeffs = {
            key: value for key, value in self.coeffs.items() if abs(value) > self.threshold
        }

    def get_coeff(self, term: tuple) -> float:
        term = tuple(sorted(term))
        return self.coeffs[term] if term in self.coeffs else 0

    def add_coeff(self, term: tuple, value: float) -> None:
        term = tuple(sorted(term))
        new = self.coeffs.get(term, 0) + value
        self.coeffs[term] = new
        if abs(new) < self.threshold:
            self.coeffs.pop(term)

    def set_coeff(self, term: tuple, value: float) -> None:
        if abs(value) < self.threshold:
            self.coeffs.pop(term)
            return

        term = tuple(sorted(term))
        self.coeffs[term] = value

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

    def _format_coeff(self, val):
        if abs(val - round(val)) < 1e-1:
            if round(val) == 1:
                return ''

            return str(round(val))

        
        return f'{val:.2f}'

    def __str__(self):
        preface = f'Poly(degree = {self.degree()}, dim = {self.num_variables()}): '
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        sorted_items = sorted(self.coeffs.items(), key=lambda pair: len(pair[0]))
        terms = [
            [
                " + " if value > 0 else " - ",
                f'{self._format_coeff(abs(value))}',
                "".join([f"x{str(i).translate(SUB)}" for i in key]),
            ]
            for key, value in sorted_items
            if value
        ]
        return preface + "".join(list(chain(*terms))[1:])

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

def get_term_table(poly: MLPoly, criterion = lambda factor, key, value: True, size = None) -> dict:
    n = poly.num_variables()

    if size is not None:
        return {
            tuple(sorted(factor)): [
                (key, value)
                for key, value in poly.coeffs.items()
                if criterion(factor, key, value)  # monomial satisfies filter criterion
            ]
            for factor in combinations(range(n), size)
        }
    
    table = {}
    for i in range(1, poly.degree()):
        table.update(get_term_table(poly, criterion, i))

    return table


def max_common_key(poly: MLPoly, size: int, criterion=lambda factor, key, value: True):
    """
    Helper function for multi-term reduction methods. Iterates through the terms of the polynomial and attempts to find the term of length size which is the factor of the most terms which satisfy the criterion function. Returns the key for the common term and the list of keys for the monomials that it factorizes.

    Note that this algorithm has been written with simplicity in mind and will scale horribly to polynomials with larger numbers of variables.
    """

    term_table = get_term_table(poly, criterion, size)
    factor = max(term_table, key=lambda i: len(term_table[i]))
    return factor, term_table[factor]

standard_heuristic = lambda pair: len(pair[1]) * len(pair[0])

def get_common_key(poly: MLPoly, criterion: Callable, heuristic = standard_heuristic):
    """
    Tries to find common keys to extract up to degree ceil(d/2) where poly is degree d. Uses a heuristic to pick the best one.
    """

    d = poly.degree()
    options = [max_common_key(poly, i, criterion) for i in range(ceil(d / 2) + 1)]

    return max(options, key=heuristic)

def fast_pairs(poly: MLPoly, n=2, minsize = 3):
    pairs = {}
    for key, value in poly.coeffs.items():
        if len(key) < minsize:
            continue
        if len(key) <= n:
            continue
        
        for C in combinations(key, n):
            pairs[C] = pairs.get(C, []) + [(key, value)]

    return pairs

def pfgbz_candidates(poly: MLPoly):
    factors = {}
    for key, value in poly.coeffs.items():
        if len(key) < 3 or value < 0:
            continue
        
        for n in range(1, len(key)-1):
            for C in combinations(key, n):
                factors[C] = factors.get(C, []) + [(key, value)]

    return factors


def nfgbz_candidates(poly: MLPoly):
    factors = {}
    if poly.degree() < 4:
        return factors

    for key, value in poly.coeffs.items():
        if len(key) < 4 or value > 0:
            continue
        
        for n in range(2, len(key)-1):
            for C in combinations(key, n):
                factors[C] = factors.get(C, []) + [(key, value)]

    return factors


weak_positive_FGBZ_criterion = lambda factor, key, value: (
    set(factor).issubset(key)
    and len(factor) < len(key)
    and len(key) > 2
    and len(factor) >= 1
    and value > 0
)

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

rosenberg_criterion = lambda factor, key, value: (
    set(factor).issubset(key) and len(factor) == 2 and len(key) > 2
)


def PositiveFGBZ(poly: MLPoly, C: tuple, H: tuple) -> tuple[MLPoly, bool]:
    """
    Implements the algorithm to reduce the order of higher-order positive coefficient monomials by extracting one common term represented by the product of variables in C from a set of monomials H, at the cost of one new auxilliary variable.

    c.f "A Graph Cut Algorithm for Higher-order Markov Random Fields" by Fix, Gruber, Boros, Zabih, Theorem 3.1
    """

    # Make a copy of the polynomial to prevent changing the state of the argument
    poly = MLPoly({key:val for key, val in poly.coeffs.items()})


    n = poly.num_variables()

    # Now, execute the algorithm by extracting C
    sum_alpha_H = sum([poly.get_coeff(key) for key, value in H])
    poly.add_coeff(tuple(set(C) | {n}), sum_alpha_H)

    for key, value in H:
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
    poly = MLPoly({key:val for key, val in poly.coeffs.items()})

    n = poly.num_variables()

    # Now, execute the algorithm by extracting C
    sum_alpha_H = sum([poly.get_coeff(key) for key, value in H])
    poly.add_coeff((n,), -sum_alpha_H)
    poly.add_coeff(tuple(set(C) | {n}), sum_alpha_H)

    for key, value in H:
        term = tuple(set(key) - set(C) | {n})
        alpha_H = poly.get_coeff(key)

        poly.add_coeff(term, alpha_H)
        poly.set_coeff(key, 0)

    return poly


def Rosenberg(poly: MLPoly, C: tuple, H: tuple) -> MLPoly:
    """
    Old standard pair-reduction algorithm.
    """
    
    M = sum([max(0, value) for key, value in H])

    assert len(C) == 2

    poly = MLPoly({key:val for key, val in poly.coeffs.items()})

    n = poly.num_variables()

    # replace occurances of the pair with a new auxilliary
    for key, value in H:
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

    poly = MLPoly({key:val for key, val in poly.coeffs.items()})
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

def single_FD(poly, C):
    value = poly.coeffs[C]
    order = len(C)

    poly = MLPoly({key:val for key, val in poly.coeffs.items()})
    n = poly.num_variables()

    poly.add_coeff((n,), -value * (order - 1))
    for i in C:
        poly.add_coeff((i, n), value)

    poly.set_coeff(C, 0)

    return poly


def full_Rosenberg(poly: MLPoly, heuristic = standard_heuristic) -> MLPoly:
    while poly.degree() > 2:
        poly, aux_map = single_rosenberg(poly, heuristic)

    return poly

def single_rosenberg(poly: MLPoly, heuristic = standard_heuristic) -> MLPoly:
    #C, H = get_common_key(poly, rosenberg_criterion, heuristic)
    term_table = fast_pairs(poly)
    if not len(term_table):
        return poly, None

    factor = max(term_table, key=lambda i: len(term_table[i]))
    C, H = factor, term_table[factor]
    if not len(H):
        return poly, None

    poly = Rosenberg(poly, C, H)
    aux_map = MLPoly({
        C: 1
    })

    return poly, aux_map


def single_positive_FGBZ(poly: MLPoly) -> MLPoly:
    term_table = pfgbz_candidates(poly)
    if not len(term_table):
        return poly, None

    C = max(term_table, key=lambda term: len(term) * len(term_table[term]))
    H = term_table[C]
    if not len(H):
        return poly, None

    poly = PositiveFGBZ(poly, C, H)
    aux_map = MLPoly({
        () : 1,
        C : -1
    })

    return poly, aux_map


def single_negative_FGBZ(poly: MLPoly) -> MLPoly:
    term_table = nfgbz_candidates(poly)
    if not len(term_table):
        return poly, None

    C = max(term_table, key=lambda term: len(term) * len(term_table[term]))
    H = term_table[C]
    if not len(H):
        return poly, None

    poly = NegativeFGBZ(poly, C, H)
    aux_map = MLPoly({
        C: 1
    })

    return poly, aux_map



def full_positive_FGBZ(poly: MLPoly) -> MLPoly:
    while True:
        poly, aux_map = single_positive_FGBZ(poly)
        if aux_map is None:
            break

    return poly


def full_negative_FGBZ(poly: MLPoly) -> MLPoly:
    while True:
        poly, aux_map = single_negative_FGBZ(poly)
        if aux_map is None:
            break

    return poly

def get_method(method):
    """
    Returns a function pointer for the named method
    """

    if method == 'rosenberg':
        return full_Rosenberg

    if method == '+fgbz':
        return full_positive_FGBZ

    if method == '-fgbz':
        return full_negative_FGBZ

    if method == 'fd':
        return FreedmanDrineas

def reduce_poly(poly: MLPoly, methods) -> MLPoly:
    """
    Applies a sequence of reduction algorithms as defined by the argument.
    """

    for method in methods:
        reduction_method = get_method(method)
        poly = reduction_method(poly)

    return poly
