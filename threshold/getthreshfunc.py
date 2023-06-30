from itertools import combinations
import numpy as np
from ortools.linear_solver import pywraplp


def dec2bin(num, fill):
    return list(bool(int(a)) for a in bin(num)[2:].zfill(fill))


def bin2dec(blist):
    return sum([1 << i if k else 0 for i, k in enumerate(blist)])


class BoolFunc:
    def __init__(self, dim, vals):
        self.dim = dim
        self.vals = vals

    def __call__(self, x):
        if isinstance(x, int):
            return self.vals[x]

        else:
            assert isinstance(x, list)
            return self.vals[bin2dec(x)]

    def __getitem__(self, x):
        return self.vals[x]

    @staticmethod
    def randfunc():
        pass


def build_solver(bfunc):
    """Builds the lp solver for this circuit, returns solver. Made purposefully verbose/explicit to aid in debugging, should be shortened eventually however."""

    # check that each input has same number of auxiliary states

    solver = pywraplp.Solver.CreateSolver("GLOP")
    inf = solver.infinity()

    # set all the variables
    params = {}
    for i in range(G):
        params[i, i] = solver.NumVar(-inf, inf, f"h_{i}")
        for j in range(i + 1, G):
            params[i, j] = solver.NumVar(-inf, inf, f"J_{i},{j}")

    # we treat case with and without aux separately
    for inspin in input_spins:
        correct_inout_pair = self.inout(inspin)
        for wrong in self.allwrong(inspin):
            inout_pair = Spin.catspin(spins=(inspin, wrong))

            # build the constraint corresponding the difference of correct and incorrect output
            constraint = solver.Constraint(0.001, inf)
            s = correct_inout_pair.spin()
            t = inout_pair.spin()
            for i in range(G):
                constraint.SetCoefficient(params[i, i], float(t[i] - s[i]))
                for j in range(i + 1, G):
                    constraint.SetCoefficient(
                        params[i, j], float(t[i] * t[j] - s[i] * s[j])
                    )

    # print(f"skipped {tally}")
    return solver
