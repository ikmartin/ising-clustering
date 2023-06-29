from itertools import combinations
import numpy as np
from ortools.linear_solver import pywraplp


def build_solver(self, input_spins=[]):
    """Builds the lp solver for this circuit, returns solver. Made purposefully verbose/explicit to aid in debugging, should be shortened eventually however."""

    # check that each input has same number of auxiliary states
    if self.check_aux_all_set(input_spins) == False:
        raise Error(
            f"Not all auxiliary states are the same size! Cannot build constraints.\n { {s : self.Ain(s) for s in input_spins} }"
        )

    G = self.Gin(self.inspace[0])
    if input_spins == []:
        input_spins = [s for s in self.inspace]

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
