from ortools.linear_solver.pywraplp import Solver
from ising import IMul, PICircuit
from spinspace import Spin, Spinspace
from constraints import traditional_constraints, get_constraints
from oneshot import reduce_poly, MLPoly

from itertools import combinations

import time, os, torch, click, math


def quadratic_gen(keys, minlist=1, maxlist=-1):
    """Generator which iterates through all possible lists of quadratics (disregarding order) provided by the argument `keys`"""
    # set maxlist to be number of possible combinations
    quads = [k for k in keys if len(k) == 2]
    numquad = len(quads)

    if maxlist == -1:
        maxlist = math.comb(numquad, 2) + 1

    for i in range(minlist, maxlist):
        total_len = math.comb(numquad, i)
        count = 0
        for keylist in combinations(quads, r=i):
            count += 1
            yield keylist, count, total_len


class Unbanner:
    def __init__(self, circuit, degree, admin=None, constraints=None, name=None):
        self.admin = admin
        self.circuit = circuit
        self.degree = degree
        self.M, self.keys = (
            constraints if constraints is not None else get_constraints(circuit, degree)
        )

        print(self.keys)
        self.name = name if name is not None else "solver"

        self.num_terms = self.M.shape[1]
        self.threshold = 0.01
        super().__init__()

    def get_key_ban_indices(self, keylist):
        """Returns the indices of all keys with the key pattern as key/value pairs"""
        # print("Provided keylist:", keylist)
        thing = {
            k
            for k in self.keys
            for key in keylist
            if set(key).issubset(k) == True and len(k) > 2
        }
        # print("Thing:", set(thing))
        valid_terms = set([k for k in self.keys if len(k) > 2]).difference(thing)
        return {self.keys.index(k) for k in valid_terms}

    def run(self, minlist=1, maxlist=-1):
        print(f"[{self.name}] Building...")
        self.solver = self.build_solver()

        highscores = []
        print(f"[{self.name}] Ready.")
        # Run an infinite loop waiting for tasks to be added to the parent queue
        for keys, c, T in quadratic_gen(self.keys, minlist, maxlist):
            banlist = self.get_key_ban_indices(keys)
            coeffs, score = self.solve(list(banlist))
            if c % 100 == 0:
                print(f"{c} of {T} possible {len(keys)}-key combos")

            if score == None:
                pass
                # print(f"Failed to solve by unbanning {keys}")

            else:
                if len(highscores) > 10:
                    del highscores[-1]
                    highscores.append((len(keys), score))
                    highscores = sorted(highscores, key=lambda x: x[1])
                else:
                    highscores.append((len(keys), score))
                print("-----------------------------------------------")
                print(f"Achieved {score} auxspins by unbanning {keys}")
                print(f"  number of quadratic keys: {len(keys)}")
                print(f"  number of bans: {len(banlist)}")
                print(f"  highscores: {highscores}")
                print("-----------------------------------------------")

    def _clear(self):
        for ban_constraint in self.ban_constraints:
            ban_constraint.Clear()

    def _ban(self, index):
        self.ban_constraints[index].SetCoefficient(self.variables[index], 1)

    def build_solver(self):
        solver = Solver.CreateSolver("GLOP")
        inf = solver.infinity()
        self.variables = [
            solver.NumVar(-inf, inf, f"x_{i}") for i in range(self.num_terms)
        ]
        self.ban_constraints = [solver.Constraint(0, 0) for var in self.variables]

        for row in self.M:
            constraint = solver.Constraint(1.0, inf)
            for i, coeff in enumerate(row):
                if not coeff:
                    continue

                constraint.SetCoefficient(self.variables[i], int(coeff))

        # remove variables from the key list if they are already zero
        for i, col in enumerate(self.M.T):
            if col.any() == False:
                del self.keys[i]

        y_vars = [solver.NumVar(-inf, inf, f"y_{var}") for var in self.variables]
        for x, y in zip(self.variables, y_vars):
            constraint1 = solver.Constraint(-inf, 0)
            constraint2 = solver.Constraint(0, inf)
            constraint1.SetCoefficient(x, 1)
            constraint1.SetCoefficient(y, -1)
            constraint2.SetCoefficient(x, 1)
            constraint2.SetCoefficient(y, 1)

        objective = solver.Objective()
        for key, var in zip(self.keys, y_vars):
            if len(key) > 2:
                objective.SetCoefficient(var, 1)

        objective.SetMinimization()

        return solver

    def get_poly(self, coeffs) -> MLPoly:
        coeff_dict = {key: val for key, val in zip(self.keys, coeffs)}
        return MLPoly(coeff_dict)

    def solve(self, bans: tuple):
        self._clear()
        # print(f"bans: {bans}")
        for i in bans:
            self._ban(i)

        status = self.solver.Solve()
        if status:
            return None, None

        coeffs = torch.tensor([var.solution_value() for var in self.variables])
        coeffs[abs(coeffs) < self.threshold] = 0
        poly = self.get_poly(coeffs.tolist())

        reduced_poly = reduce_poly(poly, ["rosenberg"])
        num_aux = reduced_poly.num_variables() - self.circuit.G

        return coeffs, int(num_aux)


def search(circuit, degree):
    Abbanaman = Unbanner(circuit, degree)
    Abbanaman.run(minlist=10)


@click.command()
@click.option("--n1", default=2, help="First input")
@click.option("--n2", default=2, help="Second input")
@click.option("--degree", default=4, help="Degree")
def main(n1, n2, degree):
    circuit = IMul(n1, n2)

    search(circuit, degree)


if __name__ == "__main__":
    main()
