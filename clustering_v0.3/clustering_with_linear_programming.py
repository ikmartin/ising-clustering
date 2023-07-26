from binarytree import Node, NodeValue
from functools import cache
from itertools import combinations
from spinspace import Spinspace, Spin, vdist, qvec
from typing import Callable, Optional, Tuple, Set, Any
from ortools.linear_solver import pywraplp
from ising import PICircuit, IMul, AND
import numpy as np
import functional_clustering as funclust


def circuit_solver(circuit: PICircuit, inputs: list[Spin] = []):
    # by default, use constraints for all inputs
    if inputs == []:
        inputs = circuit.inspace.tospinlist()

    # setup solver
    solver = pywraplp.Solver.CreateSolver("GLOP")

    # set all solver variables
    inf = solver.infinity()
    hJ_var = [solver.NumVar(-inf, inf, f"var_{i}") for i in range(circuit.num_vars)]

    print(hJ_var)
    # add constraints to the solver
    for s in inputs:
        # get the full correct in/out/aux spin and save its vspin
        correct_spin = circuit.inout(s)
        correct_vspin = correct_spin.vspin().spin()

        # iterate through all other out/aux pairs
        for t in circuit.outauxspace:
            spin = Spin.catspin((s, t))

            # this prevents adding the inconsistent constraint 0 < 0
            if spin == correct_spin:
                pass

            coeff = spin.vspin().spin() - correct_vspin
            constraint = solver.Constraint(0.001, inf)
            for i, c in enumerate(coeff):
                constraint.SetCoefficient(hJ_var[i], float(c))

    # set the objective, only need to find some solution if it exists
    solver.Minimize(0)

    status = solver.Solve()

    result = {}
    if status == solver.OPTIMAL:
        rtn = True
        for var in hJ_var:
            result[var.name()] = var.solution_value()
    else:
        rtn = False

    # Free all resources associated with the model

    solver.Clear()

    return rtn, result


def AND_example():
    circuit = IMul(1, 1)
    rtn, result = circuit_solver(circuit)
    print(rtn)


if __name__ == "__main__":
    AND_example()
