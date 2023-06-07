from ising import IMul, PICircuit
from functools import cache
from copy import Error
from spinspace import Spinspace, Spin
from abc import abstractmethod
from ortools.linear_solver import pywraplp
import numpy as np
from matplotlib.pyplot import plot as plt
import random
import json


def get_valid_aux(auxfile="", num=-1):
    if auxfile == "":
        auxfile = str(input("Enter path to feasible aux file: "))

    # get the first <runs> many auxes
    with open(auxfile, "r") as file:
        return [json.loads(line) for line in file.readlines()[:num]]


def build_partial_solver(circuit, input_spins=[]):
    """Builds the lp solver for this circuit, returns solver. Made purposefully verbose/explicit to aid in debugging, should be shortened eventually however."""

    # check that each input has same number of auxiliary states
    if circuit.check_aux_all_set(input_spins) == False:
        raise Error(
            f"Not all auxiliary states are the same size! Cannot build constraints.\n { {s : circuit.Ain(s) for s in input_spins} }"
        )

    G = circuit.Gin(circuit.inspace[0])
    if input_spins == []:
        input_spins = [s for s in circuit.inspace]

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
        correct_inout_pair = circuit.inout(inspin)
        for wrong in circuit.allwrong(inspin):
            inout_pair = Spin.catspin(spins=(inspin, wrong))

            # build the constraint corresponding the difference of correct and incorrect output
            constraint = solver.Constraint(0.001, inf)
            s = correct_inout_pair.spin()
            t = inout_pair.spin()
            # set the h and J coefficient values
            for i in range(G):
                constraint.SetCoefficient(params[i, i], float(t[i] - s[i]))
                for j in range(i + 1, G):
                    constraint.SetCoefficient(
                        params[i, j], float(t[i] * t[j] - s[i] * s[j])
                    )

    # print(f"skipped {tally}")
    return solver


def random_constraint_solver(circuit, percent=0.5):
    """Uniformly throws out constraints"""

    input_spins = [s for s in circuit.inspace]

    # check that each input has same number of auxiliary states
    if circuit.check_aux_all_set(input_spins) == False:
        raise Error(
            f"Not all auxiliary states are the same size! Cannot build constraints.\n { {s : circuit.Ain(s) for s in input_spins} }"
        )

    G = circuit.Gin(circuit.inspace[0])
    if input_spins == []:
        input_spins = [s for s in circuit.inspace]

    solver = pywraplp.Solver.CreateSolver("GLOP")
    inf = solver.infinity()

    # set all the variables
    params = {}
    for i in range(G):
        params[i, i] = solver.NumVar(-inf, inf, f"h_{i}")
        for j in range(i + 1, G):
            params[i, j] = solver.NumVar(-inf, inf, f"J_{i},{j}")

    total_constraint = 0
    constraint_count = 0
    for inspin in input_spins:
        correct_inout_pair = circuit.inout(inspin)
        for wrong in circuit.allwrong(inspin):
            total_constraint += 1
            if random.random() > percent:
                continue

            else:
                constraint_count += 1

            inout_pair = Spin.catspin(spins=(inspin, wrong))

            # build the constraint corresponding the difference of correct and incorrect output
            constraint = solver.Constraint(0.001, inf)
            s = correct_inout_pair.spin()
            t = inout_pair.spin()
            # set the h and J coefficient values
            for i in range(G):
                constraint.SetCoefficient(params[i, i], float(t[i] - s[i]))
                for j in range(i + 1, G):
                    constraint.SetCoefficient(
                        params[i, j], float(t[i] * t[j] - s[i] * s[j])
                    )

    # print(f"skipped {tally}")
    return solver, constraint_count / total_constraint


def get_random_auxarray(circuit, A):
    auxspace = Spinspace((A,))
    return [auxspace.rand().spin() for _ in circuit.inspace]


def false_positives_random_constraints(
    auxfile="", N1=3, N2=4, A=6, runs=100, percent=0.2
):
    circuitlist = [IMul(N1, N2) for _ in range(runs)]

    data = []
    count = 0
    for circuit in circuitlist:
        count += 1
        circuit.set_all_aux(get_random_auxarray(circuit, A))
        partialsolver, percent_const = random_constraint_solver(
            circuit, percent=percent
        )
        psolved = partialsolver.Solve() == partialsolver.OPTIMAL

        if psolved:
            solver = circuit.build_solver()
            solved = solver.Solve() == solver.OPTIMAL
        else:
            solved = False

        # run the solves

        data.append((psolved, solved, percent_const))

        print(
            f"Run {count}:\n  reduced constraints satisfied: {psolved}\n  full constraints satisfied: {solved}\n  percentage of constraints used: {percent_const}\n  false positive: {solved == False and psolved == True}"
        )

    false_positives = [entry[2] for entry in data if entry[1] != entry[0]]

    print(f"Number of false positives: {len(false_positives)}")
    print(false_positives)


if __name__ == "__main__":
    false_positives_random_constraints()
