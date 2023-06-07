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

from fast_constraints import filtered_constraints, keys
from solver import LPWrapper, build_solver


def get_valid_aux(auxfile="", num=-1):
    """Get the first <num> auxiliary arrays from a user-input file"""
    if auxfile == "":
        auxfile = str(input("Enter path to feasible aux file: "))

    # get the first <runs> many auxes
    with open(auxfile, "r") as file:
        return [json.loads(line) for line in file.readlines()[:num]]


#################################
## DIFFERENT SIEVE METHODS
#################################


@cache
def lvec_sieve(circuit, percent=0.2):
    """Currying method. Filters out constraints based on the relative importance of the input."""
    avglvecs = {inspin: circuit.avglvec_dist(inspin) for inspin in circuit.inspace}
    sorted_inspins = sorted(avglvecs, key=avglvecs.get)
    # add weights


@cache
def random_sieve(circuit, percent=0.2):
    """Currying method. Keeps only <percent> of the constraints, randomly throws out the rest."""
    num_outaux = 2 ** (circuit.M + circuit.A)
    num_constraints = int(num_outaux * percent)

    def sieve(inspin):
        return random.sample(list(range(num_outaux)), num_constraints), num_constraints

    return sieve, num_constraints


def partial_constraint_solver(circuit, sieve, percent=0.2):
    """Builds a partial constraint solver"""
    make, terms = filtered_constraints(circuit=circuit, degree=2, sieve=sieve)
    lpwrap = LPWrapper(keys=terms)

    for inspin in circuit.inspace:
        lpwrap.add_constraints(M=make(inspin))

    return lpwrap


def get_random_auxarray(circuit, A):
    auxspace = Spinspace((A,))
    return [auxspace.rand().spin() for _ in circuit.inspace]


def plot_random_sieve(N1=3, N2=4, A=6, runs=200):
    percents = np.linspace(0.001, 0.3, 100)
    false_pos_percents = []

    for i, percent in enumerate(percents):
        circuit = IMul(N1, N2)
        circuit.set_all_aux(get_random_auxarray(circuit, A))
        sieve, num_constraints = random_sieve(circuit=circuit, percent=percent)
        false_pos_count = 0
        for run in range(runs):
            print(f"Run {run} at percent { percent }:")
            print(f"  building solver...")
            solver = partial_constraint_solver(circuit, sieve, percent=percent).solver
            print(f"  solving...", end="")
            status = solver.Solve()
            print(f"  done.")
            psolved = status == solver.OPTIMAL
            print(f"Status: {status}")

            if psolved:
                false_pos_count += 1

            print(f"  false positive: {psolved}")

        print(f"----------------\nSUMMARY OF RUN {i}\n")
        print(f"  percentage of constraints used: {percent}")
        print(f"  number of false positives: {false_pos_count}")

        false_pos_percents.append(false_pos_count / runs)

    plt.xlabel("Percentage of constraints considered")
    plt.ylabel("Percentage of false positives")
    plt.title("Random Sieve Analysis")
    plt.plot(percents, false_pos_percents)
    plt.savefig("figure.png")

    for i in range(len(percents)):
        print(f"  Percent {percents[i]} ~~> {false_pos_percents[i]} false positives.")


if __name__ == "__main__":
    plot_random_sieve(N1=3, N2=4, A=6)
