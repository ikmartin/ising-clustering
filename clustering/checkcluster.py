"""
Generalization written by Isaac Martin of gen_aux_table.py written by Andrew Moore.

Accepts a clustering of the input space and checks is against all possible auxiliary state assignments.
"""

from spinspace import Spin, Spinspace
from ising import IMul, AND
import numpy as np
from itertools import permutations, product
from tqdm import tqdm
from scipy.optimize import linprog
from joblib import Parallel, delayed
from pqdm.processes import pqdm
import math

"""
Script for generating a list of possible auxilliary arrays given a clustering of input
spins specified in clusters.dat

Uses a different centers choice than experiment.py
"""

# these are global variables
n1, n2 = 2, 3
circuit = IMul(n1, n2)


# this needs to be defined globally for use in pqdm
# this is because pqdm uses pickle and pickle cannot
# pickle local objects
def test_aux_array(circuit):
    def check(aux_array):
        circuit.set_aux(aux_array)
        solver = circuit.build_solver()
        return solver.Solve() == solver.OPTIMAL

    return check


def check_all_aux(
    circuit,
    clusters: list[list[Spin]],
    aux_dim=-1,
):
    """Attempts to solve the given clustering by checking every possible auxiliary state assignment."""

    check = test_aux_array(circuit)
    # store the number of clusters
    num_clusters = len(clusters)  # number of clusters

    # the dimension of the aux spinspace
    aux_dim = math.ceil(math.log2(num_clusters)) if aux_dim == -1 else aux_dim
    print(aux_dim)

    # the set of all possible auxiliary states
    possible_aux_spins = np.array(
        [spin.spin() for spin in Spinspace((aux_dim,))], dtype=np.int8
    )

    aux_arrays = []
    for selected_pattern_ids in permutations(range(2**aux_dim), num_clusters):
        # a choice of aux state for each cluster
        selected_patterns = possible_aux_spins[np.array(selected_pattern_ids)]

        # container for the aux array
        aux_array = np.zeros((2 ** (circuit.N), aux_dim), dtype=np.int8)
        for cluster, pattern in zip(clusters, selected_patterns):
            input_ids = np.array([spin.asint() for spin in cluster])
            aux_array[input_ids] = pattern

        aux_arrays.append(aux_array)

    feasibility = np.array([check(aux_array) for aux_array in aux_arrays])
    print(feasibility)
    return np.any(feasibility)


if __name__ == "__main__":
    print("No test implemented")
