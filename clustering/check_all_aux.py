"""
Generalization written by Isaac Martin of gen_aux_table.py written by Andrew Moore.

Accepts a clustering of the input space and checks is against all possible auxiliary state assignments.
"""

import pickle, math
from re import I
from spinspace import Spin, Spinspace
from ising import IMul, AND
import numpy as np
from itertools import permutations, product
from tqdm import tqdm
from scipy.optimize import linprog
from joblib import Parallel, delayed
from pqdm.processes import pqdm

from experiment2 import cluster_carver

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
def test_aux_array(aux_array):
    circuit.set_all_aux(aux_array)
    solver = circuit.build_solver()
    status = solver.Solve()
    print(aux_array)
    return status == solver.OPTIMAL


def brute_force_aux_assignment(
    data: list[list[Spin]],
    aux_dim=-1,
):
    """Attempts to solve the given clustering by checking every possible auxiliary state assignment."""

    num_clusters = len(data)  # number of clusters

    # the dimension of the aux spinspace
    aux_dim = math.ceil(math.log2(num_clusters)) + 1 if aux_dim == -1 else aux_dim

    # the set of all possible auxiliary states
    possible_aux_spins = np.array(
        [spin.spin() for spin in Spinspace((aux_dim,))], dtype=np.int8
    )

    aux_arrays = []
    for selected_pattern_ids in permutations(range(2**aux_dim), num_clusters):
        # a choice of aux state for each cluster
        selected_patterns = possible_aux_spins[np.array(selected_pattern_ids)]

        # container for the aux array
        aux_array = np.zeros((2 ** (n1 + n2), aux_dim), dtype=np.int8)
        print(aux_array.shape)
        for cluster, pattern in zip(data, selected_patterns):
            input_ids = np.array([spin.asint() for spin in cluster])
            aux_array[input_ids] = pattern

        aux_arrays.append(aux_array)

    feasibility = np.array(list(pqdm(aux_arrays, test_aux_array, n_jobs=24)))
    return np.any(feasibility)


def test_lvec_clustering():
    from experiment_lvec_clustering import lvec_cluster_popular

    for i in range(100):
        # reset the
        circuit = IMul(n1, n2)
        data = lvec_cluster_popular(circuit)
        if len(data) > 2:
            continue

        for leaf in data:
            print(leaf)
        success = brute_force_aux_assignment(data, aux_dim=1)
        print(f"Success: {success}")


if __name__ == "__main__":
    test_lvec_clustering()
