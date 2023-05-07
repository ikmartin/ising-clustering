import pickle, math
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

n1 = 2
n2 = 3

def test_on_AND():
    circuit = AND()
    circuit.set_aux([[-1,-1,1,1]])
    M = np.array(sum([all_wrong_answers(circuit, s) for s in circuit.inspace], start=[]))
    print(M)
    regularizer = np.ones(M.shape[1])
    b = -10e-6 * np.ones(M.shape[0])

    result = linprog(regularizer, b_ub = b, A_ub = M, bounds = (-10, 10))
    if result.success:
        print("SUCCESS!")
    return result.success

def test_aux_array(aux_array):
    circuit = IMul(n1, n2)
    circuit.set_aux(aux_array)
    solver = circuit.build_solver()
    status = solver.Solve()
    
    return status == solver.OPTIMAL


def all_wrong_answers(circuit, spin):
    correct_spin = circuit.inout(spin).vspin().spin()
    rows = [
            correct_spin - Spin.catspin(spins = (spin, outaux)).vspin().spin()
            if Spin.catspin(spins = (spin, outaux)).asint() != circuit.inout(spin).asint()
            else None
            for outaux in circuit.fulloutspace
            ]
    return list(filter(lambda row: row is not None, rows))

def attempt(idx):
    data = cluster_carver(n1, n2)
    print(len(data))
    circuit = IMul(n1, n2)
    num_clusters = len(data)
    num_aux_spins = math.ceil(math.log2(num_clusters)) + 1

    possible_aux_spins = np.array([spin.spin() for spin in Spinspace((num_aux_spins,))], dtype = np.int8)
   
    aux_arrays = []
    for selected_pattern_ids in permutations(range(2 ** num_aux_spins), num_clusters):
        selected_patterns = possible_aux_spins[np.array(selected_pattern_ids)]
        
        aux_array = np.zeros((2 ** (n1 + n2), num_aux_spins), dtype = np.int8)
        for cluster, pattern in zip(data, selected_patterns):
            input_ids = np.array([spin.asint() for spin in cluster])
            aux_array[input_ids] = pattern

        aux_arrays.append(aux_array)

    #with open('aux_array_table.dat', 'w') as FILE:
    #    for array in aux_arrays:
    #        FILE.write(str(array.tolist()) + '\n')

    #feasibility = ParallelProgress(n_jobs = -1)([delayed(test_aux_array)(aux_array, n1, n2) for aux_array in aux_arrays])
    feasibility = np.array(list(pqdm(aux_arrays, test_aux_array, n_jobs=24)))
    return np.any(feasibility)

def main():
    for i in range(100):
        print(attempt(i))

def test_feasible_aux():
    import json

    with open('data/all_feasible_MUL_2x2x1.dat') as file:
        tally = 0
        for line in file:
            aux = [json.loads(line)[0]]
            print(aux)
            test_aux_array(aux)

if __name__ == '__main__':
    main()
    #print(test_on_AND())
    #print(test_feasible_aux())
