from functools import cache
from matplotlib import pyplot as plt
from itertools import combinations
from joblib import Parallel, delayed

import json
import random
import os
import math
import numpy as np


# imports for Isaac's code
from filtered_constraints import filtered_constraints as fc, make_correct_rows as crows


auxdirpath = "/home/ikmarti/Desktop/ising-clustering/constraint-reduction/aux_arrays/"


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def read_auxfile(filename):
    auxes = []
    with open(filename) as file:
        for line in file:
            auxes.append(json.loads(line)[0])

    return auxes


def sample_auxfile(filename, samples):
    auxes = []
    with open(filename) as file:
        for line in file:
            auxes.append(json.loads(line))

    return random.sample(auxes, k=samples)


def save_auxfile(auxes, filename):
    path = uniquify(auxdirpath + filename)
    with open(path, "w") as file:
        for aux in auxes:
            file.write(str(aux.tolist()) + "\n")


def and_aux_generator(N1, N2, A, numsamples=10000):
    filename = f"IMul{N1}x{N2}x{A}_AND_AUX.dat"
    N = N1 + N2
    correct = crows(3, 3, None)
    correct = correct.numpy()

    tups = set(
        {
            tuple(random.sample(list(combinations(list(range(N)), r=2)), k=A))
            for _ in range(numsamples)
        }
    )
    auxes = []
    for tup in tups:
        aux_vecs = np.concatenate(
            [np.expand_dims(np.prod(correct[..., key], axis=-1), axis=0) for key in tup]
        ).astype(np.int8)
        auxes.append(aux_vecs)

    save_auxfile(auxes, filename)


if __name__ == "__main__":
    N1 = 4
    N2 = 4
    M = N1 + N2
    G = N1 + N2 + M
    maxA = 20
    for A in range(1, maxA + 1):
        numsamples = min(math.comb(math.comb(G, 2), A), 100000)
        and_aux_generator(N1, N2, A, numsamples)
        print(f"Finished IMul{N1}x{N2}x{A}")
