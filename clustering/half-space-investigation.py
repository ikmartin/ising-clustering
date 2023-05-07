import numpy as np
from itertools import product
import random as rand
import copy


def sprod(a, b):
    dot = np.array(a, b)
    return np.sign(dot)


dim = 4
hypercube = product([-1, 1], repeat=dim)
samples = 2
norm_vecs = rand.sample(list(copy.copy(hypercube)), samples)
norm_vecs = [np.array(vec) for vec in norm_vecs]  # convert to numpy array


def check(u: np.ndarray):
    for vec in norm_vecs:
        if np.dot(u, vec) < 0:
            return False

    return True


def adcheck(u: np.ndarray):
    n0 = norm_vecs[0]
    n1 = norm_vecs[1]
    qvec = n0 + n1
    a = qvec + -1 * np.sign(qvec)
    print(a)
    return np.dot(u, a) >= 0


avg_vec = sum(norm_vecs)
interior = []
boundary = []
for point in copy.copy(hypercube):
    point = np.array(point)

    status = 0  # 0 for interior, 1 for boundary, 2 for neither
    for vec in norm_vecs:
        dot = np.dot(point, vec)
        # if outside one hyperplane
        if dot < 0:
            status = 2
            break
        # if on boundrary of hyperplane
        if dot == 0:
            status = 1

    if status == 0:
        interior.append(point)
    elif status == 1:
        boundary.append(point)


for point in hypercube:
    check1 = check(np.array(point))
    check2 = adcheck(np.array(point))
    print(point, int(check1), int(check2))
