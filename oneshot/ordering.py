"""
Experiment designed by Isaac. Tries different implementations of injective objective functions.
"""

from oneshot import MLPoly, generate_polynomial
from ising import IMul


def invert_list(l):
    """Used to invert roles of key and value for list of ints, (key -> value) -> (value -> key)"""
    return [i for (i, j) in sorted(list(enumerate(l)), key=lambda e: e[1])]


def kahn_algo_weak_constraint(circuit, seed=0):
    import random

    if seed != 0:
        random.seed(seed)

    # L = empty list that will contain the sorted spins, S = nodes with no incoming edge
    L = []
    S = set(circuit.inout(s).asint() for s in circuit.inspace)

    # store dictionary of all correct
    allcorrect = {circuit.inout(s).asint(): s for s in circuit.inspace}

    while S:
        # choose an element from S, remove it and add it to L
        n = random.choice(tuple(S))
        S.remove(n)
        L.append(n)

        # if n corresponds to a correct inout pair, add all its incorrect outputs to S
        if n in allcorrect:
            correct = allcorrect[n]
            for wout in [s for s in circuit.allwronginout(correct)]:
                S.add(wout.asint())

    return L


def spinrank(circuit, seed=0):
    """Returns a random list rank where rank[spin] = rank of spin"""
    return invert_list(kahn_algo_weak_constraint(circuit, seed))


#######################################
### Test for ordering methods
#######################################
def basic_kahn_test(N1, N2):
    circuit = IMul(N1, N2)
    L = kahn_algo_weak_constraint(circuit)
    print(L)
    L = invert_list(L)
    print(L)

    for inspin in circuit.inspace:
        allwell = True
        cinout = circuit.inout(inspin).asint()
        for winout in circuit.allwronginout(inspin):
            if L[winout.asint()] < L[cinout]:
                allwell = False

        print(f"input level {inspin.asint()} satisfied: {allwell}")


if __name__ == "__main__":
    basic_kahn_test(1, 1)
