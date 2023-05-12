"""
Performs iterative_clustering using
  - "LP-solver" refine_criterion, implemented using carver's criterion
  - likelihood refine_method, clustering inputs together if their auxiliary spins 'ought' to match
"""

from typing import Callable, Optional, Tuple, Set, Any
from functools import cache
from functional_clustering import (
    farthest_centers,
    outlier_centers,
    popular_centers,
    virtual_hamming_distance,
    hamming_distance,
    lvec_distance,
    carver,
    lpsolver_criterion,
    general_refine_method,
    iterative_clustering,
    FlexNode,
)
from itertools import combinations
from ising import IMul, PICircuit
from spinspace import Spin, Spinspace
import numpy as np


def refine_with_improvement(circuit, order_inputs, likelihood):
    def distance(vec1, vec2):
        return float(np.linalg.norm(vec1 - vec2))

    def score(lvecs):
        return sum(distance(vec1, vec2) for vec1, vec2 in combinations(lvecs, r=2))

    def method(clust: Set[Spin]):
        import random

        sorted_clust = order_inputs(clust, likelihood)
        auxchoice = {inspin: random.choice([-1, 1]) for inspin in clust}

        # store the best lvecs for more efficient comparison
        neglvec = {inspin: circuit.neglvec(inspin) for inspin in clust}
        poslvec = {inspin: circuit.poslvec(inspin) for inspin in clust}

        # print status
        print(f"Refining the cluster\n   {clust}")

        oldlvecs = lvecs = [
            neglvec[s] if auxchoice[s] == -1 else poslvec[s] for s in clust
        ]
        oldscore = score(oldlvecs)
        # aux_array
        for inspin in sorted_clust:
            # swap the current auxiliary, compute new lvecs, store score
            auxchoice[inspin] *= -1
            newlvecs = [neglvec[s] if auxchoice[s] == -1 else poslvec[s] for s in clust]
            newscore = score(newlvecs)

            # if improved, keep change
            if newscore < oldscore:
                print(f"Changed {inspin} to {auxchoice[inspin]}")
                oldlvecs = newlvecs
                oldscore = newscore
            # if not, change back
            else:
                auxchoice[inspin] *= -1
                print(f"Kept {inspin} at {auxchoice[inspin]}")

        for s in clust:
            circuit.add_single_aux(s, auxchoice[s])

        posclust = set([s for s, val in auxchoice.items() if val == 1])
        negclust = clust.difference(posclust)

        print("Refined to the clusters")
        print(f"  {posclust}")
        print(f"  {negclust}")
        return posclust, negclust

    return method


def refine_with_average_aux_likelihood(circuit, order_inputs, likelihood):
    def distance(vec1, vec2):
        print(f"vec1 len: {len(vec1)}  vec2 len: {len(vec2)}")
        return float(np.linalg.norm(vec1 - vec2))

    def method(clust: Set[Spin]):
        import random

        sorted_clust = random.sample(clust, len(clust))

        print(sorted_clust)

        # initialize the positive and negative auxiliary assignments
        clust_pos = set([sorted_clust[0]])
        clust_neg = set([sorted_clust[1]])

        # store the best lvecs for more efficient comparison
        current_lvecs = [
            circuit.poslvec(sorted_clust[0]),
            circuit.neglvec(sorted_clust[1]),
        ]

        # print status
        print(f"Refining the cluster {clust}")
        print(f"Chose {sorted_clust[0]} as first positive input")

        # aux_array
        for inspin in sorted_clust[2:]:
            # store the positive and negative lvecs
            poslvec = circuit.poslvec(inspin)
            neglvec = circuit.neglvec(inspin)

            print(f"on input {inspin} of shape {circuit.inout(inspin).shape}")
            # store the positive and negative likelihood
            poslike = sum([distance(poslvec, x) for x in current_lvecs])
            print("now negative")
            neglike = sum([distance(neglvec, x) for x in current_lvecs])

            print(f"pos score: {poslike},  neg score: {neglike}")
            # assign to choice with higher likelihood
            if poslike < neglike:
                print(f"Adding +1 aux to {inspin}")
                clust_pos.add(inspin)
            else:
                print(f"Adding -1 aux to {inspin}")

        clust_neg = clust.difference(clust_pos)
        print("refined to")
        print(clust_pos, clust_neg)
        for spin in clust_pos:
            circuit.add_single_aux(spin, +1)

        for spin in clust_neg:
            circuit.add_single_aux(spin, -1)

        return set(clust_pos), set(clust_neg)

    return method


def lvec_distance(circuit: PICircuit) -> Callable:
    @cache
    def distance(spin1: Spin, spin2: Spin):
        lvec1, lvec2 = circuit.lvec(spin1), circuit.lvec(spin2)
        return np.linalg.norm(lvec1 - lvec2)

    return distance


def order_by_avg_likelihood(clust, likelihood):
    return sorted(clust, key=lambda x: sum(likelihood(x, i) for i in clust if i != x))


def cluster_with_aux_likelihood(circuit):
    refine_criterion = lpsolver_criterion(circuit)
    refine_method = refine_with_average_aux_likelihood(
        circuit, order_by_avg_likelihood, likelihood=lvec_distance(circuit)
    )

    clustering = iterative_clustering(refine_criterion, refine_method)

    root = FlexNode(set(circuit.inspace))
    tree = clustering(root)

    result_clusters = [leaf.value for leaf in tree.leaves]
    return result_clusters


def lvec_cluster_popular(circuit):
    find_centers_method = popular_centers
    distance = lvec_distance(circuit)
    refine_criterion = carver(circuit)
    refine_method = general_refine_method(distance, find_centers_method)

    clustering = iterative_clustering(refine_criterion, refine_method)

    root = FlexNode(set(circuit.inspace))
    tree = clustering(root)

    result_clusters = [leaf.value for leaf in tree.leaves]
    return result_clusters


def lvec_cluster_outlier(circuit):
    find_centers_method = outlier_centers
    distance = lvec_distance(circuit)
    refine_criterion = carver(circuit)
    refine_method = general_refine_method(distance, find_centers_method)

    clustering = iterative_clustering(refine_criterion, refine_method)

    root = FlexNode(set(circuit.inspace))
    tree = clustering(root)

    result_clusters = [leaf.value for leaf in tree.leaves]
    return result_clusters


def main_pop():
    N1, N2 = 2, 3
    clusters = lvec_cluster_popular(IMul(N1, N2))

    for leaf in clusters:
        print(sorted(leaf))

    print(len(clusters))


def main_out():
    N1, N2 = 2, 3
    clusters = lvec_cluster_outlier(IMul(N1, N2))

    for leaf in clusters:
        print(sorted(leaf))

    print(len(clusters))


def main_avg_aux_likelihood():
    N1, N2 = 2, 3
    circuit = IMul(N1, N2)
    clusters = cluster_with_aux_likelihood(circuit)

    print(f"Clustering length len(clusters):")
    for leaf in clusters:
        print(sorted(leaf))

    solver = circuit.build_solver()
    print(f"Solved?", solver.Solve() == solver.OPTIMAL)


if __name__ == "__main__":
    main_avg_aux_likelihood()
    # for i in range(10):
    # main_pop()
    # main_out()
