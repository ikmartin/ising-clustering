"""
Performs iterative_clustering using
  - "LP-solver" refine_criterion, implemented using carver's criterion
  - centers-based refine_method, where distance is computed via Euclidean distance between lvecs
"""


from functional_clustering import (
    farthest_centers,
    outlier_centers,
    popular_centers,
    virtual_hamming_distance,
    hamming_distance,
    lvec_distance,
    carver,
    general_refine_method,
    iterative_clustering,
    FlexNode,
)

from ising import IMul
import pickle


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


if __name__ == "__main__":
    for i in range(10):
        main_pop()
        main_out()
