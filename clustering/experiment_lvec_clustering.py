"""
Performs iterative_clustering using
  - "LP-solver" refine_criterion, implemented using carver's criterion
  - centers-based refine_method, where distance is computed via Euclidean distance between lvecs
"""

import pickle

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


def lvec_cluster_carver(circuit):
    find_centers_method = popular_centers
    distance = lvec_distance(circuit)
    refine_criterion = carver(circuit)
    refine_method = general_refine_method(distance, find_centers_method)

    clustering = iterative_clustering(refine_criterion, refine_method)

    root = FlexNode(set(circuit.inspace))
    tree = clustering(root)

    result_clusters = [leaf.value for leaf in tree.leaves]
    return result_clusters


def main():
    N1, N2 = 2, 3
    clusters = lvec_cluster_carver(IMul(N1, N2))

    for leaf in clusters:
        print(leaf)

    print(len(clusters))

    with open(f"lvec_clustering_for_MUL{N1}x{N2}.dat", "wb") as FILE:
        pickle.dump(clusters, FILE)


if __name__ == "__main__":
    for i in range(10):
        main()
