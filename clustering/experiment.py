from functional_clustering import (
    farthest_centers,
    virtual_hamming_distance,
    carver,
    general_refine_method,
    iterative_clustering,
    FlexNode,
)
from ising import IMul
import pickle


def cluster_carver(N1, N2):
    circuit = IMul(N1, N2)
    find_centers_method = farthest_centers
    distance = virtual_hamming_distance(circuit)
    refine_criterion = carver(circuit)
    refine_method = general_refine_method(distance, find_centers_method)

    clustering = iterative_clustering(refine_criterion, refine_method)

    root = FlexNode(set(circuit.inspace))
    tree = clustering(root)

    result_clusters = [leaf.value for leaf in tree.leaves]
    return result_clusters


def main():
    result_clusters = cluster_carver(2, 3)

    for leaf in result_clusters:
        print(leaf)

    print(len(result_clusters))

    with open("clusters.dat", "wb") as FILE:
        pickle.dump(result_clusters, FILE)


if __name__ == "__main__":
    main()
