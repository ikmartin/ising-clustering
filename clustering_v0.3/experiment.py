from functional_clustering import (
        farthest_centers,
        virtual_hamming_distance,
        carver,
        general_refine_method,
        iterative_clustering,
        FlexNode
        )
from ising import (
        IMul
        )
import pickle

def main():
    circuit = IMul(N1 = 2, N2 = 2)
    find_centers_method = farthest_centers
    distance = virtual_hamming_distance(circuit)
    refine_criterion = carver(circuit)
    refine_method = general_refine_method(distance, find_centers_method)

    clustering = iterative_clustering(refine_criterion, refine_method)

    root = FlexNode(set(circuit.inspace))
    tree = clustering(root)

    result_clusters = [leaf.value for leaf in tree.leaves]
    for leaf in result_clusters:
        print(leaf)

    print(len(tree.leaves))

    with open('clusters.dat', 'wb') as FILE:
        pickle.dump(result_clusters, FILE)

if __name__ == '__main__':
    main()
