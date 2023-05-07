import pickle

from functional_clustering import (
        farthest_centers,
        outlier_centers,
        virtual_hamming_distance,
        hamming_distance,
        carver,
        general_refine_method,
        iterative_clustering,
        FlexNode
        )
from ising import (
        IMul
        )
import pickle

def cluster_carver(N1, N2):
    circuit = IMul(N1, N2)
    find_centers_method = outlier_centers
    distance = virtual_hamming_distance(circuit)
    refine_criterion = carver(circuit)
    refine_method = general_refine_method(distance, find_centers_method)

    clustering = iterative_clustering(refine_criterion, refine_method)

    root = FlexNode(set(circuit.inspace))
    tree = clustering(root)

    result_clusters = [leaf.value for leaf in tree.leaves]
    return result_clusters

def main():
    N1,N2 = 2,2
    with open(f'data/hash_clusters_from_all_feasible_MUL_{N1}x{N2}x1.dat', 'rb') as file:
        hash_table = pickle.load(file)
        result_clusters = cluster_carver(N1,N2)

        clusts_as_tuple = []
        for leaf in result_clusters:
            clusts_as_tuple.append(tuple(leaf))
            print(leaf)

        clusts_as_tuple = tuple(clusts_as_tuple)
        print(len(result_clusters))
        print(f"Clustering has feasible aux array: {hash(clusts_as_tuple) in hash_table}")

        with open('clusters2.dat', 'wb') as FILE:
            pickle.dump(result_clusters, FILE)

if __name__ == '__main__':
    main()
