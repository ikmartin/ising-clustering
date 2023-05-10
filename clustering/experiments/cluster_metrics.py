import json
import numpy as np
from itertools import combinations
from functional_clustering import carver, vector_refine_criterion, get_avglvec
from ising import IMul
from spinspace import Spin


N1 = 2
N2 = 3
A = 1

filename = f"all_feasible_MUL_{N1}x{N2}x1.dat"


###########################################
### Filehandlers
###########################################
def get_clust_from_aux(aux):
    """Returns a clustering from the provided auxiliary array"""
    clusts = []
    for a in [-1, 1]:
        clusts.append(
            set([Spin(i, shape=(N1, N2)) for i, elem in enumerate(aux) if elem == a])
        )

    return clusts


def print_metrics_for_file(filename, circuit, func):
    """Iterates through a file of feasible auxiliary arrays and prints the provided metrics for each cluster given by a feasible auxiliary"""
    with open(filename) as file:
        tally = 0
        for line in file:
            aux = json.loads(line)[0]
            clustering = get_clust_from_aux(aux)
            for i, clust in enumerate(clustering):
                print(f"  cluster {i}:", func(circuit, clust))


###########################################
### Metrics
###########################################


def satisfies_ortools(circuit, clust):
    """Checks if cluster is solvable by ortools LP"""
    # circuit setup

    solver = circuit.build_solver(input_spins=clust)
    status = solver.Solve()
    print(
        f"cluster length: {len(clust)}  num constraints: {solver.NumConstraints()}  expected: {len(clust)*(2**(5)-1)}"
    )
    return status != solver.OPTIMAL


def norm_of_lvec_sum(circuit, clust):
    """Returns the norm of the sum of lvecs.

    Hope: Bigger norm => better clustering

    Conclusion: Doesn't appear to give meaningful information
    """
    return np.linalg.norm(sum([circuit.normlvec(s) for s in clust])) / len(clust)


def sum_of_likelihood(circuit, clust):
    A = np.loadtxt(
        fname="/home/ikmarti/Desktop/ising-clustering/clustering/data/input_level_intersection_IMul2x3x0_rep=1000000_deg=2.csv",
        delimiter=",",
    )
    return sum([A[s.asint(), t.asint()] for (s, t) in combinations(clust, r=2)])


def compare_lvec_clustering_to_clustering_from_feasible(circuit, filename, name, func):
    from clustering_algos import lvec_cluster_popular

    # lvec clustering
    data = lvec_cluster_popular(circuit)
    print(f"\n{key}\n---------------\n")
    print("lvec clustering")
    for i, clust in enumerate(data):
        print(f"  cluster {i}:", func(circuit, clust))

    # feasible clusterings
    print("feasible clusterings")
    print_metrics_for_file(filename=filename, circuit=circuit, func=func)


if __name__ == "__main__":
    # setup the circuit
    circuit = IMul(2, 3)

    # dictionary of the metrics
    metrics = {
        # "Norm of lvec sum": norm_of_lvec_sum,
        "Sum of likelihoods": sum_of_likelihood,
    }
    for key, func in metrics.items():
        compare_lvec_clustering_to_clustering_from_feasible(
            circuit,
            str(input("Enter a path to a feasible array file: ")),
            name=key,
            func=func,
        )
