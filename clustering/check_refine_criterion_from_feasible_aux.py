import json
import pickle
import itertools
from functional_clustering import carver, vector_refine_criterion, get_avglvec
from ising import IMul
from spinspace import Spin


N1 = 2
N2 = 3
A = 1

filename = f"all_feasible_MUL_{N1}x{N2}x1.dat"


def get_clust_from_aux(aux):
    """Returns a clustering from the provided auxiliary array"""
    clusts = []
    for a in [-1, 1]:
        clusts.append(
            set([Spin(i, shape=(N1, N2)) for i, elem in enumerate(aux) if elem == a])
        )

    return clusts


def main_carver():
    # circuit setup
    N1, N2 = 2, 3
    circuit = IMul(N1, N2)
    refine_criterion = carver(circuit)

    failed_clusters = []
    # iterate through all clusters in file and check refine_criterion
    with open("data/" + filename) as file:
        tally = 0
        for line in file:
            aux = json.loads(line)[0]
            clustering = get_clust_from_aux(aux)
            for clust in clustering:
                check = refine_criterion(clust)
                if check:
                    failed_clusters.append(clustering)

                print(f"{tally}: {clust} gives {check}")
                tally += 1

    print("\n\nFailed Clusters\n------------------")
    print(failed_clusters)


def main_ortools():
    # circuit setup
    N1, N2 = 2, 3
    circuit = IMul(N1, N2)

    def ortools_refine_criterion(clust):
        solver = circuit.build_solver(input_spins=clust)
        status = solver.Solve()
        print(
            f"cluster length: {len(clust)}  num constraints: {solver.NumConstraints()}  expected: {len(clust)*(2**(5)-1)}"
        )
        return status != solver.OPTIMAL

    failed_clusters = []
    # iterate through all clusters in file and check refine_criterion
    with open("data/" + filename) as file:
        tally = 0
        for line in file:
            aux = json.loads(line)[0]
            clustering = get_clust_from_aux(aux)
            for clust in clustering:
                check = ortools_refine_criterion(clust)
                if check:
                    failed_clusters.append(clustering)

                print(f"{tally}: {clust} gives {check}")
                tally += 1

    print("\n\nFailed Clusters\n------------------")
    print(failed_clusters)


# very bad
def main_lvec():
    # circuit setup
    N1, N2 = 2, 3
    circuit = IMul(N1, N2)
    refine_criterion = vector_refine_criterion(
        circuit=circuit, vector_method=get_avglvec
    )

    failed_clusters = []
    # iterate through all clusters in file and check refine_criterion
    with open("data/" + filename) as file:
        tally = 0
        for line in file:
            aux = json.loads(line)[0]
            clustering = get_clust_from_aux(aux)
            for clust in clustering:
                check = refine_criterion(clust)
                if check:
                    failed_clusters.append(clustering)

                print(f"{tally}: {clust} gives {check}")
                tally += 1

    print("\n\nFailed Clusters\n------------------")
    print(failed_clusters)


if __name__ == "__main__":
    main_lvec()
