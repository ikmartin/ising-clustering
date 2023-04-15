import numpy as np
from topdown import TopDown, TopDownBreakTies, IsingCluster
from ising import PICircuit, IMul
from clustering import Cluster, Model
from spinspace import Spin

############################################
### Various refine criterion
############################################


def refine_with_qvec(model: TopDown, cluster: IsingCluster, weak=False):
    """Refine the cluster only if the qvector of the cluster doesn't satisfy one of the input levels in the cluster"""
    refine = False

    # avoid computing this again if the value already set
    if cluster.satisfied != None:
        return not cluster.satisfied

    # get the spins in the cluster
    inspins = [model.data[i] for i in cluster.indices]
    spins = [model.circuit.inout(s) for s in inspins]
    qvec = ss.qvec(spins)

    return not cluster.check_satisfied(vec=qvec)


def refine_with_sgn(model: TopDown, cluster: IsingCluster, weak=False):
    """Refine the cluster only if the sgn of the vector doesn't satisfy all input levels in the cluster"""
    refine = False

    # avoid computing this again if the value already set
    if cluster.satisfied != None:
        return not cluster.satisfied

    # get the spins in the cluster
    inspins = [model.data[i] for i in cluster.indices]
    spins = [model.circuit.inout(s) for s in inspins]
    sgn = ss.sgn(spins)

    return not cluster.check_satisfied(vec=sgn)


############################################
### Implementations of different models
############################################
class TopDownQvecRandCenters(TopDown):
    def __init__(self, circuit: PICircuit, weak=False):
        super().__init__(circuit=circuit, weak=weak)

    def refine_criterion(self, cluster: Cluster):
        return refine_with_qvec(model=self, cluster=cluster)

    def new_centers(self, cluster: Cluster) -> tuple[int, int]:
        import random

        return tuple(random.sample(cluster.indices, 2))


class TopDownSgnRandInit(TopDown):
    def __init__(self, circuit: PICircuit, weak=False):
        super().__init__(circuit=circuit, weak=weak)

    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_sgn(model=self, cluster=cluster)

    def new_centers(self, cluster: IsingCluster) -> tuple[int, int]:
        import random

        return tuple(random.sample(cluster.indices, 2))


class TopDownQvecFarthestPair(TopDown):
    def __init__(self, circuit: PICircuit, weak=False):
        super().__init__(circuit=circuit, weak=weak)

    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_qvec(model=self, cluster=cluster, weak=self.weak)

    def new_centers(self, cluster: IsingCluster) -> tuple[int, int]:
        indices = cluster.indices

        # compute pairwise distances
        dist = {}
        for l1 in range(len(indices)):
            for l2 in range(l1 + 1, len(indices)):
                i, j = indices[l1], indices[l2]
                spin1 = self.circuit.inout(self.data[i])
                spin2 = self.circuit.inout(self.data[j])
                dist[(i, j)] = self.circuit.spinspace.vdist(spin1, spin2)

        # get maximum distance key
        i1, i2 = max(dist, key=dist.get)
        return i1, i2


class TopDownSgnFarthestPair(TopDown):
    def __init__(self, circuit: PICircuit, weak=False):
        super().__init__(circuit=circuit, weak=weak)

    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_sgn(model=self, cluster=cluster, weak=self.weak)

    def new_centers(self, cluster: IsingCluster) -> tuple[int, int]:
        indices = cluster.indices

        # compute pairwise distances
        dist = {}
        for l1 in range(len(indices)):
            for l2 in range(l1 + 1, len(indices)):
                i, j = indices[l1], indices[l2]
                spin1 = self.circuit.inout(self.data[i])
                spin2 = self.circuit.inout(self.data[j])
                dist[(i, j)] = self.circuit.spinspace.vdist(spin1, spin2)
                # print(f"{i}, {j}:", dist[(i, j)])

        # get maximum distance key
        i1, i2 = max(dist, key=dist.get)
        return i1, i2


class TopDownSgnBreakTies(TopDownBreakTies):
    def __init__(self, circuit: PICircuit, weak=False):
        super().__init__(circuit=circuit, weak=weak)

    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_sgn(model=self, cluster=cluster)

    def new_centers(self, cluster: IsingCluster) -> tuple[int, int]:
        indices = cluster.indices

        # compute pairwise distances inside cluster
        dist = {}
        for l1 in range(len(indices)):
            for l2 in range(l1 + 1, len(indices)):
                i, j = indices[l1], indices[l2]
                dist[i, j] = self.dist(i, j)

        # get maximum distance key
        i1, i2 = max(dist, key=dist.get)
        return i1, i2


##############################################
### Main Code
##############################################


def print_cluster_results(model, typename="NO NAME GIVEN"):
    # make sure the model has been run first
    if model.clusters == []:
        raise ValueError("No clusters in model! Did you run model.model()?")

    # the success and failure characters
    success = "\u2713"
    failure = "x"

    print(f"RESULT OF {typename} on Mul2x2\n---------------------------------------")
    print(f"Number of generations: {model.gen_num}")
    print(f"Final number of clusters: {len(model.clusters)}")

    for i, cluster in enumerate(model.clusters):
        print(f" cluster {i}:", cluster.indices)

    print("Generation progression:\n----------------------")
    for gen in model.generations:
        print(" ", len(gen))
        for cluster in gen:
            print("   ", cluster.indices, "  center:", cluster.center, end="")
            print("   ", success if cluster.satisfied else failure)


def run_clustering_wrapper(ModelType, aux_array=[]):
    """Tell this which model to use (TopDownQvecRandCenters, TopDownSgnFarthestPair, etc) and it will run the model and return the clusters"""

    circuit = IMul(2, 2)
    if aux_array != []:
        circuit.set_aux(aux_array)

    return ModelType(circuit=circuit, weak=True)


if __name__ == "__main__":
    # get an auxiliary array from all_feasible_MUL2x2x1.dat file
    aux_array = [[-1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1]]

    # setup the parameters for the model
    ModelType = TopDownQvecFarthestPair
    name = "TopDownQvecFarthestPair"

    # run the model, get the results. Optionally provide an auxiliary array
    model = run_clustering_wrapper(ModelType=ModelType, aux_array=aux_array)

    # print the results
    print_cluster_results(model, typename=name)

    # print the number of clusters
    print(f"\n\nThe clustering achieved has {len(model.clusters)} clusters")
