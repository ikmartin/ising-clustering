import numpy as np
import spinspace as ss
import random as rand
from abc import ABC, abstractmethod
from typing import Callable, Optional
from clustering import (
    Cluster,
    RefinedCluster,
    Model,
    RefinementClustering,
)
from ising import PICircuit, IMul


class IsingCluster(RefinedCluster):
    """An implementation of Refined Cluster used specifically for clustering Ising Circuits.

    Notes
    -----

    Atributes
    ---------
    self.indices : list[int]
        The indices of the data in the cluster. We store only the indices, not references to the data itself.
    self.id_num : int
        An identification number for this cluster. Is the hash of the indices.
    self.parent : Cluster
        A reference to the parent of this cluster.
    self.vec : numpy.ndarray
        The vector in real virtual space associated to this cluster. Likely either a qvec or the sign of a qvec. Depends on the implementation of TopDown. Must be of dimension equal to the dimension of virtual spin space.
    self.diameter : int
        The maximum distance between points in this cluster.
    self.satisfied : bool
        True if the center satisfies all input levels in the cluster, false otherwise.
    """

    def __init__(
        self,
        indices,
        circuit,
        center,
        generation: int,
        parent: Optional[Cluster] = None,
    ):
        super().__init__(indices=indices, parent=parent, generation=generation)
        self.circuit = circuit
        self.center = center
        self.diameter = ss.diameter(circuit.graph)

        # these are set on check_satisfied
        self.satisfied = None
        self.vec = None

    def check_satisfied(self, vec, weak=False) -> bool:
        self.vec = vec
        a = self.circuit.levels(
            inspins=[self.circuit.inspace.getspin(i) for i in self.indices],
            ham_vec=self.vec,
            weak=weak,
        )
        b = self.circuit.levels(
            inspins=[self.circuit.inspace.getspin(i) for i in self.indices],
            ham_vec=self.vec,
            weak=weak,
        )
        if a != b:
            print(f"Weird sanity check:{a} and {b} should be equal")
            print(f"Something bizarre happening")
            print(f"Indices: {self.indices} is type {type(self.indices)}")
            print(f"vec: {vec}")
            print(f"weak: {weak}")
            print(
                f"Now we run these side by side again and see if result has changed..."
            )

            checks = []
            for i in range(10):
                checks.append(
                    self.circuit.levels(
                        inspins=[self.circuit.inspace.getspin(i) for i in self.indices],
                        ham_vec=self.vec,
                        weak=weak,
                    )
                )
            print(f"  Check #2: the following should be equal:\n\n      {checks}")
            print(
                f"THIS IS LIKELY A PROBLEM WITH THE CUSTOM __iter__ IMPLMENTATION IN SPINSPACE"
            )

        self.satisfied = self.circuit.levels(
            inspins=[self.circuit.inspace.getspin(i) for i in self.indices],
            ham_vec=self.vec,
            weak=weak,
        )
        return self.satisfied


class TopDownBugged(RefinementClustering):
    """Class implementing the topdown refine algorithm. Must inherit from this and implement both refine_criterion and new_centers in order to complete model."""

    def __init__(self, circuit: PICircuit, weak=False):
        super().__init__(data=circuit.inspace, size=circuit.inspace.size)
        self.weak = weak
        self.circuit = circuit
        self._dist = {}

    def dist(self, i1: int, i2: int):
        try:
            return self._dist[i2, i2]
        except KeyError:
            self._dist[i1, i2] = ss.vdist(
                self.circuit.inout(self.data[i1]), self.circuit.inout(self.data[i2])
            )
            return self._dist[i1, i2]

    def new_centers(self, cluster: Cluster) -> tuple[int, int]:
        import random

        try:
            return tuple(random.sample(cluster.indices, 2))
        except ValueError:
            print(f"trying to refine {cluster.indices}")
            if len(cluster.indices) == 1:
                print(
                    "BE WARNED: TRYING TO REFINE A CLUSTER WITH ONLY ONE ELEMENT, GOING TO FAIL"
                )
            return tuple(random.sample(cluster.indices, 2))

    @abstractmethod
    def refine_criterion(self, cluster: IsingCluster) -> bool:
        pass

    def initialize(self):
        indices = list(range(self.size))  # indices of all data points
        self.generations = [
            [
                IsingCluster(
                    indices=indices,
                    circuit=self.circuit,
                    center=rand.choice(indices),
                    generation=0,
                    parent=None,
                )
            ]
        ]

    def refine(self, cluster: IsingCluster) -> tuple[IsingCluster, IsingCluster]:
        """
        Splits a cluster, randomly breaks ties
        """
        import random

        # get maximum distance key
        i1, i2 = self.new_centers(cluster=cluster)

        # get new centers and create containers for new clusters
        s1, s2 = self.data[i1], self.data[i2]
        bin1, bin2 = [i1], [i2]

        # refine the cluster
        for i in cluster.indices:
            # skip over the chosen centers
            if i == i1 or i == i2:
                continue

            d1 = ss.vdist(self.data[i], s1)
            d2 = ss.vdist(self.data[i], s2)
            # place index in bin2
            if d1 > d2:
                bin2.append(i)
            elif d1 < d2:
                bin1.append(i)
            else:
                bin1.append(i) if bool(random.getrandbits(1)) else bin2.append(i)

        clust1 = IsingCluster(
            indices=bin1,
            parent=cluster,
            circuit=self.circuit,
            center=i1,
            generation=self.gen_num,
        )
        clust2 = IsingCluster(
            indices=bin2,
            parent=cluster,
            circuit=self.circuit,
            center=i2,
            generation=self.gen_num,
        )

        return clust1, clust2


############################################
### Various refine criterion
############################################


def refine_with_lvec(model: TopDownBugged, cluster: IsingCluster, weak=False):
    """Refine the cluster only if the vector of the cluster doesn't satisfy one of the input levels in the cluster"""
    refine = False

    # avoid computing this again if the value already set
    # if cluster.satisfied != None:
    # return not cluster.satisfied

    # get the spins in the cluster
    center_spin = model.data[cluster.center]
    lvec = model.circuit.lvec(center_spin)

    condition = cluster.check_satisfied(vec=lvec, weak=model.weak)
    if len(cluster.indices) == 1 and condition == False:
        print(f"{cluster.indices[0]} returned {condition}")
        print(f"center_spin is {center_spin.asint()}")
        print(f"cluster is {cluster.indices}")
    return not condition


def refine_with_avglvec(model: TopDownBugged, cluster: IsingCluster, weak=False):
    """Refine the cluster only if the vector of the cluster doesn't satisfy one of the input levels in the cluster"""
    refine = False

    # avoid computing this again if the value already set
    if cluster.satisfied != None:
        return not cluster.satisfied

    # get the spins in the cluster
    inspins = [model.data[i] for i in cluster.indices]
    avg_lvec = sum(model.circuit.lvec(s) for s in inspins)

    return not cluster.check_satisfied(vec=avg_lvec, weak=model.weak)


def refine_with_qvec(model: TopDownBugged, cluster: IsingCluster, weak=False):
    """Refine the cluster only if the qvector of the cluster doesn't satisfy one of the input levels in the cluster"""
    refine = False

    # avoid computing this again if the value already set
    if cluster.satisfied != None:
        return not cluster.satisfied

    # get the spins in the cluster
    inspins = [model.data[i] for i in cluster.indices]
    spins = [model.circuit.inout(s) for s in inspins]
    qvec = ss.qvec(spins)

    return not cluster.check_satisfied(vec=qvec, weak=model.weak)


def refine_with_sgn(model: TopDownBugged, cluster: IsingCluster, weak=False):
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
class TopDownBuggedQvec(TopDownBugged):
    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_qvec(model=self, cluster=cluster, weak=self.weak)


class TopDownBuggedSgn(TopDownBugged):
    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_sgn(model=self, cluster=cluster, weak=self.weak)


class TopDownRandLvec(TopDownBugged):
    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_lvec(model=self, cluster=cluster, weak=self.weak)


class TopDownRandAvgLvec(TopDownBugged):
    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_avglvec(model=self, cluster=cluster, weak=self.weak)


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


def loop_models(ModelType, ModelName, loops=100, weak=True):
    circuit = IMul(N1=2, N2=2)
    model = ModelType(circuit=circuit, weak=weak)
    clusters = model.model()

    best_model = model
    best_score = len(model.clusters)
    for i in range(loops):
        model = ModelType(circuit=circuit, weak=weak)
        model.model()
        score = len(model.clusters)
        print(f"Trial {i}: Score {score}")
        if best_score > score:
            best_model = model
            best_score = score

    print_cluster_results(best_model, ModelName)


if __name__ == "__main__":
    loop_models(
        ModelType=TopDownBuggedQvec, ModelName="TopDownBuggedQvec", loops=100, weak=True
    )
    loop_models(
        ModelType=TopDownBuggedSgn, ModelName="TopDownBuggedSgn", loops=100, weak=True
    )
