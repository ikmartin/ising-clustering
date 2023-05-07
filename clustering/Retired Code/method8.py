# Refining using level satisfaction first, then normal clustering second

import numpy as np
import spinspace as ss
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

    def check_satisfied(self, vec) -> bool:
        self.vec = vec
        self.satisfied = self.circuit.levels(
            inspins=[self.circuit.inspace[i] for i in self.indices], ham_vec=vec
        )
        return self.satisfied


class TopDown(RefinementClustering):
    """Class implementing the topdown refine algorithm. Must inherit from this and implement both refine_criterion and new_centers in order to complete model."""

    def __init__(self, circuit: PICircuit):
        super().__init__(data=circuit.inspace, size=circuit.inspace.size)
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

    @abstractmethod
    def new_centers(self, cluster: IsingCluster) -> tuple[int, int]:
        pass

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
                    center=None,
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

            d1 = self.dist(i, s1)
            d2 = self.dist(i, s2)
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


class TopDownBreakTies(TopDown):
    def __init__(self, circuit: PICircuit):
        super().__init__(circuit=circuit)

    def refine(self, cluster: IsingCluster) -> tuple[IsingCluster, IsingCluster]:
        """
        Splits a cluster
        """
        import random

        # get maximum distance key
        i1, i2 = self.new_centers(cluster=cluster)

        # get new centers and create containers for new clusters
        s1, s2 = self.data[i1], self.data[i2]
        bin1, bin2, ties = [i1], [i2], []

        # first pass through, find ties
        for i in cluster.indices:
            # skip over the chosen centers
            if i == i1 or i == i2:
                continue

            d1 = self.dist(i, i1)
            d2 = self.dist(i, i2)
            # place index in bin2
            if d1 > d2:
                bin2.append(i)
            elif d1 < d2:
                bin1.append(i)
            else:
                ties.append(i)

        print(f"Generation {self.gen_num}")
        print(f"bin1: {bin1}")
        print(f"bin2: {bin2}")
        print(f"ties {ties}")

        for i in ties:
            d1 = sum([self.dist(i, j) for j in bin1]) / len(bin1)
            d2 = sum([self.dist(i, j) for j in bin2]) / len(bin2)

            """
            print(
                f"  spin {i} is tied between {i1} and {i2} for distance {self.dist(i,i1)} = {self.dist(i,i2)}"
            )
            print(f"    bin1 average distance: {d1}")
            print(f"    bin1 average distance: {d2}")
            """
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


def refine_with_qvec(model: TopDown, cluster: IsingCluster):
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


def refine_with_sgn(model: TopDown, cluster: IsingCluster):
    """Refine the cluster only if the sgn of the qvector doesn't satisfy all input levels in the cluster"""
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


class MyModel(TopDownBreakTies):
    def __init__(self, circuit: PICircuit):
        super().__init__(circuit=circuit)

    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_qvec(model=self, cluster=cluster)

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


def example7():
    """NOTES
    This method of clustering breaks ties between clusters by attempting to minimize the average distance between points within clusters. It uses sgn as its ham_vec rather than qvec.

    It always terminates with 7 clusters, which is quite bad.
    """
    success = "\u2713"
    failure = "x"
    circuit = IMul(N1=2, N2=2)
    model = MyModel(circuit=circuit)
    clusters = model.model()
    print(
        "RESULT OF TopDownBreakTies on Mul2x2 Example 7\n--------------------------------------------------"
    )
    print(f"Number of generations: {model.gen_num}")
    print(f"Final number of clusters: {len(model.clusters)}")

    for i, cluster in enumerate(clusters):
        print(f" cluster {i}:", cluster.indices)

    print("Generation progression:\n----------------------")
    for gen in model.generations:
        print(" ", len(gen))
        for cluster in gen:
            print("   ", cluster.indices, "  center:", cluster.center, end="")
            print("   ", success if cluster.satisfied else failure)


if __name__ == "__main__":
    example7()
