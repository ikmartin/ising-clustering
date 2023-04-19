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

    def check_satisfied(self, vec, weak=False) -> bool:
        self.vec = vec
        self.satisfied = self.circuit.levels(
            inspins=[self.circuit.inspace[i] for i in self.indices],
            ham_vec=vec,
            weak=weak,
        )
        return self.satisfied


class TopDown(RefinementClustering):
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
                    center=0,
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


class TopDownLvec(RefinementClustering):
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
            lvec1 = self.circuit.lvec(self.data[i1])
            lvec2 = self.circuit.lvec(self.data[i2])
            self._dist[i1, i2] = ss.hamming(lvec1, lvec2)
            return self._dist[i1, i2]

    @abstractmethod
    def new_centers(self, cluster: IsingCluster) -> tuple[int, int]:
        pass

    @abstractmethod
    def refine_criterion(self, cluster: IsingCluster, weak: bool = False) -> bool:
        pass

    def initialize(self):
        indices = list(range(self.size))  # indices of all data points
        self.generations = [
            [
                IsingCluster(
                    indices=indices,
                    circuit=self.circuit,
                    center=0,
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
    def __init__(self, circuit: PICircuit, weak=False):
        super().__init__(circuit=circuit, weak=weak)

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

        for i in ties:
            d1 = sum([self.dist(i, j) for j in bin1]) / len(bin1)
            d2 = sum([self.dist(i, j) for j in bin2]) / len(bin2)

            print(
                f"  spin {i} is tied between {i1} and {i2} for distance {self.dist(i,i1)} = {self.dist(i,i2)}"
            )
            print(f"    bin1 average distance: {d1}")
            print(f"    bin1 average distance: {d2}")
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


def refine_with_lvec(model: TopDownLvec, cluster: IsingCluster, weak=False):
    """Refine the cluster only if the vector of the cluster doesn't satisfy one of the input levels in the cluster"""
    refine = False

    # avoid computing this again if the value already set
    if cluster.satisfied != None:
        return not cluster.satisfied

    # get the spins in the cluster
    print(cluster.center)
    center_spin = model.data[cluster.center]
    lvec = model.circuit.lvec(center_spin)

    return not cluster.check_satisfied(vec=lvec, weak=model.weak)


def refine_with_avglvec(model: TopDownLvec, cluster: IsingCluster, weak=False):
    """Refine the cluster only if the vector of the cluster doesn't satisfy one of the input levels in the cluster"""
    refine = False

    # avoid computing this again if the value already set
    if cluster.satisfied != None:
        return not cluster.satisfied

    # get the spins in the cluster
    inspins = [model.data[i] for i in cluster.indices]
    avg_lvec = sum(model.circuit.lvec(s) for s in inspins)

    return not cluster.check_satisfied(vec=avg_lvec, weak=model.weak)


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

    return not cluster.check_satisfied(vec=qvec, weak=weak)


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

    return not cluster.check_satisfied(vec=sgn, weak=weak)


############################################
### Implementations of different models
############################################
class TopDownQvecRandInit(TopDown):
    def __init__(self, circuit: PICircuit):
        super().__init__(circuit=circuit)

    def refine_criterion(self, cluster: Cluster):
        return refine_with_qvec(model=self, cluster=cluster, weak=self.weak)

    def new_centers(self, cluster: Cluster) -> tuple[int, int]:
        import random

        return tuple(random.sample(cluster.indices, 2))


class TopDownSgnRandInit(TopDown):
    def __init__(self, circuit: PICircuit):
        super().__init__(circuit=circuit)

    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_sgn(model=self, cluster=cluster)

    def new_centers(self, cluster: IsingCluster) -> tuple[int, int]:
        import random

        return tuple(random.sample(cluster.indices, 2))


class TopDownLvecFarthestPair(TopDownLvec):
    def __init__(self, circuit: PICircuit, weak=False):
        super().__init__(circuit=circuit, weak=weak)

    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_avglvec(model=self, cluster=cluster, weak=self.weak)

    def new_centers(self, cluster: IsingCluster) -> tuple[int, int]:
        indices = cluster.indices

        # compute pairwise distances
        dist = {}
        for l1 in range(len(indices)):
            for l2 in range(l1 + 1, len(indices)):
                i, j = indices[l1], indices[l2]
                dist[(i, j)] = self.dist(i, j)
                # print(f"{i}, {j}:", dist[(i, j)])

        # get maximum distance key
        print(cluster.indices)
        i1, i2 = max(dist, key=dist.get)
        return i1, i2


class TopDownLvecMain(TopDownLvec):
    """
    The canonical Lvec clustering. The new_centers method is inefficient, the lvecs of centers are used as the refine_criterion.
    """

    def __init__(self, circuit: PICircuit, weak=False):
        super().__init__(circuit=circuit, weak=weak)

    def refine_criterion(self, cluster: IsingCluster):
        return refine_with_lvec(model=self, cluster=cluster, weak=self.weak)

    def new_centers(self, cluster: IsingCluster) -> tuple[int, int]:
        """
        Very inefficient. Chooses the two spins together whose lvecs satisfy the most spins.
        """
        indices = cluster.indices

        #
        lvec_counts = {}
        maxlength = 0
        argmax1 = 0
        argmax2 = 0
        for l1 in range(len(indices)):
            for l2 in range(len(indices)):
                s1 = self.data[cluster.indices[l1]]
                s2 = self.data[cluster.indices[l2]]
                lvec1 = self.circuit.lvec(s1)
                lvec2 = self.circuit.lvec(s2)
                lvec_counts[l1, l2] = set([])
                # add all spins which are satisfied by either lvec1 or lvec2
                for i in indices:
                    if self.circuit.level(self.data[i], ham_vec=lvec1, weak=True):
                        lvec_counts[l1, l2].add(i)
                    elif self.circuit.level(self.data[i], ham_vec=lvec2, weak=True):
                        lvec_counts[l1, l2].add(i)

                if maxlength < len(lvec_counts[l1, l2]):
                    maxlength = len(lvec_counts[l1, l2])
                    argmax1 = l1
                    argmax2 = l2

        # get maximum distance key
        return argmax1, argmax2


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


class TopDownSgnBreakTies(TopDownLvec):
    def __init__(self, circuit: PICircuit):
        super().__init__(circuit=circuit)

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


class TopDownOg(Model):
    def __init__(self, circuit: PICircuit):
        super().__init__(data=circuit.inspace, size=circuit.inspace.size)
        self.circuit = circuit

    def refine_criterion(self, cluster: Cluster):
        refine = False

        # convert cluster into an array of spins of shape
        inspins = [self.circuit.inspace.getspin(i) for i in cluster.indices]
        spins = [self.circuit.inout(s) for s in inspins]
        qvec = ss.qvec(spins)
        for s in inspins:
            if self.circuit.level(inspin=s, ham_vec=qvec) == False:
                refine = True

        return refine

    def model(self):
        data = self.circuit.generate_graph()
        allindices = list(range(len(data)))
        clusterings = [[Cluster(id_num=0, indices=allindices)]]
        current_count = 1

        def split(indices) -> tuple[list[int], list[int]]:
            """
            Splits a cluster
            """
            import random

            # initialize new centers
            i1, i2 = random.sample(indices, 2)
            s1, s2 = data[i1], data[i2]

            # create containers for new clusters
            bin1, bin2 = [i1], [i2]

            for i in indices:
                # skip over the chosen centers
                if i == i1 or i == i2:
                    continue

                d1 = ss.vdist(data[i], s1)
                d2 = ss.vdist(data[i], s2)
                # place index in bin2
                if d1 > d2:
                    bin2.append(i)
                elif d1 < d2:
                    bin1.append(i)
                else:
                    bin1.append(i) if bool(random.getrandbits(1)) else bin2.append(i)

            return bin1, bin2

        done = False
        while done == False:
            done = True
            # iterate through the most recent clustering
            newlayer = []
            for cluster in clusterings[-1]:
                # check if the refine criterion is met
                if self.refine_criterion(cluster):
                    # if this is the first refinement
                    if done == True:
                        # set done to False
                        done = False

                    # refine the cluster
                    bin1, bin2 = split(cluster.indices)
                    clust1 = RefinedCluster(
                        id_num=current_count, indices=bin1, parent=cluster
                    )
                    newlayer.append(clust1)
                    current_count += 1
                    clust2 = RefinedCluster(
                        id_num=current_count, indices=bin2, parent=cluster
                    )
                    newlayer.append(clust2)
                    current_count += 1

                # if we don't need to refine, preserve cluster
                else:
                    newlayer.append(cluster)

            if done == False:
                clusterings.append(newlayer)

        return clusterings


def example1():
    circuit = IMul(N1=2, N2=2)
    model = TopDown(circuit=circuit)
    clusterings = model.model()

    for clustering in clusterings:
        print(len(clustering))
        for cluster in clustering:
            print("  ", cluster.indices)
    print(f"Number of refinements: {len(clusterings)}")
    print(f"Final number of clusters: {len(clusterings[-1])}")

    for i, cluster in enumerate(clusterings[-1]):
        print(f"cluster {i}:", cluster.indices)


def example2():
    circuit = IMul(N1=2, N2=2)
    model = TopDownQvecRandInit(circuit=circuit)
    clusterings = model.model()

    for clustering in clusterings:
        print(len(clustering.clusters))
        for cluster in clustering:
            print("  ", cluster.indices)

    print(f"Number of refinements: {len(clusterings)}")
    print(f"Final number of clusters: {len(clusterings[-1].clusters)}")

    for i, cluster in enumerate(clusterings[-1]):
        print(f"cluster {i}:", cluster.indices)


def example3():
    circuit = IMul(N1=2, N2=2)
    model = TopDownQvecFarthestPair(circuit=circuit)
    clusterings = model.model()

    print(
        "RESULT OF TopDownFarthestPair on Mul2x2\n---------------------------------------"
    )
    for clustering in clusterings:
        print(len(clustering.clusters))
        for cluster in clustering:
            print("  ", cluster.indices, "  center:", cluster.center)

    print(f"Number of generations: {len(clusterings)}")
    print(f"Final number of clusters: {len(clusterings[-1].clusters)}")

    for i, cluster in enumerate(clusterings[-1]):
        print(f"cluster {i}:", cluster.indices)


def example4():
    success = "\u2713"
    failure = "x"
    circuit = IMul(N1=2, N2=2)
    model = TopDownSgnFarthestPair(circuit=circuit)
    clusters = model.model()
    print(
        "RESULT OF TopDownFarthestPair on Mul2x2\n---------------------------------------"
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


def example5():
    """NOTES
    This method of clustering breaks ties between clusters by attempting to minimize the average distance between points within clusters. It uses sgn as its ham_vec rather than qvec.

    It always terminates with 7 clusters, which is quite bad.
    """
    success = "\u2713"
    failure = "x"
    circuit = IMul(N1=2, N2=2)
    model = TopDownSgnBreakTies(circuit=circuit)
    clusters = model.model()
    print(
        "RESULT OF TopDownFarthestPair on Mul2x2\n---------------------------------------"
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


def example6():
    """NOTES
    This uses lvec as its refine_criterion
    """

    success = "\u2713"
    failure = "x"
    circuit = IMul(N1=2, N2=2)
    model = TopDownLvecFarthestPair(circuit=circuit, weak=True)
    clusters = model.model()
    print(
        "RESULT OF TopDownLvecFarthestPair on Mul2x2\n---------------------------------------"
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


def example7():
    """NOTES
    This uses lvec as its refine_criterion
    """

    success = "\u2713"
    failure = "x"
    circuit = IMul(N1=2, N2=2)
    model = TopDownLvecMain(circuit=circuit, weak=True)
    clusters = model.model()
    print(
        "RESULT OF TopDownLvecMain on Mul2x2\n---------------------------------------"
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
