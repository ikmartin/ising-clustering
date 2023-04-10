import numpy as np
import ising
import spinspace as ss
from clustering import Model as ClustModel
from clustering import Cluster, RefinedCluster, RefinedClustering
from ising import PICircuit, IMul


class IsingCluster(RefinedCluster):
    def __init__(self, id_num: int, indices, parent: Cluster, center):
        super().__init__(indices=indices, id_num=id_num, parent=parent)
        self.center = center


class TopDownRandInit(RefinedClustering):
    def __init__(self, circuit: PICircuit):
        super().__init__(data=circuit.inspace, size=circuit.inspace.size)
        self.circuit = circuit

    def refine_criterion(self, cluster: Cluster):
        refine = False

        # get the spins in the cluster
        inspins = [self.data[i] for i in cluster.indices]
        spins = [self.circuit.inout(s) for s in inspins]
        qvec = ss.qvec(spins)

        # check if qvec satisfies all input levels
        for s in inspins:
            if self.circuit.level(inspin=s, ham_vec=qvec) == False:
                refine = True

        return refine

    def refine(
        self, cluster: Cluster, current_id
    ) -> tuple[RefinedCluster, RefinedCluster]:
        """
        Splits a cluster
        """
        import random

        indices = cluster.indices
        data = self.data
        # initialize new centers
        i1, i2 = random.sample(indices, 2)
        s1, s2 = self.data[i1], self.data[i2]
        # create containers for new clusters
        bin1, bin2 = [i1], [i2]

        for i in cluster.indices:
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

        clust1 = RefinedCluster(id_num=current_id + 1, indices=bin1, parent=cluster)
        clust2 = RefinedCluster(id_num=current_id + 2, indices=bin2, parent=cluster)

        return clust1, clust2


class TopDownFarthestPair(RefinedClustering):
    def __init__(self, circuit: PICircuit):
        super().__init__(data=circuit.inspace, size=circuit.inspace.size)
        self.circuit = circuit

    def refine_criterion(self, cluster: Cluster):
        refine = False

        # get the spins in the cluster
        inspins = [self.data[i] for i in cluster.indices]
        spins = [self.circuit.inout(s) for s in inspins]
        qvec = ss.qvec(spins)

        # check if qvec satisfies all input levels
        for s in inspins:
            if self.circuit.level(inspin=s, ham_vec=qvec) == False:
                refine = True

        return refine

    def refine(
        self, cluster: Cluster, current_id
    ) -> tuple[RefinedCluster, RefinedCluster]:
        """
        Splits a cluster
        """
        import random

        indices = cluster.indices
        data = self.data

        # compute pairwise distances
        dist = {}
        for l1 in range(len(indices)):
            for l2 in range(len(indices)):
                i, j = indices[l1], indices[l2]
                spin1 = self.circuit.inout(data[i])
                spin2 = self.circuit.inout(data[j])
                dist[(i, j)] = self.circuit.spinspace.vdist(spin1, spin2)
                print(f"{i}, {j}:", dist[(i, j)])

        # get maximum distance key
        i1, i2 = max(dist, key=dist.get)

        # get new centers and create containers for new clusters
        s1, s2 = self.data[i1], self.data[i2]
        bin1, bin2 = [i1], [i2]

        # refine the cluster
        for i in cluster.indices:
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

        clust1 = IsingCluster(
            id_num=current_id + 1, indices=bin1, parent=cluster, center=i1
        )
        clust2 = IsingCluster(
            id_num=current_id + 2, indices=bin2, parent=cluster, center=i2
        )

        return clust1, clust2


class TopDown(ClustModel):
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
    model = TopDownRandInit(circuit=circuit)
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
    model = TopDownFarthestPair(circuit=circuit)
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


if __name__ == "__main__":
    example3()
