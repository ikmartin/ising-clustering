import numpy as np
import ising
import spinspace as ss
from clustering import Model as ClustModel
from clustering import RefinedCluster, RefinedClustering
from ising import PICircuit, IMul


class TopDown(ClustModel):
    def __init__(self, circuit: PICircuit):
        super().__init__(data=circuit.inspace)
        self.circuit = circuit

    def refine_criterion(self, cluster: RefinedCluster):
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
        clusterings = [[RefinedCluster(id_num=0, indices=allindices)]]
        current_count = 1

        def split(indices) -> tuple[list[int], list[int]]:
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


if __name__ == "__main__":
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
