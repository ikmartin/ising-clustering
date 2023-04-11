import numpy as np
from ising import IMul, PICircuit
from enum import Enum
from spinspace import Spin
import spinspace as ss
from algorithm_u import algorithm_u as algy  # this returns all k-partitions of a list


class HamVec(Enum):
    SGN = lambda spins: ss.sgn(spins)
    QVEC = lambda spins: ss.qvec(spins)

    def __call__(self, spins):
        return self.value(spins)


def save_clusters(G: PICircuit, k: int, file_name: str, ham_vec: HamVec = HamVec.SGN):
    """
    Saves all viable k-clusterings using the refine criterion specified.
    """
    file = open(file_name, "w")
    viable_clusters = []
    counter = 0
    all_partitions = algy(G.inspace.tolist(), k)
    total = sum(1 for dummy in algy(G.inspace.tolist(), k))
    for partition in all_partitions:
        satisfied = True
        for part in partition:
            inspins = [G.inspace.getspin(i) for i in part]
            satisfied = G.levels(inspins, ham_vec=ham_vec(G.inout(inspins)))

            if satisfied == False:
                break

        counter += 1
        if counter % 500 == 0:
            print(counter, " of ", total)

        if satisfied:
            print(partition, " is viable")
            file.write(str(partition))

    file.close()


if __name__ == "__main__":
    N1 = 2
    N2 = 2
    G = IMul(N1, N2)
    k = 3
    ham_vec = HamVec.SGN
    vecname = "sgn" if ham_vec == HamVec.SGN else "qvec"
    filename = f"all_viable_{vecname}_{k}-clusters_IMul{N1}x{N2}.dat"
    save_clusters(G, k, file_name=filename, ham_vec=ham_vec)
