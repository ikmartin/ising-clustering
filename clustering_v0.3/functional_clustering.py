from binarytree import Node, NodeValue
from functools import cache
from itertools import combinations
from spinspace import Spinspace, Spin, vdist, qvec
from typing import Callable, Optional, Tuple, Set, Any
from ising import PICircuit, IMul
import numpy as np

class FlexNode(Node):
    def __init__(self, value: NodeValue, left: Optional["Node"] = None, right: Optional["Node"] = None,) -> None:
        super().__init__(value, left, right)

    def __setattr__(self, attr: str, obj: Any) -> None:
        object.__setattr__(self, attr, obj)

class HierarchicalClustering:
    def __init__(self, distance_metric: Callable, refine_criterion: Callable, refine_method: Callable) -> None:
        self.distance_metric = distance_metric
        self.refine_criterion = refine_criterion
        self.refine_method = refine_method

    def _build_tree(self, node: FlexNode) -> FlexNode:
        if self.refine_criterion(self._data, self.distance_metric, node.value):
            cluster1, cluster2 = self.refine_method(self._data, self.distance_metric, node.value)
            node.left, node.right = self._build_tree(FlexNode(cluster1)), self._build_tree(FlexNode(cluster2))

        return node

    def __call__(self, data: Spinspace) -> FlexNode:
        self._data = data
        return self._build_tree(FlexNode(tuple(range(data.size))))
    
class HierarchicalClustering2:
    def __init__(self, refine_criterion: Callable, refine_method: Callable) -> None:
        self.refine_criterion = refine_criterion
        self.refine_method = refine_method

    def __call__(self, node: FlexNode) -> FlexNode:
        if self.refine_criterion(node.value):
            cluster1, cluster2 = self.refine_method(node.value)
            node.left, node.right = self(FlexNode(cluster1)), self(FlexNode(cluster2))

        return node
 

## Functions used to test the clustering class

def test_refine_criterion(data: Spinspace, indices: Tuple) -> bool:
    return len(indices) > 4

def test_refine_method(data: Spinspace, indices: Tuple) -> Tuple[Tuple, Tuple]:
    index = np.random.randint(len(indices)-2)+1
    return indices[:index], indices[index:]

def test_clustering():
    spinspace = Spinspace((4,))
    root = FlexNode(tuple(range(spinspace.size)))

    # Curry functions with the appropriate values
    operator = HierarchicalClustering2(
        refine_criterion = lambda cluster: test_refine_criterion(spinspace, cluster),
        refine_method = lambda cluster: test_refine_method(spinspace, cluster)
        )
    
    tree = operator(root)
    print(tree)


## Re-implementations of some functions from topdown.py

def break_tie(val1, val2):
    return np.random.randint(2) == 0 if val1 == val2 else val1 > val2

@cache
def virtual_hamming_distance(spin1: int, spin2: int, circuit: PICircuit) -> int:
    return vdist(circuit.inout(circuit.inspace[spin1]), circuit.inout(circuit.inspace[spin2]))

def vector_refine_criterion(cluster: Set, circuit: PICircuit, vector, weak = False) -> bool:
    """
    Essentially just a wrapper for circuit.levels, this function decides whether or not to split
    a cluster based on a vector passed as an argument.
    """
    cluster_spins = [circuit.inspace[i] for i in cluster]
    return not circuit.levels(inspins = cluster_spins, ham_vec = vector, weak = weak)
    
def avglvec_refine_criterion(cluster: Set, circuit: PICircuit) -> bool:
    inspins = [circuit.inspace[i] for i in cluster]
    avg_lvec = sum(circuit.lvec(s) for s in inspins)
    return vector_refine_criterion(cluster, circuit, avg_lvec)

def qvec_refine_criterion(cluster: Set, circuit: PICircuit) -> bool:
    spins = [circuit.inout(circuit.inspace[i]) for i in cluster]
    return vector_refine_criterion(cluster, circuit, qvec(spins))

def general_refine_method(cluster: Set, distance: Callable, find_centers_method: Callable) -> Tuple[Set, Set]:
    """
    Takes a cluster, a notion of distance, and a method of finding centers for the child clusters.
    Breaks the cluster into two child clusters by picking new centers with the chosen method and
    grouping each element with the center that it is closest to. Random tie-breaking is used in the 
    case of equal distances.
    """
    center1, center2 = find_centers_method(distance, cluster)
    child = set(filter(lambda index: break_tie(distance(index, center1), distance(index, center2)), cluster))
    return child, cluster.difference(child)

def farthest_centers(distance: Callable, cluster: Set) -> tuple[int, int]:
    """
    Takes a cluster and picks the two farthest apart elements, based on a user-defined distance metric.
    """
    pairwise_distances = {(i, j) : distance(i,j) for i,j in combinations(cluster, 2)}
    return max(pairwise_distances, key=pairwise_distances.get)



def main():
    #test_clustering()
    
    # As an example, run an avglvec clustering on 2x2 multiplication:

    circuit = IMul(N1=2, N2=2)
    root = FlexNode(set(range(circuit.inspace.size)))

    clustering = HierarchicalClustering2(
        refine_criterion = lambda cluster: avglvec_refine_criterion(cluster, circuit),
        refine_method = lambda cluster: general_refine_method(
            cluster = cluster,
            distance = lambda i, j: virtual_hamming_distance(i, j, circuit),
            find_centers_method  = farthest_centers,
        )
    )

    tree = clustering(root)
    print(tree)


if __name__ == '__main__':
    main()