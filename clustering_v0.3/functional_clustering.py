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
    def __init__(self, refine_criterion: Callable, refine_method: Callable) -> None:
        self.refine_criterion = refine_criterion
        self.refine_method = refine_method

    def __call__(self, node: FlexNode) -> FlexNode:
        if self.refine_criterion(node.value):
            cluster1, cluster2 = self.refine_method(node.value)
            node.left, node.right = self(FlexNode(cluster1)), self(FlexNode(cluster2))

        return node
 
## EXAMPLE FUNCTIONS: TEST FUNCTIONALITY OF CLUSTERING CLASS

def test_refine_criterion(data: Spinspace, indices: Tuple) -> bool:
    return len(indices) > 4

def test_refine_method(data: Spinspace, indices: Tuple) -> Tuple[Tuple, Tuple]:
    index = np.random.randint(len(indices)-2)+1
    return indices[:index], indices[index:]

def test_clustering():
    spinspace = Spinspace((4,))
    root = FlexNode(tuple(range(spinspace.size)))

    # Curry functions with the appropriate values
    operator = HierarchicalClustering(
        refine_criterion = lambda cluster: test_refine_criterion(spinspace, cluster),
        refine_method = lambda cluster: test_refine_method(spinspace, cluster)
        )
    
    tree = operator(root)
    print(tree)

## Re-implementations of some functions from topdown.py

def break_tie(val1, val2):
    return np.random.randint(2) == 0 if val1 == val2 else val1 > val2


def vector_refine_criterion(circuit: PICircuit, vector_method : Callable, weak = False) -> bool:
    """
    Currying function to produce a vector-based refinement function based on a certain vector-finding
    method passed by the user.
    """
    def criterion(cluster: Set[Spin]):
        vector = vector_method(cluster, circuit)
        return not circuit.levels(inspins = list(cluster), ham_vec = vector, weak = weak)
    
    return criterion

def general_refine_method(distance: Callable, find_centers_method: Callable) -> Tuple[Set[Spin], Set[Spin]]:
    """
    Currying function to produce a refinement method with the specified methods.

    Takes a cluster, a notion of distance, and a method of finding centers for the child clusters.
    Breaks the cluster into two child clusters by picking new centers with the chosen method and
    grouping each element with the center that it is closest to. Random tie-breaking is used in the 
    case of equal distances.
    """
    def method(cluster: Set[Spin]):
        center1, center2 = find_centers_method(distance, cluster)
        child = set(filter(lambda index: break_tie(distance(index, center1), distance(index, center2)), cluster))
        return child, cluster.difference(child)

    return method

def virtual_hamming_distance(circuit: PICircuit) -> int:
    @cache
    def distance(spin1: Spin, spin2: Spin):
        return vdist(circuit.inout(spin1), circuit.inout(spin2))
    
    return distance

def farthest_centers(distance: Callable, cluster: Set[Spin]) -> tuple[Spin, Spin]:
    """
    Takes a cluster and picks the two farthest apart elements, based on a user-defined distance metric.
    """
    pairwise_distances = {(i, j) : distance(i,j) for i,j in combinations(cluster, 2)}
    return max(pairwise_distances, key=pairwise_distances.get)

def get_avglvec(cluster: Set[Spin], circuit: PICircuit):
    return sum(circuit.lvec(s) for s in cluster)

def get_qvec(cluster: Set[Spin], circuit: PICircuit):
    return qvec([circuit.inout(s) for s in cluster])


def main():
    # As an example, run an qvec clustering on 2x2 multiplication:

    circuit = IMul(N1=2, N2=2)
    vector_method = get_avglvec
    find_centers_method = farthest_centers
    distance = virtual_hamming_distance(circuit)
    refine_criterion = vector_refine_criterion(circuit, vector_method)
    refine_method = general_refine_method(distance, find_centers_method)

    clustering = HierarchicalClustering(refine_criterion, refine_method)

    root = FlexNode(set(circuit.inspace))
    tree = clustering(root)
    print(tree)


if __name__ == '__main__':
    main()