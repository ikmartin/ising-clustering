#!/usr/bin/env python

from binarytree import Node, NodeValue
from functools import cache
from itertools import combinations
from spinspace import Spinspace, Spin, vdist, qvec
from typing import Callable, Optional, Tuple, Set, Any
from ising import PICircuit, IMul
from scipy.optimize import linprog
import numpy as np
import heapq


class FlexNode(Node):
    """
    For some reason, the binarytree package prevents you from creating nodes with values which are not int, str, or float. This is probably because ordering of nodes is necessary for some binary tree algorithms, but those are not necessarily relevant here. This class is a wrapper for Node which overrides the type protection. We want this in order to store binary trees where the values are sets of spins (representing clusters).

    Be warned that it may not be compatible with some algorithms.
    """

    def __init__(
        self,
        value: NodeValue,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
    ) -> None:
        super().__init__(value, left, right)

    def __setattr__(self, attr: str, obj: Any) -> None:
        object.__setattr__(self, attr, obj)


def iterative_clustering(
    refine_criterion: Callable[[FlexNode], bool],
    refine_method: Callable[[FlexNode], tuple[FlexNode, FlexNode]],
) -> Callable:
    def run(node: FlexNode) -> FlexNode:
        if refine_criterion(node.value):
            cluster1, cluster2 = refine_method(node.value)
            node.left, node.right = run(FlexNode(cluster1)), run(FlexNode(cluster2))

        return node

    return run


## EXAMPLE FUNCTIONS: TEST FUNCTIONALITY OF CLUSTERING CLASS


def test_refine_criterion(data: Spinspace, indices: Tuple) -> bool:
    return len(indices) > 4


def test_refine_method(data: Spinspace, indices: Tuple) -> Tuple[Tuple, Tuple]:
    index = np.random.randint(len(indices) - 2) + 1
    return indices[:index], indices[index:]


def test_clustering():
    spinspace = Spinspace((4,))
    root = FlexNode(tuple(range(spinspace.size)))

    # Curry functions with the appropriate values
    operator = iterative_clustering(
        refine_criterion=lambda cluster: test_refine_criterion(spinspace, cluster),
        refine_method=lambda cluster: test_refine_method(spinspace, cluster),
    )

    tree = operator(root)
    print(tree)


## Re-implementations of some functions from topdown.py


def break_tie(val1, val2):
    return np.random.randint(2) == 0 if val1 == val2 else val1 > val2


def vector_refine_criterion(
    circuit: PICircuit, vector_method: Callable, weak=False
) -> Callable:
    """
    Currying function to produce a vector-based refinement function based on a certain vector-finding
    method passed by the user.
    """

    def criterion(cluster: Set[Spin]) -> bool:
        if len(cluster) < 2:
            return False

        vector = vector_method(cluster, circuit)
        return not circuit.levels(inspins=list(cluster), ham_vec=vector, weak=weak)

    return criterion


################################################
### General center-based refine method
################################################


def general_refine_method(
    distance: Callable, find_centers_method: Callable
) -> Callable:
    """
    Currying function to produce a refinement method with the specified methods.

    Takes a cluster, a notion of distance, and a method of finding centers for the child clusters.
    Breaks the cluster into two child clusters by picking new centers with the chosen method and
    grouping each element with the center that it is closest to. Random tie-breaking is used in the
    case of equal distances.
    """

    def method(cluster: Set[Spin]) -> Tuple[Set[Spin], Set[Spin]]:
        center1, center2 = find_centers_method(distance, cluster)
        child = set(
            filter(
                lambda index: break_tie(
                    distance(index, center1), distance(index, center2)
                ),
                cluster,
            )
        )
        return child, cluster.difference(child)

    return method


####################################
### Different distance notions
####################################


def hamming_distance(circuit: PICircuit) -> Callable:
    @cache
    def distance(spin1: Spin, spin2: Spin) -> int:
        return Spinspace.dist(circuit.inout(spin1), circuit.inout(spin2))

    return distance


def virtual_hamming_distance(circuit: PICircuit) -> Callable:
    @cache
    def distance(spin1: Spin, spin2: Spin) -> int:
        return vdist(circuit.inout(spin1), circuit.inout(spin2))

    return distance


def lvec_distance(circuit: PICircuit) -> Callable:
    @cache
    def distance(spin1: Spin, spin2: Spin) -> int:
        lvec1, lvec2 = circuit.normlvec(spin1), circuit.normlvec(spin2)
        return np.linalg.norm(lvec1 - lvec2)

    return distance


###########################################
### Center-finding methods
###########################################


def farthest_centers(distance: Callable, cluster: Set[Spin]) -> tuple[Spin, Spin]:
    """
    Takes a cluster and picks the two farthest apart elements, based on a user-defined distance metric.
    """
    pairwise_distances = {(i, j): distance(i, j) for i, j in combinations(cluster, 2)}
    return max(pairwise_distances, key=pairwise_distances.get)


def popular_centers(distance: Callable, cluster: Set[Spin]) -> tuple[Spin, Spin]:
    """
    Takes a cluster and picks the two elements which minimize the average distance from all other elements, based on a user-defined distance metric.
    """
    pairwise_distance_sums = {
        i: sum([distance(i, j) for j in cluster]) for i in cluster
    }
    twosmallest = heapq.nsmallest(
        2, pairwise_distance_sums, key=pairwise_distance_sums.get
    )

    print(twosmallest)

    return twosmallest


def outlier_centers(distance: Callable, cluster: Set[Spin]) -> tuple[Spin, Spin]:
    """
    Takes a cluster and picks the two elements which maximize the average distance from all other elements, based on a user-defined distance metric.
    """
    pairwise_distance_sums = {
        i: sum([distance(i, j) for j in cluster]) for i in cluster
    }
    twolargest = heapq.nlargest(
        2, pairwise_distance_sums, key=pairwise_distance_sums.get
    )
    print(twolargest)
    return twolargest


def outlier_farthest_centers(
    distance: Callable, cluster: Set[Spin]
) -> tuple[Spin, Spin]:
    """
    Takes a cluster and picks the first center to be the element which maximizes the average distance from all other elements, based on a user-defined distance metric.

    It chooses the second center to be the element in the cluster furthest from the first center.
    """
    # find the first center
    pairwise_distance_sums = {
        i: sum([distance(i, j) for j in cluster]) for i in cluster
    }
    center1 = max(pairwise_distance_sums, key=pairwise_distance_sums.get)

    # find the second center
    dist_from_center = {i: distance(center1, i) for i in cluster if i != center1}
    center2 = max(dist_from_center, key=dist_from_center.get)

    print(center1, center2)

    return (center1, center2)


##############################################################
## Various ways to guess the right interaction strength vector
##############################################################


def get_avglvec(cluster: Set[Spin], circuit: PICircuit):
    return sum(circuit.normlvec(s) for s in cluster)


def get_qvec(cluster: Set[Spin], circuit: PICircuit):
    return qvec([circuit.inout(s) for s in cluster])


def hebbian_stored_memory(pattern: Spin) -> np.ndarray:
    """
    Helper function for the Hebbian learning rule: creates the proper (h, J) interaction
    vector for the stored memory of a single pattern.
    """

    zero_bias = np.zeros(pattern.dim(), dtype=np.int8)
    outer_product = pattern.pairspin().spin()
    return np.concatenate([zero_bias, -outer_product])


def hebbian(cluster: Set[Spin], circuit: PICircuit) -> np.ndarray:
    """
    Attempts to create an interaction vector using the Hebbian learning rule.
    """

    return sum([hebbian_stored_memory(circuit.inout(s)) for s in cluster])


def all_incorrect_rows(s: Spin, circuit: PICircuit) -> list[np.ndarray]:
    """
    This method is simply used as a helper to generate the M matrix in the
    Carver refinement criterion. Given an input spin and a circuit model, it
    will give a list of virtual differences v(a) - v(b) (which are rows in the
    M matrix) for all wrong answers to the circuit given the input value s.
    These rows are then concatenated together to obtain the full M matrix.
    """

    correct_answer = circuit.f(s)
    correct_vspin = circuit.inout(s).vspin().spin()
    rows = [
        correct_vspin - Spin.catspin(spins=(s, y)).vspin().spin()
        if y.asint() != correct_answer.asint()
        else None
        for y in circuit.outspace
    ]
    return list(filter(lambda x: x is not None, rows))


##############################################
### Carver's criterion
##############################################


def carver(circuit: PICircuit) -> Callable:
    """
    Refinement criterion based on Carver's criterion for the solvability
    of a system of strict inequalities.

    Note that since H(a) - H(b) = < v(a) - v(b), x > where x is the
    interaction vector [h, J], we can write the system of inequalities
    defining the feasibility of the cluster as Mx < 0 where the rows of M
    are v(a) - v(b) for each correct pattern a and each incorrect
    variation b. By Carver's criterion, this system is feasible if and only
    if y=0 is the only solution to y >= 0, M^T y = 0. This alternative
    LP is useful because it is very easy to detect insolvability. We will
    use the scipy LP solver to get an approximate answer.
    """

    def criterion(cluster: Set[Spin]) -> bool:
        M = np.array(sum([all_incorrect_rows(s, circuit) for s in cluster], start=[]))
        approx_solution = linprog(
            -np.ones(M.shape[0]),
            A_eq=M.T,
            b_eq=np.zeros(M.shape[1]),
            bounds=(0, 1),
            # options = {'maxiter': 30000,}
        )
        return np.any(approx_solution.x > 0)

    return criterion


def lpsolver_criterion(circuit: PICircuit) -> Callable:
    """Uses the ortools lp solver built in PICircuit as a refine criterion."""

    print("Assigned lpsolver_criterion")

    def criterion(cluster: Set[Spin]) -> bool:
        solver = circuit.build_solver(input_spins=cluster)
        print(f"solver built with {solver.NumConstraints()} constraints.")
        status = solver.Solve()
        print(f"refine? {status != solver.OPTIMAL}")
        return status != solver.OPTIMAL

    return criterion
