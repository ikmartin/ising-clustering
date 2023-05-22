from ising import IMul
from spinspace import Spin
from functional_clustering import (
    outlier_centers,
    carver,
    general_refine_method,
    iterative_clustering,
    FlexNode,
)
from itertools import combinations as comb, pairwise
from itertools import permutations as perm
from more_itertools import set_partitions as partitions
import numpy as np

from checkcluster import check_all_aux as auxcheck


def pn_lvec_dist(circuit):
    """Returns the distance function which itself returns the norm of `poslvec(s) - neglvec(t)`."""

    def dist(s, t):
        return np.linalg.norm(circuit.poslvec(s) - circuit.neglvec(t)) ** 2

    return dist


def pp_lvec_dist(circuit):
    """Returns the distance function which itself returns the norm of `poslvec(s) - neglvec(t)`."""

    def dist(s, t):
        return np.linalg.norm(circuit.poslvec(s) - circuit.poslvec(t)) ** 2

    return dist


def pn_minus_pp_lvec_dist(circuit):
    """Returns the distance function which itself returns the norm of `poslvec(s) - neglvec(t)`."""
    dist1, dist2 = pn_lvec_dist(circuit), pp_lvec_dist(circuit)

    def dist(s, t):
        return dist1(s, t) - dist2(s, t)

    return dist


def pairwise_dict(circuit, func):
    """Generates a dictionary keyed by pairs (i,j) for every pair of distinct input spins i and j. Not necessarily symmetric in (i,j) permutations."""

    return {(i, j): func(i, j) for i, j in perm(circuit.inspace.tospinlist(), r=2)}


def refine_to_minimize_interaction_all(d):
    """Takes as input a cluster and a dictionary d keyed by all pairs of data values. Attempts to find C_+ and C_- such that the average _interaction_ d[i,j] between point i in C_+ and j in C_- is minimized. That is, it minimizes

    sum_{i in C_+} sum_{j in C_-} d(i,j).

    Over ALL POSSIBLE CHOICES of C1 and C2
    """

    def refine_method(cluster):
        print(f"refining {cluster}")
        print(
            f"minimization occurs over {2**(len(cluster)-1) - 1} different partitions. ALL will be checked."
        )

        # function which scores the clustering
        def score(C1, C2):
            return sum([d[(i, j)] for i in C1 for j in C2]) / (
                (len(C1) * len(C2)) ** 1.1
            )

        # dictionary of the scores
        scores = {tuple(C1): score(C1, C2) for C1, C2 in partitions(cluster, k=2)}

        # get the key which minimizes the score as a set, retrieve the second part of the clustering
        C1 = set(max(scores, key=scores.get))
        C2 = set(cluster.difference(C1))

        return C1, C2

    return refine_method


def refine_to_minimize_interaction_nearest_neighbor(d, initialize, maxsteps=1000):
    """Takes as input a cluster and a dictionary d keyed by all pairs of data values. Attempts to find C_+ and C_- such that the average _interaction_ d[i,j] between point i in C_+ and j in C_- is minimized. That is, it attempts to minimize

    sum_{i in C_+} sum_{j in C_-} d(i,j).

    Steps
    """

    def refine_method(cluster):
        print(f"refining {cluster}")

        # function which scores the clustering
        def score(C1, C2):
            return sum([d[(i, j)] for i in C1 for j in C2]) / (
                (len(C1) * len(C2)) ** 1.5
            )

        C1, C2 = initialize(d, cluster, score)

        steps = 0
        lowscore = score(C1, C2)
        changed = True
        print(f"initial score {lowscore}")
        while steps < maxsteps and changed:
            changed = False
            for spin in C1:
                C1new = C1.difference({spin})
                C2new = C2 | {spin}
                current_score = score(C1new, C2new)
                if current_score < lowscore:
                    print(f"Swapped {spin} into C2")
                    changed = True
                    lowscore = current_score
                    C1, C2 = C1new, C2new

            for spin in C2:
                C2new = C2.difference({spin})
                C1new = C1 | {spin}
                current_score = score(C1new, C2new)
                if current_score < lowscore:
                    print(f"Swapped {spin} into C1")
                    changed = True
                    lowscore = current_score
                    C1, C2 = C1new, C2new

            steps += 1

        print(f"final score {lowscore}")

        return C1, C2

    return refine_method


def refine_to_minimize_interaction_random_samples(d, samples=1000):
    """Takes as input a cluster and a dictionary d keyed by all pairs of data values. Attempts to find C_+ and C_- such that the average _interaction_ d[i,j] between point i in C_+ and j in C_- is minimized. That is, it minimizes

    sum_{i in C_+} sum_{j in C_-} d(i,j).

    Over ALL POSSIBLE CHOICES of C1 and C2
    """

    import random

    def refine_method(cluster):
        print(f"refining {cluster}")
        print(
            f"minimization occurs over {2**(len(cluster)-1) - 1} different partitions. ALL will be checked."
        )

        # function which scores the clustering
        def score(C1, C2):
            return sum([d[(i, j)] for i in C1 for j in C2]) / (
                (len(C1) * len(C2)) ** 1.1
            )

        # dictionary of the scores
        scores = {
            tuple(C1): score(C1, C2)
            for C1, C2 in random.sample(list(partitions(cluster, k=2)), samples)
        }

        # get the key which minimizes the score as a set, retrieve the second part of the clustering
        C1 = set(max(scores, key=scores.get))
        C2 = set(cluster.difference(C1))

        return C1, C2

    return refine_method


def init1(d, cluster, score):
    # minimizing spins
    newdict = {key: d[key] for key in d if key[0] in cluster and key[1] in cluster}
    key = min(newdict, key=newdict.get)
    C1 = {key[0]}
    C2 = {key[1]}

    print(f"clusters initialized with centers {C1} and {C2}")

    remaining = set(cluster).difference(C1 | C2)

    for spin in remaining:
        if score(C1 | {spin}, C2) < score(C1, C2 | {spin}):
            C1.add(spin)

        else:
            C2.add(spin)

    return C1, C2


def likelihood_clustering(circuit):
    """A clustering algo which replicates a variation of "farthest first". Each input level defines some number of constraints, whose solution space is the intersection of a bunch of hyperplanes. The larger the intersection between the solution spaces of two input levels, the more likely a random choice of hJ satisfies both input levels simultaneously.

    The matrix 'symmatrix' is a symmetric matrix whose (ij)th entry is the measure of the surface of the sphere S^d contained in the intersection between the solution spaces of input levels i and j, normalized so that the measure of S^d is 1. That is, the (ij)th entry is the probability that a random choice of hJ satisfies both input level i and j.

    This first chooses a center which is least likely to be satisfied simultaneously with another spin. It then chooses the second center to be the spin which has least intersection with the second center.
    """
    # setup

    dist1 = pn_lvec_dist(circuit)
    dist2 = pp_lvec_dist(circuit)
    dist3 = pn_minus_pp_lvec_dist(circuit)
    L = pairwise_dict(circuit, dist1)
    L2 = pairwise_dict(circuit, dist2)
    L3 = pairwise_dict(circuit, dist3)

    for key in comb(circuit.inspace.tospinlist(), r=2):
        print(key, L[key], L2[key], L3[key])

    refine_criterion = carver(circuit)
    refine_method = refine_to_minimize_interaction_random_samples(d=L)

    clustering = iterative_clustering(refine_criterion, refine_method)

    root = FlexNode(set(circuit.inspace))
    tree = clustering(root)

    result_clusters = [leaf.value for leaf in tree.leaves]
    return result_clusters


def main():
    import numpy as np

    circuit = IMul(2, 3)

    clusters = likelihood_clustering(circuit)
    for leaf in clusters:
        print(leaf)

    print(f"Checking clustering...")
    print(auxcheck(circuit, clusters))


if __name__ == "__main__":
    main()
