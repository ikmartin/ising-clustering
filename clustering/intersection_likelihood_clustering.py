from ising import IMul
from functional_clustering import (
    outlier_centers,
    carver,
    general_refine_method,
    iterative_clustering,
    FlexNode,
)


def most_likely_distance(circuit, symmatrix):
    """Spins with the largest intersections are far apart"""

    def distance(spin1, spin2):
        return symmatrix[spin1.asint(), spin2.asint()]

    return distance


def least_likely_distance(circuit, symmatrix):
    """Spins with the smallest intersections are far apart"""

    def distance(spin1, spin2):
        return 1 - symmatrix[spin1.asint(), spin2.asint()]

    return distance


def likelihood_clustering(circuit, symmatrix):
    """A clustering algo which replicates a variation of "farthest first". Each input level defines some number of constraints, whose solution space is the intersection of a bunch of hyperplanes. The larger the intersection between the solution spaces of two input levels, the more likely a random choice of hJ satisfies both input levels simultaneously.

    The matrix 'symmatrix' is a symmetric matrix whose (ij)th entry is the measure of the surface of the sphere S^d contained in the intersection between the solution spaces of input levels i and j, normalized so that the measure of S^d is 1. That is, the (ij)th entry is the probability that a random choice of hJ satisfies both input level i and j.

    This first chooses a center which is least likely to be satisfied simultaneously with another spin. It then chooses the second center to be the spin which has least intersection with the second center.
    """
    # setup
    distance = least_likely_distance(circuit, symmatrix)
    find_centers_method = outlier_centers
    refine_criterion = carver(circuit)
    refine_method = general_refine_method(distance, find_centers_method)

    clustering = iterative_clustering(refine_criterion, refine_method)

    root = FlexNode(set(circuit.inspace))
    tree = clustering(root)

    result_clusters = [leaf.value for leaf in tree.leaves]
    return result_clusters


def main():
    import numpy as np

    circuit = IMul(2, 3)
    filename = f"data/input_level_intersection_IMul2x3x0_rep=5001_deg=2.csv"
    symmatrix = np.loadtxt(filename, delimiter=",")

    clusters = likelihood_clustering(circuit, symmatrix)
    for leaf in clusters:
        print(leaf)


if __name__ == "__main__":
    main()
