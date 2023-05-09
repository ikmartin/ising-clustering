from functools import cache
from spinspace import Spin, Spinspace
from ising import IMul
from functional_clustering import (
    farthest_centers,
    outlier_centers,
    virtual_hamming_distance,
    carver,
    general_refine_method,
    iterative_clustering,
    FlexNode,
)
import numpy as np
import networkx as nx


def general_graph_refine_method():
    pass


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
