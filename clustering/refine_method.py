import numpy as np
from spinspace import Spin
from ising import PICircuit, IMul


def trade_refine_method(data, indices, score: Callable):
    def method(cluster: set[Spin]) -> tuple[set[Spin], set[Spin]]:
        clust1, clust2 = cluster, set([])

    return method
