import numpy as np
import ising
import sys
from spinspace import Spin, Spinspace
from topdown import TopDown

np.set_printoptions(threshold=200, linewidth=200)


def get_clustering_from_aux(circuit: ising.PICircuit, aux_array):
    circuit.set_aux(aux_array=aux_array)
    clustering = {}
    for s in circuit.inspace:
        key = circuit.faux(s).asint()

        # add the spin to the dictionary
        if isinstance(clustering[key], list):
            clustering[key].append(s.asint())
        # otherwise don't add it to dictionary
        else:
            clustering[key] = [s.asint()]

    return clustering


def sum_of_normals(circuit, inspin):
    s = inspin
    fs = circuit.fout(s)
    normal = np.zeros(int(circuit.G * (circuit.G + 1) / 2))
    for t in circuit.outspace:
        inout = Spin.catspin((s, t))
        normal += inout.vspin().spin() - circuit.inout(s).vspin().spin()

    return normal


def all_lvecs():
    circuit = ising.IMul(2, 2)
    for s in circuit.inspace:
        print(s.asint(), " : ", sum_of_normals(circuit, s))

    for s in circuit.inspace:
        print(
            f"{s.asint()}: lvec satisfies {circuit.level(inspin=s, ham_vec=circuit.lvec(s), weak=True)}"
        )


def test_0_lvec_on_other_levels():
    circuit = ising.IMul(2, 2)
    lvec0 = circuit.lvec(circuit.inspace.getspin(0))

    for s in circuit.inspace:
        print(
            f"{s.asint()}: 0's lvec satisfies? {circuit.level(inspin=s, ham_vec=lvec0, weak=True)}"
        )


if __name__ == "__main__":
    # all_lvecs()
    test_0_lvec_on_other_levels()
