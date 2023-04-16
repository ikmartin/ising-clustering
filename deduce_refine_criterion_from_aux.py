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


def test_i_lvec_on_other_levels(i: int):
    circuit = ising.IMul(2, 2)
    lvec = circuit.lvec(circuit.inspace.getspin(i))

    for s in circuit.inspace:
        print(
            f"{s.asint()}: 0's lvec satisfies? {circuit.level(inspin=s, ham_vec=lvec, weak=True)}"
        )


def compare_lvecs_on_other_levels():
    circuit = ising.IMul(2, 2)

    table = []
    for i in range(circuit.inspace.size):
        s = circuit.inspace.getspin(i)
        lvec = circuit.lvec(circuit.inspace.getspin(i))
        arr = []
        for j in range(circuit.inspace.size):
            t = circuit.inspace.getspin(j)
            arr.append(int(circuit.level(inspin=t, ham_vec=lvec, weak=True)))

        table.append(arr)

    return table


if __name__ == "__main__":
    # all_lvecs()
    i = 7
    j = 14
    table = compare_lvecs_on_other_levels()
    print(np.arange(16))
    for i, row in enumerate(table):
        print(i, np.array(row))

    test_i_lvec_on_other_levels(5)
