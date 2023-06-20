from numpy.random import f
from torch import threshold
from filtered_constraints import (
    filtered_constraints as fc,
    full_constraints as fullconst,
    get_basin,
)
from filtered_solver import filtered_solver as fsolve
from lmisr_interface import call_solver as lmsir_solver
from mysolver_interface import call_my_solver as solver
from aux_generation import sample_auxfile, read_auxfile, uniquify
from ising import IMul

import heapq
import matplotlib.pyplot as plt
import numpy as np
import math
import json


def top_n_rhos(circuit, n, jobnum=-1):
    aux_array = None
    if circuit.A:
        aux_array = circuit.get_aux_array(binary=True)
    _, _, rhos = lmsir_solver(circuit.N1, circuit.N2, aux_array, fullreturn=True)
    if jobnum != -1:
        print(f"  Solved job number {jobnum}.")
    rhos = np.array(rhos)
    return rhos, np.argsort(rhos)[-n:]


def zip_rhos(circuit, rhos):
    in_outaux = [
        (n.asint(), m.asint())
        for n in circuit.inspace.copy()
        for m in circuit.outauxspace.copy()
        if circuit.f(n) != m
    ]
    return zip(in_outaux, rhos)


def states_ordered_by_rho(circuit, threshold=1e-8):
    N1 = circuit.N1
    N2 = circuit.N2
    aux = circuit.get_aux_array()
    full, _ = fullconst(N1, N2, aux, degree=2)
    _, _, rhos = solver(full.to_sparse_csc(), fullreturn=True)
    zipped = list(zip_rhos(circuit, rhos))
    zipped = sorted(zipped, key=lambda x: x[1])

    return [x for x in zipped if x[1] >= threshold]


def run_level_rho_analysis(N1, N2, A, runs=100, n=1):
    circuits = [IMul.gen_random_circuit(N1, N2, A) for _ in range(runs)]
    maxrho_indices = [
        x
        for i, circ in enumerate(circuits)
        for x in top_n_rhos(circuit=circ, n=n, jobnum=i)[1]
    ]
    # maxrho_indices = sum(maxrho_indices, [])
    return maxrho_indices


def make_histogram(data, bins, title="", fname=""):
    plt.hist(data, bins=bins)
    ticks = np.array(list(set(data)))
    plt.xticks(np.array(list(set(data))))
    plt.setp(plt.gca().get_xticklabels(), rotation=50, fontsize="x-small")
    plt.title(title)
    plt.savefig(fname + ".png")
    plt.clf()


def make_rho_histograms(N1, N2, A, runs, n):
    fname = f"rhostats/IMul{N1}x{N2}x{A}_max_{n}_rhos"
    title = f"Maximum Indices of Top {n} Artifical Variables\n{runs} IMul{N1}x{N2}x{A} circuits with random auxiliary arrays"
    data = run_level_rho_analysis(N1, N2, A, runs, n)
    make_histogram(data, bins=1 << (N1 + N2), title=title, fname=fname)


def main_histogram():
    N1 = 3
    N2 = 4
    A = 6
    runs = 50
    n = 3
    make_rho_histograms(N1, N2, A, runs, n)


def hamdist(num1, num2):
    return bin(num1 ^ num2)[2:].count("1")


# count the number of times each input level has a constraint with a significant rho
def count_level_frequency(circuit):
    rhos = list(states_ordered_by_rho(circuit))
    levelcounts = {i: 0 for i in range(1 << circuit.N)}
    leveled_rhos = [[x[0] for x in rhos if x[0][0] == i] for i in range(1 << circuit.N)]
    for x in rhos:
        levelcounts[x[0][0]] += 1

    for i in levelcounts:
        print(f"level {i} featured {levelcounts[i]} times")
        print("  ", leveled_rhos[i])


def compare_rho_to_basin(circuit):
    MA = circuit.M + circuit.A
    rhos = states_ordered_by_rho(circuit)
    print(f"# significant rhos: {len(rhos)}")
    basincounts = {i: [0, 0] for i in range(1, MA + 1)}
    for x in rhos:
        s = x[0][0]
        c = circuit.f(s).asint()
        w = x[0][1]
        basincounts[hamdist(c, w)][0] += 1
        basincounts[hamdist(c, w)][1] += x[1]

    for i in basincounts:
        print(
            f"basin {i} occurred {basincounts[i][0]} times with collective rho of {basincounts[i][1]}"
        )


def remap_keys(d):
    return [{"key": k, "value": v} for k, v in d.items()]


def and_aux_rho_avg(N1, N2, A, runs=100):
    path = f"/home/ikmarti/Desktop/ising-clustering/constraint-reduction/aux_arrays/IMul{N1}x{N2}x{A}_AND_AUX.dat"
    filename = f"/home/ikmarti/Desktop/ising-clustering/constraint-reduction/rhostats/IMul{N1}x{N2}x{A}_AND_AUX_stats_runs={runs}.json"
    rhodict = {
        (n, m): 0 for n in range(2 ** (N1 + N2)) for m in range(2 ** (N1 + N2 + A))
    }

    def add_rhos(circuit):
        rhos = states_ordered_by_rho(circuit)
        for x in rhos:
            rhodict[x[0]] += x[1]

    auxes = sample_auxfile(path, samples=runs)
    circuits = [IMul(N1, N2, auxlist=aux) for aux in auxes]

    for i, circ in enumerate(circuits):
        add_rhos(circ)
        print(f"finished run {i}")

    print(heapq.nlargest(20, rhodict.items(), key=lambda i: i[1]))

    with open(uniquify(filename), "w") as file:
        file.write(
            json.dumps(
                remap_keys(rhodict), sort_keys=True, indent=4, separators=(",", ": ")
            )
        )


if __name__ == "__main__":
    N1 = 3
    N2 = 3
    A = 2
    runs = 100
    and_aux_rho_avg(N1, N2, A, runs)
