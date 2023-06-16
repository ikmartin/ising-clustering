from filtered_constraints import filtered_constraints as fc
from filtered_solver import filtered_solver as fsolve
from lmisr_interface import call_solver as lmsir_solver
from ising import IMul

import matplotlib.pyplot as plt
import numpy as np


def top_n_rhos(circuit, n, jobnum=-1):
    aux_array = None
    if circuit.A:
        aux_array = circuit.get_aux_array(binary=True)
    _, _, rhos = lmsir_solver(circuit.N1, circuit.N2, aux_array, fullreturn=True)
    if jobnum != -1:
        print(f"  Solved job number {jobnum}.")
    rhos = np.array(rhos)
    return rhos, np.argsort(rhos)[-n:]


def run_rhos_analysis(N1, N2, A, runs=100, n=1):
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
    data = run_rhos_analysis(N1, N2, A, runs, n)
    make_histogram(data, bins=1 << (N1 + N2), title=title, fname=fname)


if __name__ == "__main__":
    N1 = 3
    N2 = 3
    A = 6
    runs = 50
    n = 3
    make_rho_histograms(N1, N2, A, runs, n)
