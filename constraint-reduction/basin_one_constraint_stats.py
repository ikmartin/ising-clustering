from ising import IMul
from spinspace import Spinspace, Spin
from functools import cache
from matplotlib import pyplot as plt
from itertools import combinations
import numpy as np

import random
import json
import csv
import pickle
import time
import math

from fast_constraints import filtered_constraints, sequential_constraints
from solver import LPWrapper

from joblib import Parallel, delayed


################################
## HELPER METHODS
################################
def get_valid_aux(auxfile="", num=-1):
    """Get the first <num> auxiliary arrays from a user-input file"""
    if auxfile == "":
        auxfile = str(input("Enter path to feasible aux file: "))

    # get the first <runs> many auxes
    with open(auxfile, "r") as file:
        return [json.loads(line) for line in file.readlines()[:num]]


def dec2bin(num, fill):
    return list(bool(int(a)) for a in bin(num)[2:].zfill(fill))


# can speed this up by first inverting num and then flipping bits at indices not in ind
# whenever ind is more than twice the length of the binary string
def flip_bits(num, ind):
    for k in ind:
        # XOR with (00001) shifted into kth position
        num = num ^ (1 << k)
    return num


def num_constraints(N, M, A, percent=1.0):
    return 2**N * int((2 ** (M + A) - 2**A) * percent)


def num_constraints_per_level(M, A, percent=1.0):
    return int((2 ** (M + A) - 2**A) * (percent))


##########################
## DATA PERSISTENCE
##########################
datapath = "/home/ikmarti/Desktop/ising-clustering/constraint-reduction/data/"
ham_dist_dict = {}


def save():
    with open(datapath + "ham_dist_dict.pickle", "wb") as file:
        pickle.dump(ham_dist_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


def load():
    try:
        with open(datapath + "ham_dist_dict.pickle", "rb") as file:
            ham_dist_dict = pickle.load(file)
    except (OSError, IOError, FileNotFoundError) as e:
        ham_dist_dict = {}


##########################
## INTEGER-WISE SPINS
##########################


def add_all_aux(outnum, A):
    """Given an integer representing an output, generates a list of integers representing all possible auxiliary values appended to the end of the output."""
    # bit shift to make room for the auxiliary
    num = outnum << A
    return [num + i for i in range(2**A)]


def ham_dist_from_bin(N, num, dist):
    """Returns all binary strings length N which are hamming distance <dist> from the given number num.

    Parameters
    ----------
    N : int
        the length of the binary string
    num : int
        the number to be used as the center of the hamming distance circle
    dist : int
        the hamming distance radius
    """

    try:
        return ham_dist_dict[(N, num, dist)]
    except KeyError:
        ham_dist_dict[(N, num, dist)] = []
        for ind in combinations(range(N), r=dist):
            flipped_guy = flip_bits(num, ind)
            ham_dist_dict[(N, num, dist)].append(flipped_guy)

        return ham_dist_dict[(N, num, dist)]


############################
## LEVEL SEIVES
############################


def basin_one_sieve(circuit, bitlist):
    """Sieve which only takes basin one constraints. Includes the constraints which differ in one of the indicated bits.

    Parameters
    ----------
    circuit : PICircuit
        the circuit for whom the constraints are being built
    bitlist : list[bool]
        a boolean-valued list of length circuit.M + circuit.A.

    Example
    -------
    basin_one_sieve(circuit=IMul(2,2,1), bitlist = [0, 0, 0, 1, 1]) will allow two constraints, the wrong output which differs from correct in the auxiliary bit and the wrong output which differs from correct in the ones bit.
    """
    MA = circuit.M + circuit.A
    assert len(bitlist) == MA
    indices = [(MA - (i + 1),) for i in range(0, MA) if bitlist[i]]

    def sieve(inspin):
        c_outaux = circuit.f(inspin).asint()
        return [flip_bits(c_outaux, ind) for ind in indices], len(indices)

    return sieve


###################################
## SOLVER METHODS
###################################
def partial_constraint_solver(circuit, sieve):
    """Builds a partial constraint solver"""
    make, terms = filtered_constraints(circuit=circuit, degree=2, sieve=sieve)
    lpwrap = LPWrapper(keys=terms)
    numconst = 0

    for inspin in circuit.inspace:
        M, newconst = make(inspin)
        lpwrap.add_constraints(M=M)
        numconst += newconst

    return lpwrap, numconst


def full_solver(circuit):
    make, terms = sequential_constraints(circuit=circuit, degree=2)
    lpwrap = LPWrapper(keys=terms)

    for inspin in circuit.inspace:
        M = make(inspin)
        lpwrap.add_constraints(M=M)

    return lpwrap


def get_random_auxarray(N, A):
    auxspace = Spinspace((A,))
    return [auxspace.rand().spin() for _ in range(2**N)]


def gen_rand_circuit(N1, N2, A):
    if A:
        return IMul(N1, N2, auxlist=get_random_auxarray(N1 + N2, A))
    else:
        return IMul(N1, N2)


######################################
## STATS METHODS
######################################
def plot_success_across_aux(auxes, false_positives, fname, title=""):
    plt.xlabel("Number of Auxiliaries")
    plt.ylabel("Percentage of false positives")
    plt.title(title)
    plt.plot(auxes, false_positives)
    plt.savefig(fname + ".png")
    plt.clf()


def csv_false_positives(data, header, fname):
    sheet = [header]
    data = list(data)
    sheet += data

    with open(fname + ".csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerows(sheet)


def perform_run(circuit, sieve, run_id):
    """Performs a single check of the circuit using sieve. Returns 1 for a false positive, 0 otherwise."""
    print(f"Run {run_id}:")
    print(f"  building solver...")
    solver, numconst = partial_constraint_solver(circuit, sieve)
    solver = solver.solver
    print(f"  solving...", end="")
    status = solver.Solve()
    print(f"  done.")
    psolved = status == solver.OPTIMAL
    print(f"  false positive: {psolved}")
    print(f"  number of constraints: {numconst}")

    return int(psolved), numconst


def write_constraints_to_file(circuit, sieve, filename):
    N1, N2, A = circuit.N1, circuit.N2, circuit.A
    outauxes = {inspin.asint(): [] for inspin in circuit.inspace.copy()}
    avgconst = 0
    for inspin in circuit.inspace.copy():
        correct = circuit.f(inspin)
        wrong = sieve(inspin)[0]
        const = [f"{s} - {correct.asint()}" for s in wrong]
        outauxes[inspin.asint()] += const
        print(f"input {inspin}:")
        print(f"  correct : {correct.binary()} ~> {correct}")
        avgconst += len(wrong)
        for i, s in enumerate(wrong):
            print(
                f"  wrong #{i}: {circuit.outauxspace[s].binary()} ~> {circuit.outauxspace[s]}"
            )
    avgconst /= circuit.inspace.size

    print(f"SUMMARY IMul{circuit.N1}x{circuit.N2}x{circuit.A}")
    print(f"  avg num constraints per input level: {avgconst}")

    with open(f"data/{filename}.json", "w") as file:
        json.dump(outauxes, file)


def basin_one_stats(N1, N2, A, bitlist, runs=200, fname="", fullcheck=False):
    # initialize the data with an entry of 0 percent, 100% false positive rate, 0 constraints
    M = N1 + N2

    false_pos_count = 0
    numconst = 0

    circuit, sieve = None, None
    for run in range(runs):
        circuit = gen_rand_circuit(N1, N2, A)
        sieve = basin_one_sieve(circuit, bitlist)
        new_falsepos, newconst = perform_run(circuit, sieve, run)
        if new_falsepos and fullcheck:
            print("   BUILDING FULL SOLVER...")
            solver = full_solver(circuit=circuit).solver
            print("   SOLVING FULL SOLVER...")
            status = solver.Solve() == solver.OPTIMAL
            print(f"   RESULT: {status}")
            new_falsepos = int(not status)
        false_pos_count += new_falsepos
        numconst += newconst

    print(f"----------------\nSUMMARY OF IMul{N1}x{N2}x{A} basin_one_constraints:\n")
    print(f"  number of constraints used: {numconst}")
    print(f"  number of false positives: {false_pos_count}")

    # write_constraints_to_file(circuit, sieve, filename=fname)
    # time.sleep(0.5)

    return false_pos_count / runs, numconst / runs


def basin_one_across_aux(N1, N2, bitlist, minA, maxA, runs, fname, fullcheck=False):
    if fname == "":
        fname = input("Enter a filename to be used for graph and data:")

    fname += "_fullcheck=True" if fullcheck else "_fullcheck=False"

    data = np.zeros((maxA - minA + 1, 3))

    for i, A in enumerate(range(minA, maxA + 1)):
        data[i, 0] = A
        data[i, 1], data[i, 2] = basin_one_stats(
            N1,
            N2,
            A,
            bitlist=bitlist + [0] * A,
            runs=runs,
            fname=fname,
            fullcheck=fullcheck,
        )

    print(f"SUMMARY OF IMul{N1}x{N2} for {minA} <= A <= {maxA}")
    for row in data:
        print(
            f" {row[0]} auxes; {row[1]*runs} false positives out of {runs} runs; {row[2]} constraints used"
        )

    csv_false_positives(data, ["Percent", "False Positives", "Constraints"], fname)
    plot_success_across_aux(
        data[:, 0],
        data[:, 1],
        fname,
        title=f"Basin One Constraints for IMul{N1}x{N2} across different auxiliary values\nbitlist = {bitlist}",
    )


if __name__ == "__main__":
    N1 = 3
    N2 = 4
    M = N1 + N2
    minA = 7
    maxA = 10
    runs = 1000
    subdiv = 10
    load()
    # run_const_radial_sieve(N1, N2, A, minper, maxper, runs, subdiv)
    # run_bin_search_radial_sieve(N1, N2, A, minper, maxper, runs, reps)
    # run_lvec_random()
    cut = 2
    bitlist = [1, 1, 1, 1, 1, 1, 1]
    fname = f"IMul{N1}x{N2}_basin_one_constraints_bitlist={bitlist}"
    basin_one_across_aux(N1, N2, bitlist, minA, maxA, runs, fname, fullcheck=False)
    save()
