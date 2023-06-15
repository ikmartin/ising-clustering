from ising import IMul
from spinspace import Spinspace, Spin
from functools import cache
from matplotlib import pyplot as plt
from itertools import combinations

import random
import json
import csv
import pickle
import time

from fast_constraints import filtered_constraints
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
        b = dec2bin(num, N)
        ham_dist_dict[(N, num, dist)] = []
        for ind in combinations(range(N), r=dist):
            flipped_guy = flip_bits(num, ind)
            print(
                f"""
                  flipped {bin(num)[2:].zfill(N)}\n
                  to      {bin(flipped_guy)[2:].zfill(N)}"""
            )
            ham_dist_dict[(N, num, dist)].append(flipped_guy)
            time.sleep(0.1)

        return ham_dist_dict[(N, num, dist)]


#################################
## DIFFERENT SIEVE METHODS
#################################


def sieve_machine(circuit, weight_dict, level_sieve, percent):
    """Currying method. Produces a sieve which filters inputs based on the weight dictionary and input levels based on some other criterion (randomness, hamming distance from correct output, etc).

    Parameters
    ----------
    circuit : PICircuit
        a reference to the circuit being used
    weight_dict : dictionary
        a dictionary keyed by input spins and valued in decimals. The weight of input s is the percentage of constraints to be included from input level s.
    level_sieve : callable
        another sieve method which will be used to filter the actual input levels. For example, choose constraints randomly, by hamming distance to correct output, etc.
    """

    constraints_per_level = 2 ** (circuit.M + circuit.A) - 2**circuit.A

    def sieve(inspin):
        percent = max(weight_dict[inspin], 1 / constraints_per_level)
        sieve2 = level_sieve(circuit, percent=percent)
        return sieve2(inspin)

    return sieve


@cache
def lvec_weights(circuit, percent=0.2):
    """An importance ranking for inputs. Returns a weight_dict where harder-to-satisfy inputs are assigned higher weights.
    Parameters
    ----------
    circuit : PICircuit
        a reference to the circuit being used
    percent : float
        the total percentage of constraints to be included
    """
    avglvecs = {inspin: circuit.fastavglvec_dist(inspin) for inspin in circuit.inspace}
    sorted_inspins = sorted(avglvecs, key=avglvecs.get, reverse=True)
    n = len(sorted_inspins)

    # m = 6 * (n - percent * n) / (n * (n + 1) * (2 * n + 1))
    b = 2 * percent
    m = b / n

    def weight(i):
        return b - m * i

    thing = 0
    for i in range(len(sorted_inspins)):
        thing += weight(i)
        print(f"spin {sorted_inspins[i]} gets weight {weight(i)}")

    print(f"avg percentage {thing/n}")
    # percentage of input level i to be incorporated
    return {sorted_inspins[i]: weight(i) for i, _ in enumerate(sorted_inspins)}


def const_weights(circuit, percent):
    return {s: percent for s in circuit.inspace.copy()}


############################
## LEVEL SEIVES
############################


@cache
def random_sieve(circuit, percent=0.2):
    """Currying method. Keeps only <percent> of the constraints, randomly throws out the rest."""
    num_outaux = 2 ** (circuit.M + circuit.A)
    num_constraints = int(num_outaux * percent)

    def sieve(inspin):
        return random.sample(list(range(num_outaux)), num_constraints), num_constraints

    return sieve


def radial_seive(circuit, percent=0.2, noise=0):
    """Currying method. Aims to keep <percent> percentage of all constraints in an input level. Constraints determined by wrong outputs closest to the correct output."""
    num_outaux = 2 ** (circuit.M + circuit.A)
    num_constraints = int(num_outaux * percent)

    def sieve(inspin):
        out = circuit.fout(inspin)
        correct_out = out.asint()
        wrong_outaux = []
        radius = 1
        constraints_left = num_constraints
        while constraints_left > 0:
            # get all outputs distance <radius> from correct_out
            outshell = ham_dist_from_bin(circuit.M, num=correct_out, dist=radius)
            outauxshell = [
                num
                for sublist in [add_all_aux(out, circuit.A) for out in outshell]
                for num in sublist
            ]
            # update the number of constraints still left to add
            constraints_left -= len(outauxshell)

            # only add enough constraints to hit the desired percentage
            if constraints_left < 0:
                wrong_outaux += outauxshell[:constraints_left]
                break
            else:
                wrong_outaux += outauxshell

            radius += 1

        return wrong_outaux, num_constraints

    return sieve


###################################
## SOLVER METHODS
###################################
def partial_constraint_solver(circuit, sieve):
    """Builds a partial constraint solver"""
    make, terms = filtered_constraints(circuit=circuit, degree=2, sieve=sieve)
    lpwrap = LPWrapper(keys=terms)

    for inspin in circuit.inspace:
        lpwrap.add_constraints(M=make(inspin))

    return lpwrap


def get_random_auxarray(N, A):
    auxspace = Spinspace((A,))
    return [auxspace.rand().spin() for _ in range(2**N)]


def get_random_circuit(N1, N2, A, num):
    circuit = IMul(N1=N1, N2=N2)
    circuit.set_all_aux(get_random_auxarray(N1 + N2, A))
    return circuit


def gen_percent_circuits_list(N1, N2, A, minper, maxper, num):
    return [
        (
            IMul(N1, N2, auxlist=get_random_auxarray(N1 + N2, A)),
            minper + t * (maxper - minper) / num,
        )
        for t in range(num)
    ]


######################################
## STATS METHODS
######################################
def plot_false_positives(percents, false_positives, fname, title=""):
    plt.xlabel("Percentage of constraints considered")
    plt.ylabel("Percentage of false positives")
    plt.title(title)
    plt.plot(percents, false_positives)
    plt.savefig(fname + ".png")
    plt.clf()


def csv_false_positives(percents, false_positives, fname):
    data = [
        [
            "Percentage of Constraints",
            f"Percentage of False Positives",
        ]
    ]
    data += [[percents[i], false_positives[i]] for i in range(len(percents))]

    with open(fname + ".csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerows(data)


def perform_run(circuit, sieve, run_id):
    """Performs a single check of the circuit using sieve. Returns 1 for a false positive, 0 otherwise."""
    print(f"Run {run_id}:")
    print(f"  building solver...")
    solver = partial_constraint_solver(circuit, sieve).solver
    print(f"  solving...", end="")
    status = solver.Solve()
    print(f"  done.")
    psolved = status == solver.OPTIMAL
    print(f"  false positive: {psolved}")

    return int(psolved)


def sieve_stats(
    weight_gen,
    level_sieve,
    N1=3,
    N2=3,
    A=3,
    runs=200,
    minper=0.1,
    maxper=0.2,
    subdiv=50,
    fname="",
    title="",
):
    if fname == "":
        fname = input("Enter a filename to be used for graph and data:")

    if title == "":
        title = input("Enter a title to be used for the graph:")

    percents, false_pos = [], []
    for circuit, percent in gen_percent_circuits_list(
        N1, N2, A, minper, maxper, subdiv
    ):
        false_pos_count = 0
        sieve = sieve_machine(
            circuit, weight_gen(circuit, percent), level_sieve, percent
        )

        """
        results = Parallel(n_jobs=20)(
            delayed(perform_run)(circuit, sieve, run) for run in range(runs)
        )
        """

        for run in range(runs):
            false_pos_count += perform_run(circuit, sieve, run)

        percents.append(percent)
        false_pos.append(false_pos_count / runs)

        print(f"----------------\nSUMMARY OF PERCENT {percent}\n")
        print(f"  percentage of constraints used: {percent}")
        print(f"  number of false positives: {false_pos_count}")

        csv_false_positives(percents, false_pos, fname)
        plot_false_positives(percents, false_pos, fname, title)


def run_random_sieve(A=5):
    N1 = 3
    N2 = 4
    minper = 0.001
    maxper = 0.10
    runs = 500
    subdiv = 30
    sieve_stats(
        weight_gen=const_weights,
        level_sieve=random_sieve,
        N1=N1,
        N2=N2,
        A=A,
        runs=runs,
        minper=minper,
        maxper=maxper,
        subdiv=subdiv,
        fname=f"IMul{N1}x{N2}x{A}_random_runs={runs}_minper={minper}_maxper={maxper}",
        title="False Positives From Randomly Slashed Constraint Matrices",
    )


def run_const_radial_sieve():
    N1 = 3
    N2 = 3
    A = 1
    minper = 0.005
    maxper = 0.1
    runs = 500
    subdiv = 10
    sieve_stats(
        weight_gen=const_weights,
        level_sieve=radial_seive,
        N1=N1,
        N2=N2,
        A=A,
        runs=runs,
        minper=minper,
        maxper=maxper,
        subdiv=subdiv,
        fname=f"IMul{N1}x{N2}x{A}_const_radial_runs={runs}_minper={minper}_maxper={maxper}",
        title="False Positives\nConstraints close to correct output are kept",
    )


def run_lvec_random():
    N1 = 4
    N2 = 4
    A = 6
    minper = 0.001
    maxper = 0.1
    runs = 200
    subdiv = 30
    sieve_stats(
        weight_gen=lvec_weights,
        level_sieve=random_sieve,
        N1=N1,
        N2=N2,
        A=A,
        runs=runs,
        minper=minper,
        maxper=maxper,
        subdiv=subdiv,
        fname=f"IMul{N1}x{N2}x{A}_lvecweights_runs={runs}_minper={minper}_maxper={maxper}",
        title="False Positives from Slashed Constraints Matrices\nLvec importance w/ random constraint selection within input levels",
    )


if __name__ == "__main__":
    load()
    # run_const_radial_sieve()
    run_random_sieve(A=0)
    run_random_sieve(A=1)
    run_random_sieve(A=2)
    run_random_sieve(A=3)
    run_random_sieve(A=4)
    run_random_sieve(A=5)
    run_random_sieve(A=6)
    run_random_sieve(A=7)

    # run_lvec_random()
    save()
