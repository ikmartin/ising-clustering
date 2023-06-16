from mysolver_interface import call_my_solver, call_imul_solver
from lmisr_interface import call_solver
from itertools import chain, combinations
from ising import IMul, PICircuit
import numpy as np
import random
import torch

from filtered_constraints import filtered_constraints as fc
from filtered_constraints import basin_d_constraints


def dec2bin(num, fill):
    return list(bool(int(a)) for a in bin(num)[2:].zfill(fill))


def filtered_solver(constraint_sets):
    """Currying function. Returns a method for running through a sequence of constraint sets.

    Parameters
    ----------
    constraint_set : iterable, generator.
        the sets of constraints through which to iterate. Must be in sparse csc format (call torch.to_sparse_csc). Advantageous to make this a generator to save on build time.
    """
    tolerance = 1e-4
    cutoff = 1e-1

    def solve(detailed=False):
        """Runs the LPSolver on the given constraint filtering. Returns true if it passes all,"""
        status = []
        for const in constraint_sets:
            status.append(cutoff > call_my_solver(const, tolerance=tolerance))
            if status[-1] == False:
                if detailed == False:
                    return False
                break

        if detailed == False:
            return all(status)

        return status

    return solve


def flip_bits(num, ind):
    """Returns the integer resulting from flipping the bits of num (int) in the places indicated by ind (tuple). Note that num the binary representation of num is said to have infinitely many pre-pended zeros, i.e. 7 ~ ...000000111 rather than 7 ~ 111."""
    for k in ind:
        # XOR with (...001) shifted into kth position
        num = num ^ (1 << k)
    return num


basin_dict = {}


def get_basin(N, num, d):
    """Returns all binary strings length N which are hamming distance <dist> from the given number num.

    Parameters
    ----------
    N : int
        the length of the binary string
    num : int
        the number to be used as the center of the hamming distance circle
    d : int
        the hamming distance radius
    """

    try:
        return basin_dict[(N, num, d)]
    except KeyError:
        b = dec2bin(num, N)
        basin_dict[(N, num, d)] = []
        for ind in combinations(range(N), r=d):
            flipped_guy = flip_bits(num, ind)
            basin_dict[(N, num, d)].append(flipped_guy)

        return basin_dict[(N, num, d)]


def sample_basin(N, num, d, count):
    # count must be at most 2**N
    if count > (1 << N):
        count = 1 << N

    return random.sample(get_basin(N, num, d), count)


def run_test(circuit, basin_num=2):
    constset = [basin_d_constraints(circuit, d) for d in range(1, basin_num + 1)]
    solve = filtered_solver(constset)
    print(solve(detailed=True))


def test_lmisr():
    N1 = 2
    N2 = 2
    aux_array = np.array(IMul.gen_random_auxarray(N1 + N2, 1, binary=True))
    print(aux_array)
    return call_solver(N1, N2, aux_array)


if __name__ == "__main__":
    circuit = IMul.gen_random_circuit(2, 2, 1)
    # run_test(circuit)
    print(test_lmisr())
