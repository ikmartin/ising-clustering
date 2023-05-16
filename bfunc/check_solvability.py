from ising import IMatch
from spinspace import Spin
from itertools import combinations


def issolvable(N, vals):
    circuits = [IMatch(N, val) for val in vals]
    circuit = circuits[0]
    for circ in circuits[1:]:
        circuit += circ

    solver = circuit.build_solver()
    return solver.Solve() == solver.OPTIMAL


def hamdist(s1, s2):
    if s1.shape != s2.shape:
        raise TypeError(
            f"Cannot calculate distance between spins of shape {s1.shape} and {s2.shape}"
        )
    return sum([s1[i] != s2[i] for i in range(s1.shape[0])])


def get_solvability_dict(N, one_count=2):
    # dictionary of dictionaries
    success_dict = {}
    for ind in combinations(list(range(2**N)), r=one_count):
        spin1 = Spin(ind[0], shape=(N,))
        spin2 = Spin(ind[1], shape=(N,))
        success_dict[ind] = issolvable(N, ind)

    return success_dict


def check_solvability(N, one_count=2):
    success_dict = get_solvability_dict(N, one_count)
    for key, value in success_dict.items():
        print(f"ones at {key} are simultaneously solvable: {value}")


check_solvability(int(input("Give input size: ")), int(input("Give # of ones: ")))
