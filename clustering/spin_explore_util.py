from spinspace import Spin
import spinspace as ss
from ising import IMul
import numpy as np

np.set_printoptions(linewidth=150)


def return_if_fail_sgn(inspins: list[Spin], G):
    graph = [G.inout(s) for s in inspins]
    sgn = ss.sgn(graph)

    return G.levels(inspins=inspins, ham_vec=sgn)


def get_spin_info(inspins: list[Spin], G):
    graph = [G.inout(s) for s in inspins]
    sgn = ss.sgn(graph)
    qvec = ss.qvec(graph)

    levels_sgn = G.levels(inspins=inspins, ham_vec=sgn, list_fails=True)
    levels_qvec = G.levels(inspins=inspins, ham_vec=qvec, list_fails=True)

    print(f"  sgn:  {sgn}", flush=True)
    if len(levels_sgn) == 0:
        print("  all levels satisfied for sgn.")
    else:
        print(f"  levels NOT satisfied for sgn. Here are the spins that fail:")
        for s in levels_sgn:
            print(f"    {s.asint()} : {s.spin()}")

    print(f"  qvec: {qvec}", flush=True)
    if len(levels_qvec) == 0:
        print("  all levels satisfied for qvec.")
    else:
        print(f"  levels NOT satisfied for qvec. Here are the spins that fail:")
        for s in levels_qvec:
            print(f"    {s.asint()} : {s.spin()}")


def distances(inspins: list[Spin], G):
    graph = [G.inout(s) for s in inspins]
    sgn = ss.sgn(graph)
    qvec = ss.qvec(graph)

    # get distances
    dist = {}
    for i in range(len(inspins)):
        for j in range(i + 1, len(inspins)):
            dist[i, j] = ss.vdist(graph[i], graph[j])

    maxd = max(dist.values())
    maxpairs = [key for key, value in dist.items() if value == maxd]
    spinpairs = [(inspins[i].asint(), inspins[j].asint()) for i, j in maxpairs]
    print(f"  max distance is {maxd} achieved by\n    {spinpairs}")
    print(f"  diameter is {ss.diameter(graph)}")


def spin_info_loop_MUL2x2():
    G = IMul(N1=2, N2=2)

    print("=======================")
    print("SPIN UTILITY FOR MUL2x2")
    print("=======================\n")
    while True:
        a = [
            int(x)
            for x in input(
                "List input spins as integers, separated by spaces.\n>   "
            ).split()
        ]
        spins = [G.inspace.getspin(spin=i) for i in a]
        print()
        get_spin_info(inspins=spins, G=G)
        distances(inspins=spins, G=G)
        print("-----------------------------------------------\n")


def analyze_subsets_loop_MUL2x2():
    import itertools

    G = IMul(N1=2, N2=2)

    print("==============================")
    print("ANALYZE SUBSET LOOP FOR MUL2x2")
    print("==============================\n")
    while True:
        a = [
            int(x)
            for x in input(
                "List input spins as integers, separated by spaces.\n>   "
            ).split()
        ]

        for i in range(len(a) + 1):
            print(f"SUBSETS LENGTH {i}")
            for subset in list(itertools.combinations(a, i)):
                spins = [G.inspace.getspin(spin=i) for i in subset]
                if return_if_fail_sgn(inspins=spins, G=G) == False:
                    print(f"spins: {[s.asint() for s in spins]}")
                    get_spin_info(inspins=spins, G=G)
            print("===============================================\n")


if __name__ == "__main__":
    spin_info_loop_MUL2x2()
    # analyze_subsets_loop_MUL2x2()
