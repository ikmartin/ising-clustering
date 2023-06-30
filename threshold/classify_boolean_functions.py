from ising import BCircuit as BoolCirc
from copy import deepcopy
import math


def dec2bin(num, fill):
    return list(int(a) for a in bin(num)[2:].zfill(fill))


def bin2dec(blist):
    N = len(blist)
    return sum([1 << (N - i - 1) for i in range(N) if blist[i]])


def bin2hex(b):
    return "".join(
        [
            format(int("".join([str(int(x)) for x in b[i : (i + 4)]]), 2), "x")
            for i in range(0, len(b), 4)
        ]
    )


def hex2bin(num, fill):
    return dec2bin(int(num, 16), fill)


def check_strong_neutralizability(func):
    if isinstance(func, BoolCirc) is False:
        N = int(math.log2(len(func)))
        circ = BoolCirc(N, func)
    else:
        circ = func
    nsolver = circ.neutralizable_solver()
    return nsolver.Solve() == nsolver.OPTIMAL


def check_threshold(func):
    if isinstance(func, BoolCirc) is False:
        N = int(math.log2(len(func)))
        circ = BoolCirc(N, func)
    else:
        circ = func
    tsolver = circ.build_solver()
    return tsolver.Solve() == tsolver.OPTIMAL


def write_funcs(funcs, dim, func_class_name, ashex=True):
    # format name
    formatstr = "HEX" if ashex else "BOOL"
    # dimension
    N = dim
    with open(f"data/{formatstr}_{func_class_name}_dim={N}.dat", "w") as file:
        for func in funcs:
            if ashex:
                line = func if isinstance(func, str) else bin2hex(func)
            else:
                line = func
            file.write(str(line) + "\n")


def dual(func):
    N = int(math.log2(len(func)))
    return [int(not func[~a]) for a in range(1 << N)]


def selfdual(func):
    N = int(math.log2(len(func)))
    func = list(func)
    dvals = [int(not func[~a]) for a in range(1 << N)]
    return func + dvals


def classify_boolean_functions(N, skip=True):
    """

    Parameters
    ----------
    N : int
        the dimension of hypercube to analyze
    """
    D = 1 << N
    tfuncs = []
    nfuncs = []
    for k in range(1 << (D - 1)):
        func = dec2bin(k, D)
        tstatus = check_threshold(func)
        nstatus = check_strong_neutralizability(func) if tstatus else False
        if tstatus:
            tfuncs.append(func)
        if nstatus:
            nfuncs.append(func)
        print(f"function {k}: threshold function {tstatus}  neutralizable {nstatus}")

    neg_tfuncs = []
    neg_nfuncs = []
    for func in reversed(tfuncs):
        neg_tfuncs.append([0 if a else 1 for a in func])
    for func in reversed(nfuncs):
        neg_nfuncs.append([0 if a else 1 for a in func])

    tfuncs += neg_tfuncs
    nfuncs += neg_nfuncs
    print(f"\nREPORT\n------")
    print(f"  {1 << D} boolean functions examined")
    print(f"  {len(tfuncs)} threshold functions found")
    print(f"  {len(nfuncs)} neutralizable functions found")

    write_funcs(tfuncs, dim=N, func_class_name="threshold_functions", ashex=True)
    write_funcs(
        nfuncs,
        dim=N,
        func_class_name="strongly_neutralizable_threshold_functions",
        ashex=True,
    )
    write_funcs(tfuncs, dim=N, func_class_name="threshold_functions", ashex=True)
    write_funcs(
        nfuncs,
        dim=N,
        func_class_name="strongly_neutralizable_threshold_functions",
        ashex=False,
    )

    return tfuncs, nfuncs


def reduce_classes_of_boolean_functions(N):
    nfuncs = []
    oldnfunc = set([])
    for i in range(N + 1):
        _, newnfunc = classify_boolean_functions(i)
        newnfunc = [tuple(func) for func in newnfunc if func[0] == 0]
        if i == 0:
            nfuncs.append(set(newnfunc))
        else:
            reduced = set(newnfunc)
            # cut out all self duals of lower dimensional functions
            reduced = reduced.difference(
                set(tuple(selfdual(func)) for func in oldnfunc)
            )
            # cut out all duals of other dimension i functions
            numduals = 0
            for func in reduced:
                dfunc = tuple(dual(func))
                if func != dfunc:
                    numduals += 1
                    print(f"Found dual #{numduals}:", func, " and ", dfunc)
                    reduced = reduced.difference(set([dfunc]))

            nfuncs.append(reduced)

        oldnfunc = deepcopy(newnfunc)

        write_funcs(
            nfuncs[-1],
            dim=i,
            func_class_name="strongly_neutralizable_threshold_functions_nonredundant",
            ashex=True,
        )


def filter_strongly_neutralizable(funcs):
    return [func for func in funcs if check_strong_neutralizability(func)]


def filter_threshold(funcs):
    return [func for func in funcs if check_threshold(func)]


def check_self_duals_for_strong_neutralizability(N):
    tfuncs, nfuncs = classify_boolean_functions(N)
    tfuncs2 = [BoolCirc(N, func).selfdual().func() for func in tfuncs]
    nfuncs2 = [BoolCirc(N, func).selfdual().func() for func in nfuncs]

    neuts_from_selfdualtcirc = filter_strongly_neutralizable(tfuncs2)
    neuts_from_selfdualncirc = filter_strongly_neutralizable(nfuncs2)

    print(
        f"\nREPORT for {N}-Dim Threshold Functions\n----------------------------------------"
    )
    print(f"  {len(tfuncs)} self-duals examined")
    print(f"  {len(filter_threshold(tfuncs2))} threshold functions found")
    print(f"  {len(neuts_from_selfdualtcirc)} neutralizable functions found")

    print(
        f"\nREPORT for {N}-Dim Strongly Neutralizable Functions\n------------------------------------------------------"
    )
    print(f"  {len(nfuncs)} self-duals examined")
    print(f"  {len(filter_threshold(nfuncs2))} threshold functions found")
    print(f"  {len(neuts_from_selfdualncirc)} neutralizable functions found")


def check_duals_for_strong_neutralizability(N):
    tfuncs, nfuncs = classify_boolean_functions(N)
    tfuncs2 = [BoolCirc(N, func).dual().func() for func in tfuncs]
    nfuncs2 = [BoolCirc(N, func).dual().func() for func in nfuncs]

    neuts_from_selfdualtcirc = filter_strongly_neutralizable(tfuncs2)
    neuts_from_selfdualncirc = filter_strongly_neutralizable(nfuncs2)

    print(
        f"\nREPORT for {N}-Dim Threshold Functions\n----------------------------------------"
    )
    print(f"  {len(tfuncs)} duals examined")
    print(f"  {len(filter_threshold(tfuncs2))} threshold functions found")
    print(f"  {len(neuts_from_selfdualtcirc)} neutralizable functions found")

    print(
        f"\nREPORT for {N}-Dim Strongly Neutralizable Functions\n------------------------------------------------------"
    )
    print(f"  {len(nfuncs)} duals examined")
    print(f"  {len(filter_threshold(nfuncs2))} threshold functions found")
    print(f"  {len(neuts_from_selfdualncirc)} neutralizable functions found")


reduce_classes_of_boolean_functions(5)
# check_duals_for_strong_neutralizability(4)
