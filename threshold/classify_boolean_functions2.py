from boolfunc import BoolFunc
from ising import BCircuit as BoolCirc
from copy import deepcopy
import math

from itertools import filterfalse

####################
# BASE CONVERSIONS
####################


def dec2bin(num, fill):
    return list(int(a) for a in bin(num)[2:].zfill(fill))


def bin2dec(blist):
    N = len(blist)
    return sum([1 << (N - i - 1) for i in range(N) if blist[i]])


def bin2hex(b):
    return "".join(
        [
            format(int("".join([str(int(x)) for x in b[i: (i + 4)]]), 2), "x")
            for i in range(0, len(b), 4)
        ]
    )


def hex2bin(num, fill):
    return dec2bin(int(num, 16), fill)


def int2hex(num, fill):
    return hex(num)[2:].zfill(fill)

################################
# FUNCTION CLASS CHECKS
################################


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

##############################
# OPERATIONS ON FUNCTIONS
##############################


def dual(func):
    N = int(math.log2(len(func)))
    return [int(not func[~a]) for a in range(1 << N)]


def selfdual(func):
    N = int(math.log2(len(func)))
    func = list(func)
    dvals = [int(not func[~a]) for a in range(1 << N)]
    return func + dvals


##################################
# READING WRITING
##################################

def filename_builder(dim, func_class_name, ashex=True):
    formatstr = "HEX" if ashex else "BOOL"
    return f"data/{formatstr}_signed_hyperoctahedral_orbits_of_{func_class_name}_dim={dim}.dat"


def write_funcs(funcs, dim, func_class_name, ashex=True):
    filename = filename_builder(dim, func_class_name, ashex)
    with open(filename, "w") as file:
        for func in funcs:
            if ashex:
                line = func if isinstance(func, str) else bin2hex(func)
            else:
                line = func
            file.write(str(line) + "\n")


def read_funcs(dim, func_class_name, ashex=True):
    import json
    filename = filename_builder(dim, func_class_name, ashex)
    tfuncs = []
    with open(filename, "r") as file:
        for line in file:
            tfuncs.append(int(line, 16))

    return tfuncs 

def classify_threshold_functions(dim):
    """Classifies all threshold functions of the given dimension."""
    D=1 << dim  # the size of the input space
    hexfill=math.ceil(math.log(1 << D, 16))

    # create function pool
    print(f"Creating function pool with {1 << (D - 1)} elements...", end="")
    function_pool=list(range(1 << (D - 1)))
    print("done")
    # initialize tally and array of threshold functions
    tfuncs=[]  # the threshold functions
    tally=0
    while len(function_pool) > 0:
        index=function_pool.pop()
        bfunc=BoolFunc(dim, index)
        orbit=bfunc.signed_orbit()
        tstatus=check_threshold(bfunc.vals)
        if tstatus:
            orbit_size=len(orbit)
            print(f"New threshold function orbit found: length {orbit_size}")
            tfuncs.append(int2hex(index, hexfill))
            tally += orbit_size

        for x in orbit:
            try:
                function_pool.remove(x)
            except ValueError:
                pass


    print(f"\nREPORT\n------")
    print(f"  {1 << D} boolean functions examined")
    print(f"  {tally} threshold functions found")
    print(f"  {len(tfuncs)} unique threshold functions orbits found")

    write_funcs(tfuncs, dim=dim,
                func_class_name="threshold_functions", ashex=True)

    return tfuncs

def classify_threshold_functions_efficient(dim):
    """Classifies all threshold functions of the given dimension."""
    D=1 << dim  # the size of the input space
    hexfill=math.ceil(math.log(1 << D, 16))

    # create function pool
    print(f"Creating function pool with {1 << (D - 1)} elements...", end="")
    poolsize = 1 << (D - 1)
    print("done")
    # initialize tally and array of threshold functions
    tfuncs=[]  # the threshold functions
    tally=0 
    minreps = set([])
    checked = 0
    for index in range(poolsize):
        if index % 100 == 0:
            print(f"Checked {checked/poolsize * 100}% of bfuncs. Minreps is size {len(minreps)}")
        bfunc = BoolFunc(dim, index)
        orbit = bfunc.signed_orbit()
        orbit_size = len(orbit)
        minrep = min(orbit)
        checked += orbit_size
        if minrep in minreps:
            continue

        minreps.add(minrep)
        tstatus = check_threshold(bfunc.vals)
        if tstatus:
            orbit_size=len(orbit)
            tally += orbit_size
            tfuncs.append(int2hex(index, hexfill))
            print(f" Length {orbit_size} found, {tally} total")


    print(f"\nREPORT\n------")
    print(f"  {1 << D} boolean functions examined")
    print(f"  {tally} threshold functions found")
    print(f"  {len(tfuncs)} unique threshold functions orbits found")

    write_funcs(tfuncs, dim=dim,
                func_class_name="threshold_functions", ashex=True)

    return tfuncs

def classify_strongly_neutralizable_functions(dim, ashex=True):
    """Classifies all threshold functions of the given dimension."""
    D=1 << dim  # the size of the input space
    hexfill=math.ceil(math.log(1 << D, 16))
    nfuncs=[]  # the threshold functions

    # the functions left to check
    function_pool = set(read_funcs(dim, "threshold_functions", ashex))
    tally=0

    while len(function_pool) > 0:
        index=function_pool.pop()
        bfunc=BoolFunc(dim, index)
        orbit=bfunc.signed_orbit()
        nstatus=check_strong_neutralizability(bfunc.vals)
        print(f"Function index {index} : {nstatus} ")
        if nstatus:
            orbit_size=len(orbit)
            print(f"length {orbit_size} found, {tally} total.")
            nfuncs.append(int2hex(index, hexfill))
            tally += orbit_size

        function_pool=function_pool.difference(orbit)

    print(f"\nREPORT\n------")
    print(f"  examined pool of dimension {dim} threshold functions")
    print(f"  {tally} strongly neutralizable functions found")
    print(f"  {len(nfuncs)} unique strongly neutralizable function orbits found")

    write_funcs(nfuncs, dim=dim,
                func_class_name="SNT_functions", ashex=True)

    return nfuncs


if __name__ == "__main__":
    dim = 5
    classify_threshold_functions_efficient(dim)
    #classify_strongly_neutralizable_functions(dim)
