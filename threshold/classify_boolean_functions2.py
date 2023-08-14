from numpy import dtype
from boolfunc import BoolFunc
import torch
import random
from ising import BCircuit as BoolCirc
from copy import deepcopy
import math

from itertools import filterfalse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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


def write_funcs(funcs, dim, func_class_name, ashex=True, mode="a"):
    filename = filename_builder(dim, func_class_name, ashex)
    with open(filename, mode) as file:
        for func in funcs:
            if ashex:
                line = func if isinstance(func, str) else bin2hex(func)
            else:
                line = func
            file.write(line + "\n")


def read_funcs(dim, func_class_name, ashex=True):
    import json, os.path
    filename = filename_builder(dim, func_class_name, ashex)
    if os.path.isfile(filename) == False:
        return []
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
    function_pool=set(range(1 << (D - 1)))
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
            write_funcs([int2hex(index, hexfill)], dim=dim,
                        func_class_name="threshold_functions", ashex=True)
            tfuncs.append(int2hex(index, hexfill))
            tally += orbit_size

        function_pool = function_pool.difference(orbit)


    print(f"\nREPORT\n------")
    print(f"  {1 << D} boolean functions examined")
    print(f"  {tally} threshold functions found")
    print(f"  {len(tfuncs)} unique threshold functions orbits found")

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
        if minrep in minreps:
            continue

        checked += orbit_size
        minreps.add(minrep)
        tstatus = check_threshold(bfunc.vals)
        if tstatus:
            orbit_size=len(orbit)
            print(f"New threshold function orbit found: length {orbit_size}")
            write_funcs([int2hex(index, hexfill)], dim=dim,
                        func_class_name="threshold_functions", ashex=True)
            tfuncs.append(int2hex(index, hexfill))
            tally += orbit_size


    print(f"\nREPORT\n------")
    print(f"  {1 << D} boolean functions examined")
    print(f"  {tally} threshold functions found")
    print(f"  {len(tfuncs)} unique threshold functions orbits found")

    return tfuncs

def d2b(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def all_answers(dim):
    return d2b(torch.arange(2**dim).to(DEVICE), dim)

# samples random threshold function from hyperplane
def getsample(dim, mean=0, std=1, minb=-1, maxb=1, batch=1000):
    X = all_answers(dim)
    W = torch.normal(mean, std, size=(dim, batch))
    b = random.uniform(minb,maxb)
    A = torch.matmul(X,W)
    A = torch.unique(A > b, dim=1)
    A = torch.transpose(A.int(), 0, 1)
    return A.tolist()

def test_sampling(dim):
    batch = 1000
    statuses = []
    samples = getsample(dim, batch=batch)
    tcount = 0
    for func in samples:
        print(func)
        bfunc = BoolFunc.from_func(func)
        statuses.append(check_threshold(bfunc.vals))
        tcount += statuses[-1]

    print(f"T = {tcount},  F = {len(samples) - tcount}")

def sample_threshold_functions(dim, batch_size=1000, maxiter=100):
    OEIS = [2, 4, 14, 104, 1882, 94572, 15028134, 8378070864, 17561539552946, 144130531453121108]

    # hyperparameters
    D=1 << dim  # the size of the input space
    hexfill=math.ceil(math.log(1 << D, 16))

    # loop
    loopcount = 0
    minreps = set(read_funcs(dim=dim, func_class_name="threshold_functions", ashex=True))
    # calculate initial tally
    tally = 0
    for i,index in enumerate(minreps):
        if i % 10:
            print(f"{i} out of {len(minreps)} tallied")
        tally += len(BoolFunc(dim, index).signed_orbit())

    print(f"CURRENTLY HAVE {tally} out of {OEIS[dim]} {math.ceil(tally/OEIS[dim] * 10000)/100} % of dim = {dim} threshold functions.\n")
    while OEIS[dim] > tally and loopcount < maxiter:
        # update loop count
        loopcount += 1
        print(f"Sample {loopcount}")

        # acquire samples
        samples = getsample(dim, batch=batch_size)
        for func in samples:
            bfunc = BoolFunc.from_func(func)
            orbit = bfunc.signed_orbit()
            orbit_size = len(orbit)
            minrep = min(orbit) 
            print(f"  [{int2hex(minrep, hexfill)}] orbit size {orbit_size}...", end="")
            # check if already found orbit
            if minrep in minreps:
                print("is a REPEAT.")
                continue

            print("is NEW!")
            minreps.add(minrep)
            tally += orbit_size

            write_funcs([int2hex(minrep, hexfill)], dim=dim,
                        func_class_name="threshold_functions", ashex=True)

            # stop check if finished
            if tally >= OEIS[dim]:
                break

        print(f"BATCH {loopcount} REPORT: \n  {tally} out of {OEIS[dim]} {math.ceil(tally/OEIS[dim] * 10000)/100} % of dim = {dim} threshold functions found.\n")

    print(f"\nREPORT\n------")
    print(f"  {tally} threshold functions found")
    print(f"  {len(minreps)} unique threshold functions orbits found")

    return minreps

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
    dim = 7
    batch = 20
    maxiter = 300
    #classify_threshold_functions(dim)
    #classify_threshold_functions_efficient(dim)
    #classify_strongly_neutralizable_functions(dim)
    sample_threshold_functions(dim,batch, maxiter)
    classify_strongly_neutralizable_functions(dim)
