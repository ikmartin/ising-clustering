from ising import BCircuit as BoolCirc


def dec2bin(num, fill):
    return list(bool(int(a)) for a in bin(num)[2:].zfill(fill))


def bin2dec(blist):
    return sum([1 << i if k else 0 for i, k in enumerate(blist)])


def classify_boolean_functions(N):
    D = 1 << N
    funcs = [dec2bin(k, D) for k in range(1 << D)]
    goodfuncs = []
    tcount = 0
    ncount = 0
    for k, func in enumerate(funcs):
        circuit = BoolCirc(N, func)
        tsolver = circuit.build_solver()
        nsolver = circuit.neutralizable_solver()
        tstatus = tsolver.Solve() == tsolver.OPTIMAL
        nstatus = nsolver.Solve() == nsolver.OPTIMAL
        if tstatus:
            tcount += 1
        if nstatus:
            ncount += 1
            goodfuncs.append(func)
        print(f"function {k}: threshold function {tstatus}  neutralizable {nstatus}")

    print(f"\nREPORT\n------")
    print(f"  {len(funcs)} boolean functions examined")
    print(f"  {tcount} threshold functions found")
    print(f"  {ncount} neutralizable functions found")
    with open(f"data/neutralizable_threshold_functions_dim={N}.dat", "w") as file:
        for func in goodfuncs:
            file.write(str(func) + "\n")


classify_boolean_functions(2)
