import ising

def AND_test():
    AND = ising.AND()
    solver = AND.build_solver()
    status = solver.Solve()
    print(status)
    print(status == solver.OPTIMAL)

def check_single_aux(circuit, aux):
    #print(f"aux = {aux}")
    circuit.set_aux(aux)
    solver = circuit.build_solver()
    status = solver.Solve()
    print(solver.NumConstraints())
    #print("NumConstraints:", solver.NumConstraints())
    #print(f"Status: {status}")
    print(f"aux = {aux}  Solvable? ", status == solver.OPTIMAL, "\n")

def MUL2x2_test():
    circuit = ising.IMul(2,3)
    
    import json

    with open('data/all_feasible_MUL_2x3x1.dat') as file:
        tally = 0
        for line in file:
            tally += 1
            aux = [json.loads(line)[0]]
            check_single_aux(circuit, aux)



if __name__ == "__main__":
    MUL2x2_test()
    #AND_test()

