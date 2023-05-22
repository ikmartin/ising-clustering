from ising import IMul, PICircuit
from spinspace import Spin, Spinspace
from ortools.linear_solver import pywraplp


def test_set_aux(circuit, aux):
    circuit.set_aux(aux)
    for s in circuit.inspace:
        print(
            f"input {s} has aux {circuit.faux(s).spin()} which matches {aux[0][s.asint()]}: {circuit.faux(s).asint() == aux[0][s.asint()]}"
        )
        if circuit.faux(s):
            circuit.faux(s).asint() == aux[0][s.asint()]


if __name__ == "__main__":
    circuit = IMul(2, 2)
    with open("data/all_feasible_MUL_2x2x1.dat") as file:
        tally = 0
        for line in file:
            aux = [json.loads(line)[0]]
            print(aux)
            test_aux_array(aux)
