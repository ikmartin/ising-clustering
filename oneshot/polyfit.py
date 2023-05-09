from ising import PICircuit
from spinspace import Spin
from oneshot import MLPoly
from itertools import combinations, chain, product
from math import prod
from ortools.linear_solver.pywraplp import Solver
import numpy as np

def search_polynomial(circuit: PICircuit) -> MLPoly:
    """
    Searches for the lowest degree multilinear polynomial in the original spinspace (inputs/outputs only with no auxilliaries) that satisfies the constraint set. The idea is just to iterate through degrees from 2 upwards, attempting to fit a polynomial of each degree by representing the fitting problem as a linear programming problem in the tensor algebra (the original method of finding h, J that satisfy the constraint set is simply the degree-2 special case). 

    We work without auxilliary spins. The idea is that instead of adding auxilliaries to make the degree 2 fitting problem possible, we will instead find the minimum d such that the degree d fitting problem is possible, then apply quadritization algorithms to add the auxilliaries.
    """

    # Number of variables in the domain space
    num_variables = circuit.N + circuit.M

    for degree in range(2, num_variables):
        solver, status, params = build_polynomial_fitter(circuit, degree)
        print(f'degree={degree} status={status}')
        if status == 2:
            continue 

        coeffs = {key: var.solution_value() for key, var in params.items()}
        return MLPoly(coeffs = coeffs)

def binary_spin(spin: Spin):
    binstring = np.binary_repr(spin.asint()).zfill(spin.dim())
    return np.array([int(b) for b in binstring]).astype(float)
    
def prod_term(array: np.ndarray, key: tuple):
    return prod([array[i] for i in key])
    
def build_polynomial_fitter(circuit: PICircuit, degree: int) -> Solver:
    """
    Builds a linear programming solver for fitting a multilinear polynomial of a given degree to the constraint system of the input/output pairs of the given circuit.

    The optimization objective is set to be the average of all the rows in the constraint matrix (an idea due to Teresa)
    """

    solver = Solver.CreateSolver("GLOP")
    inf = solver.infinity()

    num_variables = circuit.N + circuit.M

    variable_keys = chain.from_iterable([combinations(range(num_variables), i) for i in range(1, degree+1)])
    params = {key: solver.NumVar(-inf, inf, str(key))
            for key in variable_keys}
    
    """
    old constraint method: enforce that the correct answer be the global min of the input level

    for inspin, outspin in product(circuit.inspace, circuit.outspace):
        if circuit.fout(inspin) == outspin:
            continue

        constraint = solver.Constraint(1e-5, inf)
        correct_array = binary_spin(circuit.inout(inspin))
        wrong_array = binary_spin(Spin.catspin(spins = (inspin, outspin)))
        
        for key, var in params.items():
            constraint.SetCoefficient(var, prod_term(wrong_array, key) - prod_term(correct_array, key))
    """

    # New constraint method: add a new constraint for each edge of the hypercube representing the possible output spins, thus enforcing convexity on the Hamiltonian restricted to this input level.
    for inspin in circuit.inspace:
        correct_out = circuit.fout(inspin)
        correct_out_array = correct_out.spin()

        for outspin in circuit.outspace:
            out_array = outspin.spin()

            # for each output state, we want to enforce that the hamiltonian be less than the other output states which are one further away from the correct output in Hamming distance. Thus we iterate through the bits of the output sate, and for each one that is equal to the correct output, we flip it and add a constraint. This is because only flipping a bit that is equal to the correct output will move us one further array in Hamming distance.
            for i, s in enumerate(out_array):
                if s == correct_out_array[i]:
                    constraint = solver.Constraint(1e-5, inf)

                    current_binary = binary_spin(Spin.catspin(spins = (inspin, outspin)))
                    other_spin_array = out_array.copy()
                    other_spin_array[i] *= -1
                    other_spin = Spin(other_spin_array, shape=outspin.shape)
                    other_binary = binary_spin(Spin.catspin(spins = (inspin, other_spin)))
                    for key, var in params.items():
                        constraint.SetCoefficient(var, prod_term(other_binary, key) - prod_term(current_binary, key))

    
    """
    objective = solver.Objective()
    for key, var in params.items():
        if len(key) > 2:
            objective.SetCoefficient(var, -1)

    objective.SetMaximization()
    """

    status = solver.Solve()
    return solver, status, params



    

