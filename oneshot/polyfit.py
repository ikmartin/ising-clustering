from ising import PICircuit
from spinspace import Spin
from oneshot import MLPoly, reduce_poly
from itertools import combinations, chain, product
from math import prod
from ortools.linear_solver.pywraplp import Solver
from solver import build_solver
from fast_constraints import fast_constraints
import numpy as np

def search_polynomial(circuit: PICircuit, weak = False) -> MLPoly:
    """
    Searches for the lowest degree multilinear polynomial in the original spinspace (inputs/outputs only with no auxilliaries) that satisfies the constraint set. The idea is just to iterate through degrees from 2 upwards, attempting to fit a polynomial of each degree by representing the fitting problem as a linear programming problem in the tensor algebra (the original method of finding h, J that satisfy the constraint set is simply the degree-2 special case). 

    We work without auxilliary spins. The idea is that instead of adding auxilliaries to make the degree 2 fitting problem possible, we will instead find the minimum d such that the degree d fitting problem is possible, then apply quadritization algorithms to add the auxilliaries.
    """

    # Number of variables in the domain space
    num_variables = circuit.G

    for degree in range(2, num_variables):
        #solver, status, params = build_polynomial_fitter(circuit, degree, weak)
        #print(f'degree={degree} status={status}')
        #if degree > 2:
        #    degree = 3
        M, keys = fast_constraints(circuit, degree)
        solver, variables, bans = build_solver(M, keys)
        status = solver.Solve()
        if status == 2:
            print(f'{degree} failed')
            continue 

        coeffs = {key: var.solution_value() for key, var in zip(keys, variables)}
        poly = MLPoly(coeffs = coeffs)
        poly.clean(threshold = 0.1)
        return poly

def tryall(circuit: PICircuit, weak = False) -> MLPoly:
    """
    Searches for the lowest degree multilinear polynomial in the original spinspace (inputs/outputs only with no auxilliaries) that satisfies the constraint set. The idea is just to iterate through degrees from 2 upwards, attempting to fit a polynomial of each degree by representing the fitting problem as a linear programming problem in the tensor algebra (the original method of finding h, J that satisfy the constraint set is simply the degree-2 special case). 

    We work without auxilliary spins. The idea is that instead of adding auxilliaries to make the degree 2 fitting problem possible, we will instead find the minimum d such that the degree d fitting problem is possible, then apply quadritization algorithms to add the auxilliaries.
    """

    # Number of variables in the domain space
    num_variables = circuit.G

    for degree in range(2, num_variables):
        #solver, status, params = build_polynomial_fitter(circuit, degree, weak)
        #print(f'degree={degree} status={status}')
        M, keys = fast_constraints(circuit, degree)
        solver, variables = build_solver(M, keys)
        status = solver.Solve()
        if status == 2:
            print(f'{degree} failed')
            continue 

        coeffs = {key: var.solution_value() for key, var in zip(keys, variables)}
        poly = MLPoly(coeffs = coeffs)
        poly.clean(threshold = 0.1)
        print(poly)
        print("")

    
def prod_term(array: np.ndarray, key: tuple):
    return prod([array[i] for i in key])
    
def gen_var_keys(circuit, degree):
    dimension = circuit.G
    input_length = circuit.N
    variable_keys = chain.from_iterable([combinations(range(dimension), i) for i in range(1, degree+1)])
    variable_keys = [key for key in variable_keys
                     if max(key) >= input_length]
    return variable_keys
    

def build_polynomial_fitter(circuit: PICircuit, degree: int, weak = False) -> Solver:
    """
    Builds a linear programming solver for fitting a multilinear polynomial of a given degree to the constraint system of the input/output pairs of the given circuit. The objective is l1 minimization since it gives a good approximation of l0 optimization, or at least as good as can be done with linear programming.

    The theory is the following:
    min_x ||x||_1           s.t. Mx >= 1
    <-> min_{x,y} <1,y>     s.t. Mx >= 1 and x-y <= 0 and x+y >= 0
    """

    solver = Solver.CreateSolver("GLOP")
    inf = solver.infinity()


    variable_keys = gen_var_keys(circuit, degree)

    params = {key: 
        solver.NumVar(-inf, inf, str(key))
        for key in variable_keys
    }
    
    #old constraint method: enforce that the correct answer be the global min of the input level
    for inspin in circuit.inspace:
        correct = circuit.inout(inspin).binary()
        for outaux in circuit.allwrong(inspin):
            constraint = solver.Constraint(1, inf)
            wrong = Spin.catspin(spins = (inspin, outaux)).binary() 
            
            for key, var in params.items():
                constraint.SetCoefficient(var, int(prod_term(wrong, key) - prod_term(correct, key)))
    
    # Now that the hypercube constraints on x have been loaded, we need to create the new variables y and the constraints to force y to represent |x|.
    
    y_params = {key: solver.NumVar(-inf, inf, 'y_' + str(key))
            for key in variable_keys}

    for key in variable_keys:
        constraint1 = solver.Constraint(-inf, 0)
        constraint2 = solver.Constraint(0, inf)

        x_var = params[key]
        y_var = y_params[key]
        constraint1.SetCoefficient(x_var, 1)
        constraint1.SetCoefficient(y_var, -1)
        constraint2.SetCoefficient(x_var, 1)
        constraint2.SetCoefficient(y_var, 1)

    # the objective is then set to simply be the 1 vector on the y variables. But we only care about sparsity on the higher-order terms, quadratics can be whatever.
    
    objective = solver.Objective()
    for key, var in y_params.items():
        if len(key) > 2:
            objective.SetCoefficient(var, 1)

    objective.SetMinimization()
    

    status = solver.Solve()
    return solver, status, params



    

