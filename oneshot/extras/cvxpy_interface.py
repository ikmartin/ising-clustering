import cvxpy as cp
from solver import build_solver
from fast_constraints import fast_constraints
from ising import IMul
from time import perf_counter as pc

degree = 4
circuit = IMul(4,4)
M, keys = fast_constraints(circuit, degree)

print(M)

start = pc()
x = cp.Variable(M.shape[1])
obj = cp.Minimize(cp.norm(x,1))
constraints = [ M @ x >= 1 ]
prob = cp.Problem(obj, constraints)
prob.solve(solver = cp.GLOP)
res = x.value
end = pc()
print(f'CVXPY GLOP backend took {end-start}')
#print(x.value)

start = pc()
solver, vars = build_solver(M.to_sparse(), keys)
status = solver.Solve()
res = [var.solution_value() for var in vars]
end = pc()
print(f'Direct GLOP took {end-start}')
#print(res)
