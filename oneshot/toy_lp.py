from ortools.linear_solver.pywraplp import Solver

solver = Solver.CreateSolver("GLOP")
inf = solver.infinity()

x = solver.NumVar(-inf, inf, 'x')
y = solver.NumVar(-inf, inf, 'y')

# x + y >= 1
# -x - y >= 1



