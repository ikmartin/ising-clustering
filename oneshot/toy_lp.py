from ortools.linear_solver.pywraplp import Solver

solver = Solver.CreateSolver("GLOP")
inf = solver.infinity()

x = solver.NumVar(-inf, inf, 'x')
y = solver.NumVar(-inf, inf, 'y')
r1 = solver.NumVar(0, inf, 'r1')
r2 = solver.NumVar(0, inf, 'r2')
r3 = solver.NumVar(0, inf, 'r3')

# x + y + r1 >= 1
# -x - y + r2 >= 1
constraint1 = solver.Constraint(1, inf)
constraint1.SetCoefficient(x, 1)
constraint1.SetCoefficient(y, 1)
constraint1.SetCoefficient(r1, 1)
constraint2 = solver.Constraint(1, inf)
constraint2.SetCoefficient(x, -1)
constraint2.SetCoefficient(y, 0)
constraint2.SetCoefficient(r2, 1)
constraint3 = solver.Constraint(1, inf)
constraint3.SetCoefficient(x, 1)
constraint3.SetCoefficient(y, -1)
constraint3.SetCoefficient(r3, 1)

objective = solver.Objective()
objective.SetCoefficient(r1, 1)
objective.SetCoefficient(r2, 1)
objective.SetCoefficient(r3, 1)
objective.SetMinimization()

status = solver.Solve()

r_vals = [x.solution_value(), y.solution_value(), r1.solution_value(), r2.solution_value(), r3.solution_value()]
print(r_vals)



