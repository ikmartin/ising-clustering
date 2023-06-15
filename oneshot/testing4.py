from fast_constraints import constraints_basin, fast_constraints, constraints_building
from fittertry import IMulBit
from ising import IMul
from solver import LPWrapper
from guided_lmisr import CircuitFactory
import torch

old_way, keys = constraints_building(3,3, None, 3, mask = [5], include = None)
solver = LPWrapper(keys)
solver.add_constraints(old_way)
solution = solver.solve()
print(solution)
exit()
factory = CircuitFactory(IMul, (3,3))
circuit = factory.get()
old_way, keys = fast_constraints(circuit, 3)

solver = LPWrapper(keys)
solver.add_constraints(old_way)
solution = solver.solve()
print(solution)
