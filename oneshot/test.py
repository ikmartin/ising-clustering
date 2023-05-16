from constraints import get_constraint_matrix
from ising import IMul

circuit = IMul(2,2)
print(get_constraint_matrix(circuit, 2))
