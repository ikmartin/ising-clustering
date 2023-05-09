from polyfit import search_polynomial
from ising import IMul
from oneshot import FGBZ_FD

circuit = IMul(3,3)
poly = search_polynomial(circuit)
print(poly)
print("")
poly = FGBZ_FD(poly)
print(poly)
