from scipy.fft import dct
from ising import IMul
from fittertry import IMulBit
from polyfit import fit
import numpy as np

circuit = IMulBit(3,3,1)
iopairs = [
    inspin.splitspin()
    for inspin in circuit.inspace
]
iopairs = np.concatenate([
    np.expand_dims(np.concatenate([np.sign(dct(x)) for x in row]), axis = 0)
    for row in iopairs
])

circuit.set_all_aux(iopairs.tolist())

poly = fit(circuit, 2)
print(poly)
