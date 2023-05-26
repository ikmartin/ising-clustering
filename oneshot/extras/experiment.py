from ising import IMul
from spinspace import Spin, dist
from oneshot import MLPoly, generate_polynomial, FGBZ_FD
from typing import Callable
from polyfit import search_polynomial
import numpy as np

import networkx as nx
from pyvis.network import Network

def multiply_hamming_loss(a: int, b: int) -> Callable:
    circuit = IMul(a, b)
    n = a + b

    def loss(state: tuple) -> float:
        ising_state = np.array([1 if s else -1 for s in state])
        spin = Spin(ising_state, (n, n))
        instate, outstate = spin.splitspin()
        correct_spin = circuit.inout(Spin(instate, (a,b)))
        return dist(spin, correct_spin)

    return loss

def hJ(poly: MLPoly):
    n = poly.num_variables()
    c = poly.coeffs
    h = np.array([poly.get_coeff((i,)) for i in range(n)])
    J = np.array([[poly.get_coeff(tuple(sorted((i,j)))) if i != j else 0 for i in range(n)] for j in range(n)])
    return h, J

def interaction_graph(poly: MLPoly):
    G = nx.Graph()
    for i in range(poly.num_variables()):
        G.add_node(str(i))
    for key, value in poly.coeffs.items():
        if len(key) == 2:
            G.add_edge(str(key[0]), str(key[1]), weight=abs(value), color='red' if value < 0 else 'blue')

    return G


def simulate(poly, instate, loss):
    n = poly.num_variables()
    state = np.random.choice([0,1], n)
    state[0:(a+b)] = instate
    temperature = 1e-7


    while True:
        for i in range(1000):
            target = np.random.randint(n)
            flip_pattern = np.zeros(n)
            flip_pattern[target] = 1
            newstate = np.mod(state + flip_pattern, 2)
            H_diff = poly(tuple(newstate)) - poly(tuple(state))
            if temperature == 0:
                if H_diff < 0:
                    state = newstate
            elif np.random.uniform() < np.exp(-H_diff/temperature):
                state = newstate
            state[0:(a+b)] = instate

        display = ''.join(['#' if s else '_' for s in state])
        print(f'{display} {poly(tuple(state))} {loss(state[:(2*(a+b))])}')

        

a, b = 2,2
loss = multiply_hamming_loss(a, b)
#poly = generate_polynomial(loss, (a+b)*2)

circuit = IMul(a,b)
original_poly = search_polynomial(circuit)
original_poly.clean(threshold = 1e-7)

print(f"Polynomial form of Hamming loss for {a}x{b} multiply:")
print(original_poly)
print("")

poly = FGBZ_FD(original_poly)
print(f"Quadratic form:")
print(poly)
h, J = hJ(poly)
print(h)
print(J.astype(int))
G = interaction_graph(poly)
net = Network(notebook = True)
net.from_nx(G)
net.show('example.html')

instate = np.array([1,1,1,1])

simulate(poly, instate, loss)

