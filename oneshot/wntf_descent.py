from new_constraints import constraints  
from mysolver_interface import call_my_solver
from torch import tensor
from tqdm import tqdm
from itertools import combinations
from random import choices
import torch
import click

@click.command()
@click.option('--n1', default = 3)
@click.option('--n2', default = 3)
@click.option('--a', default = 3)
def main(n1, n2, a):
    A = a
    print(f'{n1} {n2} {A}')
    cache = dict()

    def rho(n1, n2, i, g):
        M, _, correct = constraints(n1, n2, radius=i, hyperplanes = g)
        hash_key = f'rad{i}' + str(correct.tolist())
        if hash_key in cache:
            return cache[hash_key]
        
        obj = call_my_solver(M.to_sparse_csc())
        cache[hash_key] = obj
        return obj


    hyperplanes = []
    with open(f"../oneshot/dat/WNTF{n1}{n2}.dat", "r") as FILE:
        for line in FILE.readlines():
            w, b = eval(line)

            hyperplanes.append((tensor(w), b))
        """
        line_accumulator = ""
        for line in FILE.readlines():
            if line.startswith("tensor") and len(line_accumulator) > 0:
                hyperplanes.append((eval(line_accumulator),0))
                line_accumulator = ""

            line_accumulator += line
        """

    num_base_vars = (n1+n2)*2
    for indices in combinations(range(num_base_vars), 3):
        t = torch.zeros(num_base_vars)
        for j in indices:
            t[j] = 1
        hyperplanes.append((t, 1.5))
    for indices in combinations(range(num_base_vars), 2):
        t = torch.zeros(num_base_vars)
        for j in indices:
            t[j] = 1
        hyperplanes.append((t, 1.5))

    print(f'library size = {len(hyperplanes)}')
    max_r = n1+n2

    def run():
        g = choices(hyperplanes, k=A)
        S = set(range(A))
        r = 1
        cur_obj = rho(n1, n2, r, g)
        while True:
            if len(S) == 0:
                print('failure')
                return

            eta = [(i, rho(n1, n2, r, list(set(g) - set([g[i]]))) - rho(n1, n2, r, g)) for i in S]
            j = min(eta, key = lambda pair:pair[1])[0]

            for pair in eta:
                print(f'bit {pair[0]} : {pair[1]:.1f}')
            loop = tqdm(hyperplanes, leave=True, desc = f'r {r} bit {j}')
            found = False
            for plane in loop:
                cur_aux = g[j]
                g[j] = plane
                new_rho = rho(n1, n2, r, g)
                if new_rho >= cur_obj:
                    g[j] = cur_aux
                else:
                    found = True
                    cur_obj = new_rho

                if cur_obj < 1:
                    if r == max_r:
                        print('success')
                        print(g)
                        return
                    r += 1 
                    cur_obj = rho(n1, n2, r, g)
                
                loop.set_postfix(obj = cur_obj, r=r)

            if not found:
                S = S - {j}
            else:
                S = set(range(A))

    while True:
        run()


if __name__ == "__main__":
    main()