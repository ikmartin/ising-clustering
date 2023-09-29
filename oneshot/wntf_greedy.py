from new_constraints import constraints  
from mysolver_interface import call_my_solver
from torch import tensor
from tqdm import tqdm


n1, n2 = 3,4 
hyperplanes = []
with open(f"dat/WNTF{n1}{n2}.dat", "r") as FILE:
    for line in FILE.readlines():
        hyperplanes.append((tensor(eval(line)), 0))
    """
    line_accumulator = ""
    for line in FILE.readlines():
        if line.startswith("tensor") and len(line_accumulator) > 0:
            hyperplanes.append((eval(line_accumulator),0))
            line_accumulator = ""

        line_accumulator += line
    """

print(hyperplanes)
current_planes = []

while True:
    plane_scores = []

    best_obj = 1e6
    loop = tqdm(hyperplanes, leave=True)
    for plane in loop:
        M, _, correct = constraints(n1, n2, hyperplanes = current_planes + [plane])
        objective = call_my_solver(M.to_sparse_csc())

        if objective < best_obj:
            best_obj = objective
        
        if objective < 1:
            print('done')
            print(current_planes + [plane])

        plane_scores.append({'plane': plane, 'objective': objective})
        loop.set_postfix(obj = best_obj)

    plane_dict = min(plane_scores, key = lambda pair: pair['objective'])
    current_planes.append(plane_dict['plane'])
    print(f"added {plane_dict['plane']}")
    print(f"current objective {plane_dict['objective']}")
