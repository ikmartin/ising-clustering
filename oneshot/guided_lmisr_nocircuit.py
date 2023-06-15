from multiprocessing.managers import SyncManager
from multiprocessing import Process, Lock, freeze_support, Manager
from queue import PriorityQueue
import multiprocessing
from collections import deque
import time, os, torch, click
import numpy as np
from math import comb

from solver import LPWrapper
from oneshot import full_Rosenberg, rosenberg_criterion, get_term_table, Rosenberg, weak_positive_FGBZ_criterion, negative_FGBZ_criterion, PositiveFGBZ, NegativeFGBZ, single_FD

from dataclasses import dataclass, field
from typing import Any

from fast_constraints import fast_constraints, sequential_constraints, constraints_basin, constraints_building
from oneshot import reduce_poly, MLPoly
from ising import IMul
from fittertry import IMulBit

from lmisr_interface import call_solver
from mysolver_interface import call_my_solver

import cProfile


"""
Specifications
========================================================

The idea is to try the alternating fitting/reduction method in a multiprocessed manner in order to explore multiple avenues at once in a more or less efficient manner. Once we have a certain polynomial, we will add all possible new aux arrays that could come from reduction of that polynomial to the task queue. So the task queue is composed entirely of auxilliary arrays in np.ndarray format, along with an estimated priority value.

Overall design
--------------------------------------------------------
Master process: Starts the solvers and the delegator. Adds the initial task. Runs an infinite monitoring loop
    1. Print some information about the current queue sizes
    2. Report best currently known solutions
    3. Sleep for 1 second.

Solver process: Runs an infinite loop with the following steps:
    1. Pull an auxilliary array from the task queue. Check to see if the aux array is at least as long as the currently known minimum. If so, skip this task.
    2. Build a circuit and LP solver. Attempt to fit a quadratic.
        2a. If poly of degree 2 fits, append the solution to the success queue and register the size of the successful auxilliary array with the administrator object.
        2b. Otherwise, append the polynomial and the aux array it was solved with to the done queue, with same priority as the original task.

Delegator process: Runs an infinite loop with the following steps
    1. Pull a polynomial and the aux array used to generate it from the done queue
        1a. Check to see if this aux array is longer than the currently known minimum. If so, skip this task.
    2. Generate a list of new candidate auxilliary vectors by attempting single reductions steps. For each reduction candidate, find the new polynomial. Then score each one with the polynomial scoring heuristic.
    3. For each new auxilliary vector:
        3a. Append it to the current aux array
        3b. If this array is already in dibs, throw it out.
        3c. Add this array and its estimated score to the task queue
        3d. Add this array to dibs.

Admin Object:
    -- Task queue (PriorityQueue):
        (priority, array)    
    -- Done queue (PriorityQueue):
        (priority, array, poly)
    -- Success list (dict)
        {'best': int, 'list': list, 'total': int}
    -- dibs (dict):
        hash({frozenset{tuple(vec) for each vec in array}: None})
        Used for preventing task duplication.
    -- Success list lock

"""

## Helper functions for dealing with polynomials

def get_poly(keys, coeffs) -> MLPoly:
    coeff_dict = {key: val for key, val in zip(keys, coeffs)}
    poly = MLPoly(coeff_dict)
    poly.clean(0.1)
    return poly

def estimate_score(poly) -> int:
    """
    This provides a heuristic to estimate how good a particular intermediate set of coefficients is. The idea is to just try full Rosenberg reduction with the standard heuristic, and use the total number of variables as a quality score. The hope is that this provides a good notion of the complexity of a particular situation.
    """

    return reduce_poly(poly, ['+fgbz', '-fgbz', 'fd']).num_variables() + reduce_poly(poly, ['rosenberg']).num_variables()


## Helper functions for uniquely identifying aux arrays

def permutation_invariant_hash(array):
    """
    Returns a row-permutation invariant hash using frozenset.
    """

    return hash(frozenset({
        tuple(row.tolist())
        for row in array
    }))

def hash_aux_array(array):
    """
    Generates a permutation-invariant and sign-invariant hash of the given aux array. These are the  only currently known symmetries.
    """

    return permutation_invariant_hash(array) + permutation_invariant_hash(1-array)


## Multiprocessing boilerplate for getting a manager equipped with priority queues

class PriorityManager(SyncManager): pass

def get_manager():
    PriorityManager.register("PriorityQueue", PriorityQueue, exposed = ['put', 'get', 'qsize', 'empty'])
    manager = PriorityManager()
    manager.start()
    return manager

@dataclass(order = True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare = False)
    history: Any=field(compare = False)


## Class which is capable of producing many copies of the same circuit, optionally pre-initialized with specified auxilliary arrays.

class ConstraintFactory:
    def __init__(self, n1, n2, mask, include):
        self.n1 = n1
        self.n2 = n2
        self.mask = mask
        self.include = include

    def get(self, aux_array, degree, radius):
        return constraints_building(self.n1, self.n2, aux_array, degree, radius = radius, mask = self.mask, include = self.include)


## A function which makes colored console output. Looks a little nicer.

def log(owner, name, message):
    reset = '\x1b[0m'
    bold = '\033[1m'
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    color = YELLOW
    if owner == 'master':
        color = MAGENTA
    if owner == 'delegator':
        color = CYAN

    if owner == 'master':
        format = f'{bold}[\x1b[{30+color};20m{name}{reset}{bold}] {message}{reset}'
    else:
        format = f'[\x1b[{30+color};20m{name}{reset}] {message}{reset}'

    print(format, flush = True)
        

def search(factory, num_solvers, num_delegators, limit):
    log('master', 'Master', "Creating global manager.")
    manager = get_manager()
    admin = {
        "success_lock": manager.Lock(),
        "task_queue": manager.PriorityQueue(),
        "done_queue": manager.PriorityQueue(),
        "success": manager.dict(),
        "dibs": manager.dict()
    }

    # Initial value: way higher than any actual solution
    admin["success"]["best"] = 1e10
    admin["success"]["list"] = []
    admin["success"]["total"] = 0
    admin["success"]["found"] = False

    log('master', 'Master', "Creating solvers...")
    workers = [
        Solver(admin, factory, limit)
        for i in range(num_solvers)
    ]

    log('master', 'Master', "Creating delegators...")
    delegators = [
        Delegator(admin, factory, limit)
        for i in range(num_delegators)
    ]

    log('master', 'Master', "Initialized.")

    log('master', 'Master', "Starting processes.")
    for worker in workers:
        worker.start()

    for delegator in delegators:
        delegator.start()

    log('master', 'Master', "Setting initial task.")
    admin["task_queue"].put(PrioritizedItem(priority = 0, item = (None, None), history = []))

    log('master', 'Master', "Letting solvers work.")
    done_flag = False
    while True:
        queue_length = admin["task_queue"].qsize()
        done_length = admin["done_queue"].qsize()

        best_score = admin["success"]["best"]
        num_done = admin["success"]["total"]

        log('master', 'Master', f"{num_done} done / {queue_length} ready / {done_length} docket / best {best_score}")


        time.sleep(1.0)

        with admin['success_lock']:
            if admin['success']['found']:
                done_flag = True

        if done_flag:
            log('master', 'Master', 'Quitting...')
            for worker in workers:
                worker.join()

            for delegator in delegators:
                delegator.join()
            
            break


class Delegator(Process):
    def __init__(self, admin, factory, limit):
        self.admin = admin
        self.factory = factory
        self.limit = limit
        self.STABLE_QSIZE = 10000000
        super().__init__()

    def run(self):
        log('delegator', self.name, 'Running.')
        while True:
            with self.admin['success_lock']:
                if self.admin['success']['found']:
                    log('delegator', self.name, 'Quitting...')
                    return
            self.loop()

    def loop(self):
        # If action is not needed or not possible, just wait.
        if self.admin["task_queue"].qsize() > self.STABLE_QSIZE or self.admin["done_queue"].empty():
            return

        # Get a recently completed task and unpack it
        task = self.admin["done_queue"].get()
        array, poly = task.item
        priority = task.priority
        history = task.history

        # Check to see if the current aux array is even with the best final solutions. Since we know that we are about to add a new row to the aux array, if this is the case, then whatever tasks we would add would be guarunteed not to improve on the currently known best answer. Therefore, simply skip.
        with self.admin['success_lock']:
            if array is not None and (array.shape[0] >= self.admin['success']['best'] - 1 or array.shape[0] >= self.limit) and priority > 1:
                log('delegator', self.name, 'Worthless docket item.')
                return

            
        log('delegator', self.name, f'Accepting docket item with priority {task.priority}.')

        """
        if array is None:
            new_history = []
            new_aux_array = array
            circuit = self.factory.get(array)
            for C in [(0,3), (0,5), (2,7), (1,7), (1,0,5), (3,5), (4,7), (6,2,7), (6,3,5)]:
                new_history.append((C, 'ros'))   
                aux_map = MLPoly({C : 1})
                new_aux_vector = np.array([[aux_map(tuple(circuit.inout(inspin).binary())) for inspin in circuit.inspace]])
                new_aux_array = new_aux_vector if new_aux_array is None else np.concatenate([new_aux_array, new_aux_vector])

            new_task = PrioritizedItem(priority = priority, item = (new_aux_array, poly), history = history + new_history)
            self.admin['task_queue'].put(new_task)
            return
        """
        
        # Generate new tasks and add them to the queue
        self.make_new_tasks(poly, array, priority, history)

    def make_new_tasks(self, poly, array, priority, history):

        """
        if priority < 1 and array is not None:
            # This is code for "we almost got it!"
            new_score = estimate_score(poly)
            log('delegator', self.name, 'Exploring variations...')
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    new_array = array.copy()
                    new_array[i, j] = 1 - new_array[i, j]
                    new_task = PrioritizedItem(priority = new_score, item = (new_array, poly))

                    self.admin['task_queue'].put(new_task)
        """


        _, _, correct_rows = self.factory.get(torch.tensor(array) if array is not None else None, 1, radius = 1)

        # table of further options --- should be tuples of (new_poly, aux_map)
        options = []

        rosenberg_term_table = get_term_table(poly, rosenberg_criterion, 2)
       
        """
        pos_fgbz_term_table = get_term_table(poly, weak_positive_FGBZ_criterion)
        neg_fgbz_term_table = get_term_table(poly, negative_FGBZ_criterion)
        fd_options = [
            (key, value)
             for key, value in poly.coeffs.items()
            if value < 0 and len(key) > 2
        ]
        """
        

        for C, H in rosenberg_term_table.items():
            if not len(H):
                continue

            options.append((Rosenberg(poly, C, H), MLPoly({C: 1}), C, 'ros'))

        
        """
        
        for C, H in pos_fgbz_term_table.items():
            if not len(H):
                continue

            options.append((
                PositiveFGBZ(poly, C, H),
                MLPoly({
                    () : 1,
                    C : -1
                }),
                C,
                '+fgbx'
            ))

        for C, H in neg_fgbz_term_table.items():
            if not len(H):
                continue

            options.append((
                NegativeFGBZ(poly, C, H),
                MLPoly({
                    C : 1
                }),
                C,
                '-fgbz'
            ))

        for key, val in fd_options:
            options.append((
                single_FD(poly, key),
                MLPoly({
                    key: 1
                })
            ))
        
        """

        # Printout information objects
        log_priorities = []
        reduction_choices = []
        new_length = array.shape[0] + 1 if array is not None else 1

        for new_poly, aux_map, C, method in options:
            reduction_choices.append(C)
            new_aux_vector = np.array([[aux_map(tuple(correct_rows[i])) for i in range(correct_rows.shape[0])]])
            
            new_aux_array = new_aux_vector if array is None else np.concatenate([array, new_aux_vector])

            new_array_id = hash_aux_array(new_aux_array)

            if new_array_id in self.admin['dibs']:
                continue

            self.admin['dibs'][new_array_id] = True
            new_priority = estimate_score(new_poly) + 2 * new_length

            new_task = PrioritizedItem(priority = new_priority, item = (new_aux_array, new_poly), history = history + [(C, method)])

            self.admin['task_queue'].put(new_task)
            log_priorities.append(new_priority)

        log_priorities = sorted(log_priorities)
        log('delegator', self.name, f'Poly : {poly}')
        log('delegator', self.name, f'Reduction choices: {reduction_choices}')
        log('delegator', self.name, f'Added new tasks (length = {new_length}): priorities {log_priorities}')

class Solver(Process):
    def __init__(self, admin, factory, limit):
        self.admin = admin
        self.factory = factory
        self.limit = limit
        
        super().__init__()


    def run(self):
        log('solver', self.name, 'Running.')
        while True:
            with self.admin['success_lock']:
                if self.admin['success']['found']:
                    log('solver', self.name, 'Quitting...')
                    return
            self.loop()

    def loop(self):
        # Obtain a task from the queue, unpack the aux array, and generate the correct circuit
        task = self.request_task()
        if task is None:
            return

        array, poly = task.item
        priority = task.priority
        history = task.history
        
        # Check to see if this task is worth doing at all
        with self.admin['success_lock']:
            if array is not None and (array.shape[0] >= self.admin['success']['best'] or array.shape[0] > self.limit):
                log('solver', self.name, f'Worthless task (priority {priority}), skipping.')
                return
        
        length = 0 if array is None else array.shape[0]
        log('solver', self.name, f'Accepting task with priority {priority} and length {length}')

        
        if poly is None:
            poly = self.degree_search()
            result, new_priority = None, 10000
        else:
            result, new_priority = self.validate(array)

        # Update the results log
        with self.admin['success_lock']:
            self.admin['success']['total'] += 1
            
            # Success condition is that we got a successful quadratic
            if result is not None:
                self.admin['success']['list'].append(array)
                if array.shape[0] < self.admin['success']['best']:
                    self.admin['success']['best'] = array.shape[0]
                    log('solver', self.name, f'Found new best aux array with length {array.shape[0]}')
                    log('solver', self.name, f'History: {history}')
                    log('solver', self.name, f'{array}')
                    
                    #self.admin['success']['found'] = True
                
                return

        log('solver', self.name, f'Adding item with priority {priority}')
        
        # Otherwise, put the appropriate information in the completed task queue.
        self.admin['done_queue'].put(PrioritizedItem(priority = new_priority, item = (array, poly), history = history))

    def request_task(self):
        if not self.admin["task_queue"].empty():
            task = self.admin["task_queue"].get()
            return task


        return None

    def degree_search(self):
        for degree in range(2, 6):
            #if degree > 2:
            #    degree = 6
            M, keys, correct = self.factory.get(aux_array = None, degree = degree, radius = None)
            solver = LPWrapper(keys)
            M = M.to_sparse()
            solver.add_constraints(M)
            solution = solver.solve()
            if solution is None:
                continue

            poly = get_poly(keys, solution)
            log('solver', self.name, f'Found solution with degree {degree}.')
            print(poly)

            return poly

        log('solver', self.name, 'ERROR: No solution at any degree! This should not be possible.')
        return None

    def validate(self, array):
        """
        Attempts to fit a quadratic. This method operates under the assumption that most arguments passed to it will be infeasible, so it attempts to disqualify them cheaply using sequential constraint building. This will be much more expensive for a feasible aux array, but much cheaper for an infeasible. This means that insofar as most aux arrays are infeasible, total cost should be reduced. 
        """

        
        
        for radius in [2, 4]:
            constraints, keys, correct = self.factory.get(aux_array = torch.tensor(array), degree = 2, radius = radius)
            objective = call_my_solver(constraints.to_sparse_csc(), tolerance = 1e-8)
            if objective > 0.1:
                return None, 100 / radius + 100 * objective / constraints.shape[0]
            log('solver', self.name, f'Passed basin {radius}') 


        constraints, keys, correct = self.factory.get(aux_array = torch.tensor(array), degree = 2, radius = None)
        constraints = constraints.to_sparse_csc()
        objective = call_my_solver(constraints, tolerance = 1e-8)
        if objective > 0.1:
            return None, 100 * objective / constraints.shape[0]

        return objective, 100 * objective / constraints.shape[0]
        

        for radius in range(1,3):
            constraints, keys = constraints_basin(self.N1, self.N2, torch.tensor(array), 2, radius, 50)
            objective = call_my_solver(constraints.to_sparse_csc(), tolerance = 1e-8)
            if objective > 0.1:
                if radius > 1:
                    log('solver', self.name, f'Failed at basin {radius}.')

                return None, 100/radius + 100 * objective / constraints.shape[0]
            log('solver', self.name, f'Passed basin {radius}.')


        """
        glop = LPWrapper(keys)
        glop.add_constraints(constraints.to_sparse())
        result = glop.solve()

        if result is None:
            log('solver', self.name, 'ERROR! PASSED MY SOLVER BUT FAILED GLOP')
            return None
        """
        
        log('solver', self.name, 'Looking promising!')
        print(array)
        print(constraints.size())
        
        objective = call_solver(self.N1, self.N2, array)
        feasible = objective < 0.5
        new_priority = objective/(1 << (2 * (self.N1 + self.N2) + array.shape[0]) - 1 << (self.N1 + self.N2 + array.shape[0])) 
        log('solver', self.name, f'full check {feasible}, priority = {new_priority}')
        return (None, new_priority) if not feasible else (feasible, new_priority)
        
"""

330     0   
331     06 16 46
332     16 26 36 58
333     26 37 56 09
334     46 26

comb    06 16 26 36 46 56 236 056 256

33      09 1,10 39 08

#340 1      
            (1,2)
#341 4      
            (0,7) (3,7) (1,7) (4,7)

#342 6?     (4,7) (3,7) (3,8) (2,7) (0,7) (1,8) (1,9) (2,9) (5,6)
            (3,7) (4,7) (1,8) (1,9) (2,8) (2,7) (5,6) (5,9)
            (4,7) (3,7) (1,8) (5,8) (2,(4,7)) (1,(4,7))

#343 8?
            (0,3) (0,5) (2,7) (1,7) (1,(0,5)) (3,5) (4,7) (6,(2,7)) (6,(3,5))
            07 17 23 25 47 35 68 39
#344 4      
            (2,7) (4,8) (6,7) (0,(4,8))
#345 2      
            (1,7) (6,7)
#346 0

340     12
341     07 17 37 47
342     37 47 137 237 147 547
343     07 17 47 23 25 35 067 137
344     27 247 67 067
345     17 67


(1,2) (0,7) (1,7) (2,7) (3,7) (4,7) (6,7) (4,8) (0,4,8) (0,3) (0,5) (0,1,5) (3,5) (2,6,7) (3,5,6) 
(1,8) (5,8) 
(2,4,7) (1,4,7)



440     16
441     08 18 28 48 58






334     17 47



330     0
331     1       (0, 4, 5)
332     1       (0, 1, 3, 4, 5)
333     3       (0, 1, 4, 5)
334     2       (0, 5)
335     0


330     0
331     1       (0, 4, 5)
332     1       (0, 1, 4, 5)
333     2       (0, 1, 2, 4, 5)
334     2       (0, 5)
335     0

340     n/a
341     n/a
342     n/a
343     2
344     5   (incl 2)
345     3   (incl 0, 1)
346     0

340     n/a
341     n/a
342     n/a
343     n/a
344     5   (incl 3)        (use 0,1,2,5,6)
345     3   (incl 0, 1, 2) (use 0)
346     0

440     n/a
441     1   (0,5,6,7)
442     2   (0,1,5,6,7)
443     3   (0,1,2,5,6,7)
444     3   (0,1,2,3,5,6,7)
445     4   (0,6,7)
446     3   (incl 0)
447     0

[1,5] incl [0,6,7] got 6
[5] incl [0,6,7] got 4

"""


@click.command()
@click.option("--n1", default=2, help="First input")
@click.option("--n2", default=2, help="Second input")
@click.option("--bit", default=None, help="Select an output bit.")
@click.option("--solvers", default=os.cpu_count()-1, help="Number of solver processes.")
@click.option("--delegators", default=1, help="Number of delegator processes.")
@click.option("--limit", default = 16, help="Maximum aux array size.")
def main(n1, n2, bit, solvers, delegators, limit):

    factory = ConstraintFactory(n1, n2, mask = [4], include = [0, 1, 2, 3, 5, 6, 7])

    search(factory, solvers, delegators, limit)


if __name__ == "__main__":
    freeze_support()
    multiprocessing.set_start_method('spawn')
    #cProfile.run('main()')
    main()
