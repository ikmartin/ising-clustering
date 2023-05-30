from multiprocessing.managers import SyncManager
from multiprocessing import Process, Lock, freeze_support, Manager
from queue import PriorityQueue
import multiprocessing
from collections import deque
import time, os, torch, click
import numpy as np
from math import comb

from solver import LPWrapper
from oneshot import full_Rosenberg, rosenberg_criterion, get_term_table, Rosenberg

from dataclasses import dataclass, field
from typing import Any

from fast_constraints import fast_constraints
from oneshot import reduce_poly, MLPoly
from ising import IMul
from fittertry import IMulBit


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
    2. Build a circuit and LP solver. Fit a polynomial of minimum degree.
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

    return full_Rosenberg(poly).num_variables()


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


## Class which is capable of producing many copies of the same circuit, optionally pre-initialized with specified auxilliary arrays.

class CircuitFactory:
    def __init__(self, CircuitClass, args = None):
        self.args = args
        self.CircuitClass = CircuitClass

    def get(self, aux_array = None):
        circuit = self.CircuitClass(*self.args)
        if aux_array is None:
            return circuit

        circuit.set_all_aux(aux_array.tolist())
        return circuit

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

    print(format)
        

def search(circuit_args, num_solvers, num_delegators):
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

    # Generate circuit factory, used for making new circuits on the specified pattern.
    factory = CircuitFactory(circuit_args['class'], circuit_args['args'])

    log('master', 'Master', "Creating solvers...")
    workers = [
        Solver(admin, factory)
        for i in range(num_solvers)
    ]

    log('master', 'Master', "Creating delegators...")
    delegators = [
        Delegator(admin, factory)
        for i in range(num_delegators)
    ]

    log('master', 'Master', "Initialized.")

    log('master', 'Master', "Starting processes.")
    for worker in workers:
        worker.start()

    for delegator in delegators:
        delegator.start()

    log('master', 'Master', "Setting initial task.")
    admin["task_queue"].put(PrioritizedItem(priority = 0, item = None))

    log('master', 'Master', "Letting solvers work.")
    while True:
        queue_length = admin["task_queue"].qsize()
        done_length = admin["done_queue"].qsize()

        best_score = admin["success"]["best"]
        num_done = admin["success"]["total"]

        log('master', 'Master', f"{num_done} done / {queue_length} ready / {done_length} docket / best {best_score}")
        time.sleep(1.0)

class Delegator(Process):
    def __init__(self, admin, factory):
        self.admin = admin
        self.factory = factory
        self.STABLE_QSIZE = 10000
        super().__init__()

    def run(self):
        log('delegator', self.name, 'Running.')
        while True:
            self.loop()

    def loop(self):
        # If action is not needed or not possible, just wait.
        if self.admin["task_queue"].qsize() > self.STABLE_QSIZE or self.admin["done_queue"].empty():
            return

        # Get a recently completed task and unpack it
        task = self.admin["done_queue"].get()
        array, poly = task.item

        # Check to see if the current aux array is even with the best final solutions. Since we know that we are about to add a new row to the aux array, if this is the case, then whatever tasks we would add would be guarunteed not to improve on the currently known best answer. Therefore, simply skip.
        with self.admin['success_lock']:
            if array is not None and array.shape[0] >= self.admin['success']['best'] - 1:
                log('delegator', self.name, 'Worthless docket item.')
                return

            
        log('delegator', self.name, f'Accepting docket item with priority {task.priority}.')

        # Generate new tasks and add them to the queue
        self.make_new_tasks(poly, array)

    def make_new_tasks(self, poly, array):
        circuit = self.factory.get(array)

        term_table = get_term_table(poly, rosenberg_criterion, 2)

        # Printout information objects
        log_priorities = []
        new_length = array.shape[0] + 1 if array is not None else 1

        for C, H in term_table.items():
            if not len(H):
                continue
            
            new_poly = Rosenberg(poly, C, H)
            aux_map = MLPoly({C: 1})
            new_aux_vector = np.array([[aux_map(tuple(circuit.inout(inspin).binary())) for inspin in circuit.inspace]])
            
            new_aux_array = new_aux_vector if array is None else np.concatenate([array, new_aux_vector])

            new_array_id = hash_aux_array(new_aux_array)

            if new_array_id in self.admin['dibs']:
                continue

            self.admin['dibs'][new_array_id] = True
            new_priority = estimate_score(new_poly)

            new_task = PrioritizedItem(priority = new_priority, item = new_aux_array)

            self.admin['task_queue'].put(new_task)
            log_priorities.append(new_priority)

        log_priorities = sorted(log_priorities)
        log('delegator', self.name, f'Added new tasks (length = {new_length}): priorities {log_priorities}')

class Solver(Process):
    def __init__(self, admin, factory):
        self.admin = admin
        self.factory = factory
        super().__init__()

    def run(self):
        log('solver', self.name, 'Running.')
        while True:
            self.loop()

    def loop(self):
        # Obtain a task from the queue, unpack the aux array, and generate the correct circuit
        task = self.request_task()
        if task is None:
            return

        array = task.item
        priority = task.priority
        
        # Check to see if this task is worth doing at all
        with self.admin['success_lock']:
            if array is not None and array.shape[0] >= self.admin['success']['best']:
                log('solver', self.name, f'Worthless task (priority {priority}), skipping.')
                return

            # Set a flag indicating that this task is only worthwhile if it is solved immediately in case we are currently one aux vector behind the best. 
            quad_only = (array.shape[0] == self.admin['success']['best'] - 1) if array is not None else False
        
        
        log('solver', self.name, f'Accepting task with priority {priority}')

        circuit = self.factory.get(array)
        
        # Search for the minimum viable degree. This is the expensive step.
        poly = self.degree_search(circuit, quad_only = quad_only)

        # If we didn't get any polynomial (likely intentionally skip of higher degree attempts),
        # then we are essentially skipping this task as worthless.
        if poly is None:
            log('solver', self.name, 'Cannot fit quadratic, abandoning task.')
            return

        # Update the results log
        with self.admin['success_lock']:
            self.admin['success']['total'] += 1
            
            # Success condition is that we got a successful quadratic
            if poly.degree() == 2:
                self.admin['success']['list'].append(array)
                if array.shape[0] < self.admin['success']['best']:
                    self.admin['success']['best'] = array.shape[0]
                    log('solver', self.name, f'Found new best aux array with length {array.shape[0]}')
                    print(array)
                
                return

        
        # Otherwise, put the appropriate information in the completed task queue.
        priority = estimate_score(poly)
        self.admin['done_queue'].put(PrioritizedItem(priority = priority, item = (array, poly)))

    def request_task(self):
        if not self.admin["task_queue"].empty():
            task = self.admin["task_queue"].get()
            return task


        return None

    def degree_search(self, circuit, quad_only = False):
        for degree in range(2, circuit.G):
            M, keys = fast_constraints(circuit, degree)
            solver = LPWrapper(M, keys)
            solution = solver.solve()
            if solution is None:
                if quad_only:
                    return None

                continue

            poly = get_poly(keys, solution)
            log('solver', self.name, f'Found solution with degree {degree}.')

            return poly

        log('solver', self.name, 'ERROR: No solution at any degree! This should not be possible.')
        return None



@click.command()
@click.option("--n1", default=2, help="First input")
@click.option("--n2", default=2, help="Second input")
@click.option("--bit", default=0, help="Select an output bit.")
@click.option("--solvers", default=os.cpu_count()-1, help="Number of solver processes.")
@click.option("--delegators", default=1, help="Number of delegator processes.")
def main(n1, n2, bit, solvers, delegators):

    circuit_args = {
        'class': IMulBit,
        'args': (n1, n2, bit)
    }
    search(circuit_args, solvers, delegators)


if __name__ == "__main__":
    freeze_support()
    multiprocessing.set_start_method('spawn')
    main()
