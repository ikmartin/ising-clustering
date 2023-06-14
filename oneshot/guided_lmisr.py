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

from fast_constraints import fast_constraints, sequential_constraints, constraints_basin
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


## Class which is capable of producing many copies of the same circuit, optionally pre-initialized with specified auxilliary arrays.

class CircuitFactory:
    def __init__(self, CircuitClass, args = None):
        self.args = args
        self.CircuitClass = CircuitClass

    def get(self, aux_array = None):
        circuit = self.CircuitClass(*self.args)
        if aux_array is None:
            return circuit

        aux_array = aux_array * 2 - 1

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

    print(format, flush = True)
        

def search(circuit_args, num_solvers, num_delegators, limit):
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

    constraint_priority = gen_constraint_priority(factory.get().inspace)

    print([[spin.splitint() for spin in stage] for stage in constraint_priority])

    log('master', 'Master', "Creating solvers...")
    workers = [
        Solver(admin, factory, constraint_priority, limit)
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
    admin["task_queue"].put(PrioritizedItem(priority = 0, item = (None, None)))

    log('master', 'Master', "Letting solvers work.")
    while True:
        queue_length = admin["task_queue"].qsize()
        done_length = admin["done_queue"].qsize()

        best_score = admin["success"]["best"]
        num_done = admin["success"]["total"]

        log('master', 'Master', f"{num_done} done / {queue_length} ready / {done_length} docket / best {best_score}")


        time.sleep(1.0)

        """
        if best_score < 1e5:
            exit()

        """

def rank_spin(spin):
    n1, n2 = spin.splitint()
    rank = abs(int(n1 + n2 - (2 ** (spin.dim()/2) - 1)))
    return rank


def gen_constraint_priority(inspace):
    inspin_list = [
        (spin, rank_spin(spin))
         for spin in inspace
    ]

    inspin_list = sorted(inspin_list, key = lambda pair: pair[1])
    stage0 = [inspace.getspin(2 ** inspace.shape[0] - 1)]
    stage1 = [
        spin 
        for spin, rank in inspin_list
        if rank < 1
        and spin not in stage0
    ]
    stage2 = [
        spin
        for spin, rank in inspin_list
        if spin not in stage1
        and spin not in stage0
    ]
    return [stage0, stage1, stage2]


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
            self.loop()

    def loop(self):
        # If action is not needed or not possible, just wait.
        if self.admin["task_queue"].qsize() > self.STABLE_QSIZE or self.admin["done_queue"].empty():
            return

        # Get a recently completed task and unpack it
        task = self.admin["done_queue"].get()
        array, poly = task.item
        priority = task.priority

        # Check to see if the current aux array is even with the best final solutions. Since we know that we are about to add a new row to the aux array, if this is the case, then whatever tasks we would add would be guarunteed not to improve on the currently known best answer. Therefore, simply skip.
        with self.admin['success_lock']:
            if array is not None and (array.shape[0] >= self.admin['success']['best'] - 1 or array.shape[0] >= self.limit) and priority > 1:
                log('delegator', self.name, 'Worthless docket item.')
                return

            
        log('delegator', self.name, f'Accepting docket item with priority {task.priority}.')

        # Generate new tasks and add them to the queue
        self.make_new_tasks(poly, array, priority)

    def make_new_tasks(self, poly, array, priority):
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



        circuit = self.factory.get(array)

        # table of further options --- should be tuples of (new_poly, aux_map)
        options = []

        rosenberg_term_table = get_term_table(poly, rosenberg_criterion, 2)
        
        pos_fgbz_term_table = get_term_table(poly, weak_positive_FGBZ_criterion)
        neg_fgbz_term_table = get_term_table(poly, negative_FGBZ_criterion)
        """
        fd_options = [
            (key, value)
             for key, value in poly.coeffs.items()
            if value < 0 and len(key) > 2
        ]
        """

        for C, H in rosenberg_term_table.items():
            if not len(H):
                continue

            options.append((Rosenberg(poly, C, H), MLPoly({C: 1})))

        
        for C, H in pos_fgbz_term_table.items():
            if not len(H):
                continue

            options.append((
                PositiveFGBZ(poly, C, H),
                MLPoly({
                    () : 1,
                    C : -1
                })
            ))

        for C, H in neg_fgbz_term_table.items():
            if not len(H):
                continue

            options.append((
                NegativeFGBZ(poly, C, H),
                MLPoly({
                    C : 1
                })
            ))
        
        
        """

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
        new_length = array.shape[0] + 1 if array is not None else 1

        for new_poly, aux_map in options:
            new_aux_vector = np.array([[aux_map(tuple(circuit.inout(inspin).binary())) for inspin in circuit.inspace]])
            
            new_aux_array = new_aux_vector if array is None else np.concatenate([array, new_aux_vector])

            new_array_id = hash_aux_array(new_aux_array)

            if new_array_id in self.admin['dibs']:
                continue

            self.admin['dibs'][new_array_id] = True
            new_priority = estimate_score(new_poly)

            new_task = PrioritizedItem(priority = new_priority + new_length, item = (new_aux_array, new_poly))

            self.admin['task_queue'].put(new_task)
            log_priorities.append(new_priority)

        log_priorities = sorted(log_priorities)
        log('delegator', self.name, f'Added new tasks (length = {new_length}): priorities {log_priorities}')

class Solver(Process):
    def __init__(self, admin, factory, constraint_priority, limit):
        self.admin = admin
        self.factory = factory
        example_circuit = factory.get()
        self.N1, self.N2 = example_circuit.N1, example_circuit.N2
        self.limit = limit
        self._gen_constraint_priority(constraint_priority)
        
        super().__init__()

    def _gen_constraint_priority(self, stages):
        if stages is None:
            example_circuit = self.factory.get()
            self.stages = [example_circuit.inspace]

        self.stages = stages

    def run(self):
        log('solver', self.name, 'Running.')
        while True:
            self.loop()

    def loop(self):
        # Obtain a task from the queue, unpack the aux array, and generate the correct circuit
        task = self.request_task()
        if task is None:
            return

        array, poly = task.item
        priority = task.priority
        
        # Check to see if this task is worth doing at all
        with self.admin['success_lock']:
            if array is not None and (array.shape[0] >= self.admin['success']['best'] or array.shape[0] > self.limit):
                log('solver', self.name, f'Worthless task (priority {priority}), skipping.')
                return
        
        length = 0 if array is None else array.shape[0]
        log('solver', self.name, f'Accepting task with priority {priority} and length {length}')

        
        if poly is None:
            circuit = self.factory.get(array)
            poly = self.degree_search(circuit, quad_only = False)
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
                    log('solver', self.name, f'{array}')
                
                return

        log('solver', self.name, f'Adding item with priority {priority}')
        
        # Otherwise, put the appropriate information in the completed task queue.
        self.admin['done_queue'].put(PrioritizedItem(priority = new_priority, item = (array, poly)))

    def request_task(self):
        if not self.admin["task_queue"].empty():
            task = self.admin["task_queue"].get()
            return task


        return None

    def degree_search(self, circuit, quad_only = False):
        for degree in range(2, circuit.G):
            #if degree > 2:
            #    degree = 5
            M, keys = fast_constraints(circuit, degree)
            solver = LPWrapper(keys)
            solver.add_constraints(M)
            solution = solver.solve()
            if solution is None:
                if quad_only:
                    return None

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


        for radius in range(1,3):
            constraints, keys = constraints_basin(self.N1, self.N2, torch.tensor(array), 2, radius)
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
        
#340 1
#341 4
#342 8?
#343 9?
#344 4
#345 2
#346 0

@click.command()
@click.option("--n1", default=2, help="First input")
@click.option("--n2", default=2, help="Second input")
@click.option("--bit", default=None, help="Select an output bit.")
@click.option("--solvers", default=os.cpu_count()-1, help="Number of solver processes.")
@click.option("--delegators", default=1, help="Number of delegator processes.")
@click.option("--limit", default = 16, help="Maximum aux array size.")
def main(n1, n2, bit, solvers, delegators, limit):

    if bit is None:
        circuit_args = {
            'class': IMul,
            'args': (n1, n2)
        }
    else:
        circuit_args = {
            'class': IMulBit,
            'args': (n1, n2, int(bit))
        }

    search(circuit_args, solvers, delegators, limit)


if __name__ == "__main__":
    freeze_support()
    multiprocessing.set_start_method('spawn')
    #cProfile.run('main()')
    main()
