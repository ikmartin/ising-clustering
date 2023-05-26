from multiprocessing.managers import BaseManager
from multiprocessing import Process, Lock, freeze_support, Manager
import multiprocessing
from collections import deque
import time, os, torch, click
import numpy as np
from math import comb

from solver import LPWrapper
from oneshot import full_Rosenberg

from ortools.linear_solver.pywraplp import Solver

from fast_constraints import fast_constraints
from oneshot import reduce_poly, MLPoly
from ising import IMul
#from fittertry import IMulBit


def request_task(admin):
    if not admin["queue"].empty():
        task = admin["queue"].get()
        # with admin['cache_lock']:
        admin["cache"][task] = False
        return task

    return None


def submit(admin, task, coeffs, score):
    # with admin['cache_lock']:
    admin["cache"][task] = True

    admin["done"].put((task, coeffs))

    if score is not None:
        with admin["score_lock"]:
            if admin["score"]["score"] is None:
                admin["score"]["score"] = score
            else:
                if admin["score"]["score"] > score:
                    admin["score"]["score"] = score
                    admin["score"]["task"] = task

            admin["score"]["total"] += 1


def add_new_tasks(admin, new_tasks):
    for task in new_tasks:
        if task not in admin["cache"]:
            admin["queue"].put(task)


def make_new_tasks(task, coeffs, min_ban_number):
    return (
        [task | frozenset({i}) for i in coeffs if i >= min_ban_number]
        if coeffs is not None
        else []
    )


def search(circuit, degree, num_workers):
    print("[Master] Getting constraints...")
    M, keys = fast_constraints(circuit, degree)

    min_ban_length = 3
    min_ban_idx = len([key for key in keys if len(key) < min_ban_length])

    print("[Master] Creating global manager.")
    manager = Manager()
    admin = {
        "cache_lock": manager.Lock(),
        "score_lock": manager.Lock(),
        "queue": manager.Queue(),
        "cache": manager.dict(),
        "score": manager.dict(),
        "done": manager.Queue(),
    }
    admin["score"]["score"] = None
    admin["score"]["task"] = None
    admin["score"]["total"] = 0

    print("[Master] Creating solvers...")
    workers = [
        SolverProcess(
            admin,
            circuit,
            degree,
            constraints=(M, keys),
            name=f"solve_process_{i}",
        )
        for i in range(num_workers)
    ]

    print("[Master] Creating delegator...")
    delegator = Delegator(admin, circuit, min_ban_idx)

    print("[Master] Initialized.")

    print("[Master] Starting processes.")
    delegator.start()
    for worker in workers:
        worker.start()

    print("[Master] Setting initial task.")
    admin["queue"].put(frozenset())

    print("[Master] Letting solvers work.")
    while True:
        queue_length = admin["queue"].qsize()

        best_score = admin["score"]["score"]
        best_task = admin["score"]["task"]
        num_done = admin["score"]["total"]

        print(f"[Master] {queue_length} tasks in queue.")
        print(f"[Master] {num_done} complete. Best so far is {best_task}: {best_score}")
        time.sleep(0.1)


# 3x3 bitwise best
# 0 + 4 + 5 + 6 + 2 + 0
# +fgbz/fd
# 0 + 10 + 19 + 24 + 5 + 0

# 2x3 bitwise best
# 0 + 1 + 2 + 2 + 0
# fgbz/fd
# 0 + 3 + 6 + 5 + 0

# 4x4 bitwise best
# 1 + 11 + [16] + [23] + [23] + 6 + 2 + 0


class Delegator(Process):
    def __init__(self, admin, circuit, min_ban_idx):
        self.admin = admin
        self.circuit = circuit
        self.min_ban_idx = min_ban_idx
        super().__init__()

    def run(self):
        while True:
            need = self.admin["queue"].qsize() < 100 and not self.admin["done"].empty()

            if not need:
                continue

            task, coeffs = self.admin["done"].get()

            new_tasks = make_new_tasks(task, coeffs, self.min_ban_idx)
            add_new_tasks(self.admin, new_tasks)


class SolverProcess(Process):
    def __init__(self, admin, circuit, degree, constraints=None, name=None):
        self.admin = admin
        self.circuit = circuit
        self.degree = degree
        self.M, self.keys = (
            constraints if constraints is not None else fast_constraints(circuit, degree)
        )

        self.name = name if name is not None else "solver"

        self.num_terms = self.M.shape[1]
        self.threshold = 0.01
        super().__init__()

    def run(self):
        print(f"[{self.name}] Building...")
        self.solver = LPWrapper(self.M, self.keys)

        print(f"[{self.name}] Ready.")
        # Run an infinite loop waiting for tasks to be added to the parent queue
        while True:
            task = request_task(self.admin)
            if task is None:
                time.sleep(0.1)
                continue

            coeffs = self.solver.solve(task)

            nonzero = (
                frozenset(np.nonzero(coeffs)[0].tolist())
                if coeffs is not None
                else None
            )
            if coeffs is not None:
                score = full_Rosenberg(self.get_poly(coeffs)).num_variables() - self.circuit.G
            else:
                score = None

            # print(f'[{self.name}] Task {task}: {score}')
            submit(self.admin, task, nonzero, score)


    def get_poly(self, coeffs) -> MLPoly:
        coeff_dict = {key: val for key, val in zip(self.keys, coeffs)}
        return MLPoly(coeff_dict)


@click.command()
@click.option("--n1", default=2, help="First input")
@click.option("--n2", default=2, help="Second input")
@click.option("--degree", default=3, help="Degree")
@click.option("--bit", default=None, help="Select an output bit.")
def main(n1, n2, degree, bit):
    if bit is None:
        circuit = IMul(n1, n2)
    else:
        circuit = IMulBit(n1, n2, int(bit))

    num_workers = len(os.sched_getaffinity(0))

    search(circuit, degree, num_workers)


if __name__ == "__main__":
    main()
