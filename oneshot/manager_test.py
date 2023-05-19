from dqrl import BanGameEmulator
import os

from multiprocessing.managers import BaseManager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time
from ising import IMul

class Foo:
    def __init__(self, arg):
        print(arg)
        self.arg = arg

    def f(self):
        time.sleep(2)
        print(f'called f {self.arg}')
        return 1

class GameManager(BaseManager): pass    

num_threads = len(os.sched_getaffinity(0))

circuit = IMul(2,3)
degree = 3

GameManager.register('Foo', BanGameEmulator)


start = time.perf_counter()

managers = []
emulators = []
for i in range(num_threads):
    manager = GameManager()
    manager.start()
    managers.append(manager)
    emulators.append(manager.Foo(circuit, degree))




futures = []
with ProcessPoolExecutor(max_workers = num_threads) as executor:
    for emulator in emulators:
        futures.append(executor.submit(emulator.setup))

for future in futures:
    print(future.result())

end = time.perf_counter()

print(f'manager setup {end-start}')

start = time.perf_counter()

futures = []
with ProcessPoolExecutor(max_workers = num_threads) as executor:
    for emulator in emulators:
        futures.append(executor.submit(emulator.run))

for future in futures:
    print(future.result())

end = time.perf_counter()

print(f'managers took {end - start}')

start = time.perf_counter()

emulators = [
    BanGameEmulator(circuit, degree)
    for i in range(num_threads)
]

for emulator in emulators:
    print(emulator.setup())

end = time.perf_counter()

print(f'trad setup {end-start}')

start = time.perf_counter()

for emulator in emulators:
    print(emulator.run())

end = time.perf_counter()

print(f'trad took {end-start}')
