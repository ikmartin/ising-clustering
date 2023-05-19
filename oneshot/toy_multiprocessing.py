import multiprocessing, time, random, os
from multiprocessing.managers import BaseManager
from multiprocessing import Process, Manager

class Master(Process):
    def __init__(self, num_workers):
        self.globals = Manager()
        self.task_queue = self.globals.Queue()
        self.result_queue = self.globals.Queue()

        self.workers = [
            Worker(i, self.task_queue, self.result_queue)
            for i in range(num_workers)
        ]
        super().__init__()

    def run(self):
        for process in self.workers:
            process.start()

        while not self.task_queue.empty():
            print(f'{self.task_queue.qsize()} {self.result_queue.qsize()}')


class Worker(Process):
    def __init__(self, id, task_queue, result_queue):
        self.id=id 
        self.task_queue=task_queue
        self.result_queue = result_queue
        super().__init__()

    def run(self):
        print(f'[{self.id}] running')
        
        while not self.task_queue.empty():
            task = self.task_queue.get()
            time.sleep(random.random())
            self.result_queue.put(task)
        

class WorkerManager(BaseManager): pass
WorkerManager.register('WorkerClass', Worker)

def test():
    print('hi')

def main():
    num_workers = os.cpu_count()
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    
    for i in range(100):
        task_queue.put(random.random())
    
    workers = [
        Worker(i, task_queue, result_queue)
        for i in range(num_workers)
    ]
    for worker in workers:
        worker.start()
   

    while result_queue.qsize() < 100:
        print(f'{task_queue.qsize()} {result_queue.qsize()}')

    for worker in workers:
        worker.join()

if __name__ == '__main__':
    main()
