from multiprocessing import cpu_count
from agents.a3c_worker import A3CWorker


class Agent():

    def __init__(self,
                action_dim,
                state_dim, 
                num_workers=cpu_count()):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_workers = num_workers
        self.global_worker = A3CWorker()

    def train(self, max_episodes=20000):
        workers = []
        for i in range(self.num_workers):
            workers.append(
                A3CWorker()
            )
        
        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()
