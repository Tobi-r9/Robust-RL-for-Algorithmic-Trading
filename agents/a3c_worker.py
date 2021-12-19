from threading import Thread


class A3CWorker(Thread):

    def __init__(self, nn_model):
        self.global_worker = nn_model