import numpy as np
import sys
sys.path.append('..')
from models.MLP.MLP import MLP

class Autoencoders(MLP):
    def __init__(self, hidden, neurons, active_fun):
        super().__init__(hidden, neurons, active_fun)

    def fit(self, x, y, optimizer, loss_function, lr, epochs, batch_size):
        super().fit(x, x, optimizer, loss_function, lr, epochs, batch_size)

    def get_latent(self, x):
        super().predict(x)
        return self.forward_activation[self.hidden//2]