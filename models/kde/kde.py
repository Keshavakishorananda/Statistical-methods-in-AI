import numpy as np
import matplotlib.pyplot as plt

class kernels:
    def __init__(self):
        pass

    def gaussian(self, x):
        return np.product(np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), axis=1)
    
    def triangular(self, x):
        return np.product(np.maximum(0, 1 - np.abs(x)), axis=1)

    def box(self, x):
        return np.product(np.where(np.abs(x) <= 1, 1, 0), axis=1)

        

class KDE(kernels):
    def __init__(self, kernels, bandwidth):
        super(KDE, self).__init__()
        self.kernels = kernels
        self.bandwidth = bandwidth

    def fit(self, X):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]

    def predict(self, x):
        kernel_input = (x - self.X) / self.bandwidth

        if self.kernels == 'Gaussian':
            kernel_output = self.gaussian(kernel_input)
            kernel_output = kernel_output / (self.bandwidth ** self.d)
            return np.mean(kernel_output)
        elif self.kernels == 'triangular':
            kernel_output = self.triangular(kernel_input)
            kernel_output = kernel_output / (self.bandwidth ** self.d)
            return np.mean(kernel_output)
        elif self.kernels == 'box':
            kernel_output = self.box(kernel_input)
            kernel_output = kernel_output / (self.bandwidth ** self.d)
            return np.mean(kernel_output)
        else:
            raise ValueError('Kernel not supported')

    def visualize(self):
        if self.d != 2:
            raise ValueError("Visualization is only supported for 2D data.")

        # plot the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.predict(np.array([X[i, j], Y[i, j]]))
        ax.plot_surface(X, Y, Z, cmap='viridis')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('KDE')
        plt.savefig('figures/KDE.png')
        plt.close()
