import numpy as np

class PCA():
    def __init__(self, N_components):
        self.N_components = N_components
        self.components = None
        self.sort_eigen = None

    def fit(self, X):
        means = np.mean(X, axis=0)
        X = X - means
        cov_matrix = np.dot(X.T, X) / (X.shape[0] - 1)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        idx = eigenvalues.argsort()[::-1]
        sorted_eigenvectors = eigenvectors[:, idx]
        sorted_eigenvectors = sorted_eigenvectors.real
        sorted_eigenvectors = sorted_eigenvectors.T

        for x in sorted_eigenvectors:
            if np.sum(x) < 0:
                x *= -1   

        components = sorted_eigenvectors[:self.N_components]

        self.components = components
        self.sort_eigen = eigenvalues[idx]


    def transform(self, X):
        means = np.mean(X, axis=0)
        X = X - means
        return np.dot(X, self.components.T)

    def checkPCA(self, x, X_reduced):
        x_reconstructed = np.dot(X_reduced, self.components) + np.mean(x, axis=0)
        reconstruction_error = np.mean(np.abs(x - x_reconstructed))

        if reconstruction_error < 1:
            return True
        else:
            return False