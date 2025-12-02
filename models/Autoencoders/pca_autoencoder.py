import numpy as np

class PCA_Autoencoder():
    def __init__(self, n_components):
        self.n_components = n_components
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

        components = sorted_eigenvectors[:self.n_components]

        self.components = components
        self.sort_eigen = eigenvalues[idx]

    def encode(self, X):
        means = np.mean(X, axis=0)
        X = X - means
        return np.dot(X, self.components.T)
        

    def forward(self, X):
        X_reconstructed = np.dot(self.encode(X), self.components) + np.mean(X, axis=0)
        return X_reconstructed
          