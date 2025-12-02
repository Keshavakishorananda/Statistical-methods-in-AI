import numpy as np
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, K):
        self.K = K
        self.means = None
        self.covariances = None
        self.weights = None

    def fit(self, X_train):
        np.random.seed(42)
        self.means = np.random.rand(self.K, X_train.shape[1])
        self.covariances = np.array([np.eye(X_train.shape[1]) for _ in range(self.K)])
        self.weights = np.ones(self.K) / self.K
        prev_likelihood = -np.inf

        while True:
            # E-step:
            log_posterior_prob = np.zeros((self.K, X_train.shape[0]))
            for i in range(self.K):
                log_posterior_prob[i] = np.log(self.weights[i]) + multivariate_normal.logpdf(X_train, mean=self.means[i], cov=self.covariances[i])

            max_log = np.max(log_posterior_prob, axis=0)
            log_posterior_prob -= max_log
            posterior_prob = np.exp(log_posterior_prob)
            posterior_prob /= np.sum(posterior_prob, axis=0)

            # M-step:
            # For means:
            for i in range(self.K):
                self.means[i] = np.sum(posterior_prob[i].reshape(-1,1) * X_train, axis=0)/np.sum(posterior_prob[i])

            # For covariances:
            for i in range(self.K):
                cov_matrix = np.dot((posterior_prob[i].reshape(-1,1) * (X_train - self.means[i])).T, (X_train - self.means[i]))/np.sum(posterior_prob[i])
                self.covariances[i] = cov_matrix + 1e-6 * np.eye(X_train.shape[1])

            # For weights:
            for i in range(self.K):
                self.weights[i] = np.sum(posterior_prob[i])/(X_train.shape[0])

            # Calculate the log-likelihood
            likelihood_list = np.zeros((self.K, X_train.shape[0]))
            self.likelihood = 0
            for i in range(self.K):
                likelihood_list[i] = np.log(self.weights[i]) + multivariate_normal.logpdf(X_train, mean=self.means[i], cov=self.covariances[i])    
            
            max_log = np.max(likelihood_list, axis=0)
            log_likelihood_point = likelihood_list - max_log
            likelihood_list = np.exp(log_likelihood_point)
            log_likelihood = np.log(np.sum(likelihood_list, axis=0)) + max_log

            self.likelihood = np.sum(log_likelihood)

            if np.abs(self.likelihood - prev_likelihood) < 1e-6:
                break

            prev_likelihood = self.likelihood


    def getParams(self):
        return self.means, self.covariances, self.weights

    def getMembership(self, data_points):
        log_posterior_prob = np.zeros((self.K, data_points.shape[0]))
        for i in range(self.K):
            log_posterior_prob[i] = np.log(self.weights[i]) + multivariate_normal.logpdf(data_points, mean=self.means[i], cov=self.covariances[i])

        max_log = np.max(log_posterior_prob, axis=0)
        log_posterior_prob -= max_log
        posterior_prob = np.exp(log_posterior_prob)
        posterior_prob /= np.sum(posterior_prob, axis=0)
        posterior_prob = posterior_prob.T

        return posterior_prob
    
    def getprob_density(self, data_points):
        # create a nn list of size n to store the probability density of each data point
        prob_density = np.zeros(data_points.shape[0])
        for i in range(self.K):
            prob_density += self.weights[i] * multivariate_normal.pdf(data_points, mean=self.means[i], cov=self.covariances[i])
        return prob_density

    def getLikelihood(self):
        return self.likelihood