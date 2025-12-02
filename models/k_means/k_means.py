import numpy as np

class Kmeans():
    def __init__(self, K):
        self.K = K
        self.centroids = None
        self.WCSS = None
        self.labels = None

    def fit(self, X_train):
        np.random.seed(42)
        self.centroids = X_train[np.random.choice(X_train.shape[0], self.K, replace=False)]
        self.WCSS = 0
        self.labels = np.zeros(len(X_train))

        while True:
            self.WCSS = 0
            for i in range(len(X_train)):
                diff = self.centroids - X_train[i]
                dist = np.sum(diff ** 2, axis=1)
                self.labels[i] = np.argmin(dist)

            prev_centroids = self.centroids.copy()

            for i in range(self.K):
                list = []
                for j in range(len(X_train)):
                    if self.labels[j] == i:
                        list.append(X_train[j])

                self.centroids[i] = np.mean(list, axis=0)

            for i in range(len(X_train)):
                diff = self.centroids[int(self.labels[i])] - X_train[i]
                self.WCSS += np.sum(diff ** 2)
            
            if np.all(prev_centroids == self.centroids):
                break

        return self.labels



    def predict(self, X_test):
        labels = np.zeros(len(X_test))
        for i in range(len(X_test)):
            diff = self.centroids - X_test[i]
            dist = np.sum(diff ** 2, axis=1)
            labels[i] = np.argmin(dist)

        return labels

    def getCost(self):
        return self.WCSS
    