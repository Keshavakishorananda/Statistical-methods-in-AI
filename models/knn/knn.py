import numpy as np

class distance():
    def __init__(self):
        pass

    def Eucledian(self, a , b):
        dist = np.sqrt(np.sum((a-b)**2))
        return dist
    
    def Manhattan(self, a ,b):
        dist = np.sum(np.abs(a-b))
        return dist
    
    def Cosine(self, a, b):
        dist = np.dot(a,b) / (np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b)))
        return dist
    

class KNN(distance):
    # Initialize the class with attributes
    def __init__(self, k, dist_metric):
        super().__init__()
        self.k = k
        self.dist_mertic = dist_metric


    # To calculate distance between datapoints along with finding distance metric
    def find_dist(self, a, b):
        if self.dist_mertic == 'Eucledian':
            return self.Eucledian(a, b)
        if self.dist_mertic == 'manhattan':
            return self.Manhattan(a, b)
        if self.dist_mertic == 'cosine':
            return self.Cosine(a, b)
    

    # Give the data to the KNN model
    def train_model(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    

    # Predict the class of data point from given data
    def predict(self, x):

        # form a list of lists each contains label of datapoint and distance between them
        dist_list = []

        for i in range(len(self.x_train)):
            dist = self.find_dist(self.x_train[i], x)
            dist_list.append([self.y_train[i], dist])

        sorted_dist_list = sorted(dist_list, key=lambda x:x[1])

        sorted_dist_list = sorted_dist_list[:self.k]


        # form a dictionary to find the number of classes in nearest K points 
        class_dict = {}

        for i in sorted_dist_list:
            class_dict[i[0]] = class_dict.get(i[0], 0) + 1

        most_common_class = max(class_dict, key=class_dict.get)

        return most_common_class
    


class Distance_vect():
    def __init__(self):
        pass

    def euclidean(self, a, b):
        # LLM prompt : How to implement vetorization when a and b are different dimensions 
        # start code
        dist = np.sqrt(np.sum((a[:, np.newaxis] - b) ** 2, axis=2))
        return dist
        # end code

    def manhattan(self, a, b):
        dist = np.sum(np.abs(a[:, np.newaxis] - b), axis=2)
        return dist

    def cosine(self, a, b):
        a_norm = a / np.linalg.norm(a, axis=1)[:, np.newaxis]
        b_norm = b / np.linalg.norm(b, axis=1)[:, np.newaxis]
        
        similarity = np.dot(a_norm, b_norm.T)
        
        dist = 1 - similarity
        
        return dist
    

class KNN_optim(Distance_vect):
    def __init__(self, k, dist_metric):
        super().__init__()
        self.k = k
        self.dist_metric = dist_metric

    def find_dist(self, a, b):
        if self.dist_metric == 'euclidean':
            return self.euclidean(a, b)
        elif self.dist_metric == 'manhattan':
            return self.manhattan(a, b)
        elif self.dist_metric == 'cosine':
            return self.cosine(a, b)

    def train_model(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x, batch_size=100):
        # LLM prompt : How to implement batch wise vactorization
        # start code
        num_points = x.shape[0]
        predictions = []
        
        for i in range(0, num_points, batch_size):
            y = min(i+batch_size, num_points)
            x_batch = x[i:y]
            dist = self.find_dist(x_batch, self.x_train)
            
            nearest_indices = np.argsort(dist, axis=1)[:, :self.k]
            nearest_labels = self.y_train[nearest_indices]
            
            batch_predictions = np.array([np.bincount(labels).argmax() for labels in nearest_labels])
            predictions.extend(batch_predictions)
        
        return np.array(predictions)
        # end code

