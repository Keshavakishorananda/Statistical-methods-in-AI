import random
import numpy as np

# class for linear regression
class linear_regression():
    def __init__(self, k, n, seed, r=0, regularizer=None):
        self.k = k + 1
        self.n = n
        self.seed = seed
        self.r = r
        self.regularizer = regularizer

        if self.r != 0 and self.regularizer == None:
            raise ValueError('Regularizer must be provided if r is not 0')
        elif self.r == 0 and self.regularizer == None:
            self.regularizer = 'l1'
        elif self.r == 0 and self.regularizer != None:
            self.regularizer = regularizer
                

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        

        random.seed(self.seed)
        self.parameters = []
        for _ in range(self.k):
            self.parameters.append(random.uniform(-1, 1))


    def update_parameters(self):
        if self.regularizer == 'l2':
            gradient = []

            for _ in range(self.k):
                gradient.append(0)
            
            for x,y in zip(self.x_train, self.y_train):
                pred = 0
                for i in range(self.k):
                    pred += (self.parameters[i])*(pow(x, i))

                for i in range(self.k):
                    gradient[i] = gradient[i] + (y-pred)*(-2)*pow(x,i) + 2*(self.r)*(self.parameters[i])
                                    
            for i in range(self.k):
                gradient[i] /= len(self.x_train)

            for i in range(self.k):
                self.parameters[i] = self.parameters[i] - (self.n * gradient[i])



        elif self.regularizer == 'l1':
            gradient = []

            for _ in range(self.k):
                gradient.append(0)
            
            for x,y in zip(self.x_train, self.y_train):
                pred = 0
                for i in range(self.k):
                    pred += (self.parameters[i])*(pow(x, i))

                for i in range(self.k):
                    gradient[i] = gradient[i] + (y-pred)*(-2)*pow(x,i) + (self.r)*(1 if self.parameters[i] > 0 else -1)
                
            for i in range(self.k):
                gradient[i] /= len(self.x_train)

            for i in range(self.k):
                self.parameters[i] = self.parameters[i] - (self.n * gradient[i])

        else:
            pass
        
    

    def predict(self, x):
        pred = 0
        for i in range(self.k):
            pred += (self.parameters[i])*(pow(x,i))

        return pred

        
