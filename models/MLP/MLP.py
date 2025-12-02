'''
1. manual derivative check
2. early stopping
3. multi label classification implementation
4. regression implementation
5. formatting the code
'''

import numpy as np
import sys
sys.path.append('../../')
import performance_measures.classification_measures as Classification_metrics
import performance_measures.regression_measures as Regression_metrics

class activation_functions():
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return ((exp_x).T / np.sum(exp_x, axis=1)).T

    def linear(self, x):
        return x
    
class activation_derivative():
    def __init__(self):
        pass

    def sigmoid_derivative(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def softmax_derivative(self, x):
        return x * (1 - x)
    
    def linear_derivative(self, x):
        return np.ones_like(x)

class MLP(activation_functions, activation_derivative):
    def __init__(self, hidden, neurons, active_fun):
        super().__init__()
        self.hidden = hidden
        self.neurons = neurons
        self.active_fun = active_fun

        if len(self.active_fun) != self.hidden+1:
            raise ValueError('Number of activation functions should be equal to hidden layers')
        if self.hidden < 0:
            raise ValueError('Number of hidden layers should be greater than 0')
        
    def activation(self, function):
        if function == 'sigmoid':
            return self.sigmoid
        elif function == 'relu':
            return self.relu
        elif function == 'tanh':
            return self.tanh
        elif function == 'softmax':
            return self.softmax
        elif function == 'linear':
            return self.linear
        else:
            raise ValueError('Activation function not found')
        
    def activation_derivative(self, function):
        if function == 'sigmoid':
            return self.sigmoid_derivative
        elif function == 'relu':
            return self.relu_derivative
        elif function == 'tanh':
            return self.tanh_derivative
        elif function == 'softmax':
            return self.softmax_derivative
        elif function == 'linear':
            return self.linear_derivative
        else:
            raise ValueError('Activation function not found')

    def loss_calculation(self, y_true, y_pred):
        if self.loss_function == 'MSE':
            return np.mean((y_true - y_pred)**2)/2
        if self.loss_function == 'cross_entropy': # Check this
            return -np.sum(y_true*np.log(y_pred+1e-9))/y_true.shape[0]
        if self.loss_function == 'binary_crossentropy':
            return -np.sum(y_true*np.log(y_pred+1e-9) + (1-y_true)*np.log(1-y_pred+1e-9))/y_true.shape[0]
        else:
            raise ValueError('Loss function not found')

    def loss_derivative(self, y_true, y_pred):
        if self.loss_function == 'MSE':
            return -(y_true - y_pred)
        if self.loss_function == 'cross_entropy': # Check this
            return y_pred - y_true
        else:
            raise ValueError('Loss function not found')

    def forward(self, x):
        self.forward_layers = []
        self.forward_activation = []

        if self.hidden == 0:
            self.forward_layers.append(np.dot(x, self.weights[0]) + self.biases[0])
            self.forward_activation.append(self.activation(self.active_fun[0])(self.forward_layers[0]))
            return self.forward_activation[-1]
        else:    
            # first layers
            self.forward_layers.append(np.dot(x, self.weights[0]) + self.biases[0])
            self.forward_activation.append(self.activation(self.active_fun[0])(self.forward_layers[0]))

            # hidden layers
            for i in range(1, self.hidden):
                self.forward_layers.append(np.dot(self.forward_activation[-1], self.weights[i]) + self.biases[i])
                self.forward_activation.append(self.activation(self.active_fun[i])(self.forward_layers[-1]))

            # last layer
            self.forward_layers.append(np.dot(self.forward_activation[-1], self.weights[-1]) + self.biases[-1])
            self.forward_activation.append(self.activation(self.active_fun[-1])(self.forward_layers[-1]))

            return self.forward_activation[-1]

    def backward(self, x, y, optimizer, batch_size=None):
        self.backward_deltas = []
        backward_errors = []

        if self.hidden == 0:
            if optimizer == 'batch':
                if self.loss_function == 'MSE':
                    backward_errors.append(np.multiply(self.loss_derivative(y, self.forward_activation[-1]), self.activation_derivative(self.active_fun[-1])(self.forward_layers[-1])))
                if self.loss_function == 'cross_entropy':
                    backward_errors.append(self.forward_activation[-1] - y)
                if self.loss_function == 'binary_crossentropy':
                    backward_errors.append(self.forward_activation[-1] - y)
                self.backward_deltas.append({
                    'weights' : np.dot(x.T, backward_errors[-1])/len(y),
                    'biases' : np.sum(backward_errors[-1], axis=0)/len(y)}
                    )
            
            if optimizer == 'stochastic':
                if self.loss_function == 'MSE':
                    backward_errors.append(np.multiply(self.loss_derivative(y[self.point].reshape(1, -1), self.forward_activation[-1][self.point].reshape(1, -1)), self.activation_derivative(self.active_fun[-1])(self.forward_layers[-1][self.point].reshape(1,-1))))
                if self.loss_function == 'cross_entropy':
                    backward_errors.append(self.forward_activation[-1][self.point].reshape(1,-1) - y[self.point].reshape(1,-1))
                if self.loss_function == 'binary_crossentropy':
                    backward_errors.append(self.forward_activation[-1][self.point].reshape(1,-1) - y[self.point].reshape(1,-1))
                self.backward_deltas.append({
                    'weights' : np.dot(x[self.point].reshape(1, -1).T, backward_errors[-1]),
                    'biases' : np.sum(backward_errors[-1], axis=0)
                })

            if optimizer == 'mini-batch':
                if self.loss_function == 'MSE':
                    backward_errors.append(np.multiply(self.loss_derivative(y[self.batch:self.batch + batch_size],self.forward_activation[-1][self.batch:self.batch+batch_size]), self.activation_derivative(self.active_fun[-1])(self.forward_layers[-1][self.batch:self.batch+batch_size])))
                if self.loss_function == 'cross_entropy':
                    backward_errors.append(self.forward_activation[-1][self.batch:self.batch+batch_size] - y[self.batch:self.batch+batch_size])
                if self.loss_function == 'binary_crossentropy':
                    backward_errors.append(self.forward_activation[-1][self.batch:self.batch+batch_size] - y[self.batch:self.batch+batch_size])
                self.backward_deltas.append({
                    'weights' : np.dot(x[self.batch:self.batch+batch_size].T, backward_errors[-1])/batch_size,
                    'biases' : np.sum(backward_errors[-1], axis=0)/batch_size
                })

            return self.backward_deltas

        else:
            #1. Optimizer = Bacth GD
            if optimizer == 'batch':
                # last layer
                if self.loss_function == 'MSE':
                    backward_errors.append(np.multiply(self.loss_derivative(y, self.forward_activation[-1]), self.activation_derivative(self.active_fun[-1])(self.forward_layers[-1])))
                if self.loss_function == 'cross_entropy':
                    backward_errors.append(self.forward_activation[-1] - y)
                if self.loss_function == 'binary_crossentropy':
                    backward_errors.append(self.forward_activation[-1] - y)
                self.backward_deltas.append({
                    'weights' : np.dot(self.forward_activation[-2].T, backward_errors[-1])/len(y),
                    'biases' : np.sum(backward_errors[-1], axis=0)/len(y)}
                    )


                # hidden layers
                for i in range(self.hidden-1, 0, -1):
                    backward_errors.append(np.multiply(np.dot(backward_errors[-1], self.weights[i+1].T),self.activation_derivative(self.active_fun[i])(self.forward_layers[i])))
                    self.backward_deltas.append({
                        'weights' : np.dot(self.forward_activation[i-1].T, backward_errors[-1])/len(y),
                        'biases' : np.sum(backward_errors[-1], axis=0)/len(y)
                    })

                # first layer
                backward_errors.append(np.multiply(np.dot(backward_errors[-1], self.weights[1].T),self.activation_derivative(self.active_fun[0])(self.forward_layers[0])))
                self.backward_deltas.append({
                    'weights' : np.dot(x.T, backward_errors[-1])/len(y),
                    'biases' : np.sum(backward_errors[-1], axis=0)/len(y)
                })
            
                self.backward_deltas = self.backward_deltas[::-1]
                return self.backward_deltas
        
            # 2. Optimizer = Stochastic GD
            if optimizer == 'stochastic':
                # last layer
                if self.loss_function == 'MSE':
                    backward_errors.append(np.multiply(self.loss_derivative(y[self.point].reshape(1, -1), self.forward_activation[-1][self.point].reshape(1, -1)), self.activation_derivative(self.active_fun[-1])(self.forward_layers[-1][self.point].reshape(1,-1))))
                if self.loss_function == 'cross_entropy':
                    backward_errors.append(self.forward_activation[-1][self.point].reshape(1,-1) - y[self.point].reshape(1,-1))
                if self.loss_function == 'binary_crossentropy':
                    backward_errors.append(self.forward_activation[-1][self.point].reshape(1,-1) - y[self.point].reshape(1,-1))
                self.backward_deltas.append({
                    'weights' : np.dot(self.forward_activation[-2][self.point].reshape(1,-1).T, backward_errors[-1]),
                    'biases' : np.sum(backward_errors[-1], axis=0)
                })

                # hidden layers
                for j in range(self.hidden-1, 0, -1):
                    backward_errors.append(np.dot(backward_errors[-1], self.weights[j+1].T) * self.activation_derivative(self.active_fun[j])(self.forward_layers[j][self.point].reshape(1,-1)))
                    self.backward_deltas.append({
                        'weights' : np.dot(self.forward_activation[j-1][self.point].reshape(1,-1).T, backward_errors[-1]),
                        'biases' : np.sum(backward_errors[-1], axis=0)
                    })

                # first layer
                backward_errors.append(np.dot(backward_errors[-1], self.weights[1].T) * self.activation_derivative(self.active_fun[0])(self.forward_layers[0][self.point].reshape(1,-1)))
                self.backward_deltas.append({
                    'weights' : np.dot(x[self.point].reshape(1, -1).T, backward_errors[-1]),
                    'biases' : np.sum(backward_errors[-1], axis=0)
                })

                self.point = (self.point + 1) % x.shape[0]

                self.backward_deltas = self.backward_deltas[::-1]
                return self.backward_deltas

            # 3. Optimizer = Mini-batch GD
            if optimizer == 'mini-batch':
                # last layer
                if self.loss_function == 'MSE':
                    backward_errors.append(np.multiply(self.loss_derivative(y[self.batch:self.batch + batch_size],self.forward_activation[-1][self.batch:self.batch+batch_size]), self.activation_derivative(self.active_fun[-1])(self.forward_layers[-1][self.batch:self.batch+batch_size])))
                if self.loss_function == 'cross_entropy':
                    backward_errors.append(self.forward_activation[-1][self.batch:self.batch+batch_size] - y[self.batch:self.batch+batch_size])
                if self.loss_function == 'binary_crossentropy':
                    backward_errors.append(self.forward_activation[-1][self.batch:self.batch+batch_size] - y[self.batch:self.batch+batch_size])
                self.backward_deltas.append({
                    'weights' : np.dot(self.forward_activation[-2][self.batch:self.batch+batch_size].T, backward_errors[-1])/batch_size,
                    'biases' : np.sum(backward_errors[-1], axis=0)/batch_size
                })

                # hidden layers
                for j in range(self.hidden-1, 0, -1):
                    backward_errors.append(np.dot(backward_errors[-1], self.weights[j+1].T) * self.activation_derivative(self.active_fun[j])(self.forward_layers[j][self.batch:self.batch+batch_size]))
                    self.backward_deltas.append({
                        'weights' : np.dot(self.forward_activation[j-1][self.batch:self.batch+batch_size].T, backward_errors[-1])/batch_size,
                        'biases' : np.sum(backward_errors[-1], axis=0)/batch_size
                    })

                # first layer
                backward_errors.append(np.dot(backward_errors[-1], self.weights[1].T) * self.activation_derivative(self.active_fun[0])(self.forward_layers[0][self.batch:self.batch+batch_size]))
                self.backward_deltas.append({
                    'weights' : np.dot(x[self.batch:self.batch+batch_size].T, backward_errors[-1])/batch_size,
                    'biases' : np.sum(backward_errors[-1], axis=0)/batch_size
                })

                self.batch = (self.batch + batch_size) % x.shape[0]

                self.backward_deltas = self.backward_deltas[::-1]
                return self.backward_deltas


    def update_weights(self, lr):
        for j in range(self.hidden+1):
            self.weights[j] -= lr * self.backward_deltas[j]['weights']
            self.biases[j] -= lr * self.backward_deltas[j]['biases']


    def train(self, x, y, optimizer, lr, epochs, batch_size, x_valid, y_valid):
        self.point = 0
        self.batch = 0
        
        if self.loss_function == 'cross_entropy':
            loss_train_epoch = []
            accuracy_train_epoch = []
            recall_train_epoch = []
            precision_train_epoch = []
            f1_train_epoch = []

            loss_valid_epoch = []
            accuracy_valid_epoch = []
            recall_valid_epoch = []
            precision_valid_epoch = []
            f1_valid_epoch = []

        if self.loss_function == 'MSE':
            MSE_valid_epoch = []
            RMSE_valid_epoch = []
            R_squared_valid_epoch = []

            MSE_train_epoch = []
            RMSE_train_epoch = []
            R_squared_train_epoch = []

        if self.loss_function == 'binary_crossentropy':
            Hamming_loss = []
            loss_train_epochs = []
            accuracy_train_epochs = []
            recall_train_epochs = []
            precision_train_epochs = []
            f1_train_epochs = []

            loss_valid_epochs = []
            accuracy_valid_epochs = []
            recall_valid_epochs = []
            precision_valid_epochs = []
            f1_valid_epochs = []


        for i in range(epochs):
            y_pred = self.forward(x)
            
            if self.loss_function == 'cross_entropy':        
                # Train set
                loss_train_epoch.append(self.loss_calculation(y, y_pred))
                print(f"Epoch {i+1} Loss: {loss_train_epoch[-1]}")

                y_pred_train = np.argmax(y_pred, axis=1)
                y_true_train = np.argmax(y, axis=1)
                
                metric_class = Classification_metrics.Measures(y_true_train, y_pred_train)
                accuracy_train_epoch.append(np.mean(y_pred_train == y_true_train))
                recall_train_epoch.append(metric_class.recall_macro())
                precision_train_epoch.append(metric_class.precision_macro())
                f1_train_epoch.append(metric_class.f1_score_macro())

                # Valid set
                y_pred_valid = self.forward(x_valid)
                loss_valid_epoch.append(self.loss_calculation(y_valid, y_pred_valid))

                y_pred_valid = np.argmax(y_pred_valid, axis=1)
                y_true_valid = np.argmax(y_valid, axis=1)

                metric_class = Classification_metrics.Measures(y_true_valid, y_pred_valid)
                accuracy_valid_epoch.append(np.mean(y_pred_valid == y_true_valid))
                recall_valid_epoch.append(metric_class.recall_macro())
                precision_valid_epoch.append(metric_class.precision_macro())
                f1_valid_epoch.append(metric_class.f1_score_macro())

            if self.loss_function == 'MSE':
                # Train set
                loss = self.loss_calculation(y, y_pred)
                print(f"Epoch {i+1} Loss: {loss}")

                if x_valid is None or y_valid is None:
                    MSE_train_epoch.append(loss)
                else:
                    MSE_train_epoch.append(loss)
                    RMSE_train_epoch.append(np.sqrt(loss))
                    R_squared_train_epoch.append(1 - np.sum((y - y_pred)**2)/np.sum((y - np.mean(y))**2))

                    # Valid set
                    y_pred_valid = self.forward(x_valid)

                    metric_class = Regression_metrics.RegressionMeasures()
                    MSE_valid_epoch.append(metric_class.MSE(y_valid, y_pred_valid))
                    RMSE_valid_epoch.append(metric_class.RMSE(y_valid, y_pred_valid))
                    R_squared_valid_epoch.append(metric_class.R_squared(y_valid, y_pred_valid))
            
            if self.loss_function == 'binary_crossentropy':
                loss = self.loss_calculation(y, y_pred)
                print(f"Epoch {i+1} Loss: {loss}")
                loss_train_epochs.append(loss)

                y_pred_train = np.round(y_pred)
                y_true_train = y


                # train set
                accuracy_train_epochs.append(np.mean(np.mean(y_pred_train == y_true_train, axis=0)))

                label_wise_precision = np.zeros(y_pred_train.shape[1])
                for i in range(y_pred_train.shape[1]):
                    label_wise_precision[i] = np.sum((y_pred_train[:, i] == 1) & (y_true_train[:, i] == 1)) / np.sum(y_pred_train[:, i] == 1)

                precision_train_epochs.append(np.mean(label_wise_precision))

                label_wise_recall = np.zeros(y_pred_train.shape[1])
                for i in range(y_pred_train.shape[1]):
                    label_wise_recall[i] = np.sum((y_pred_train[:, i] == 1) & (y_true_train[:, i] == 1)) / np.sum(y_true_train[:, i] == 1)

                recall_train_epochs.append(np.mean(label_wise_recall))


                # valid set
                y_pred_valid = self.forward(x_valid)
                loss_valid_epochs.append(self.loss_calculation(y_valid, y_pred_valid))

                y_pred_valid = np.round(y_pred_valid)
                y_true_valid = y_valid

                Hamming_loss.append(np.sum(np.abs(y_pred_valid - y_true_valid))/(len(y_pred_train[0]) * len(y_true_valid)))

                accuracy_valid_epochs.append(np.mean(np.mean(y_pred_train == y_true_train, axis=0)))

                label_wise_precision = np.zeros(y_pred_valid.shape[1])
                for i in range(y_pred_valid.shape[1]):
                    label_wise_precision[i] = np.sum((y_pred_valid[:, i] == 1) & (y_true_valid[:, i] == 1)) / np.sum(y_pred_valid[:, i] == 1)
                precision_valid_epochs.append(np.mean(label_wise_precision))

                label_wise_recall = np.zeros(y_pred_valid.shape[1])
                for i in range(y_pred_valid.shape[1]):
                    label_wise_recall[i] = np.sum((y_pred_valid[:, i] == 1) & (y_true_valid[:, i] == 1)) / np.sum(y_true_valid[:, i] == 1)
                recall_valid_epochs.append(np.mean(label_wise_recall))

            self.forward(x)
            self.backward(x, y, optimizer, batch_size)
            self.update_weights(lr)

        if self.loss_function == 'MSE':
            if x_valid is None or y_valid is None:
                return MSE_train_epoch
            return MSE_train_epoch, RMSE_train_epoch, R_squared_train_epoch, MSE_valid_epoch, RMSE_valid_epoch, R_squared_valid_epoch
        if self.loss_function == 'cross_entropy':
            return loss_train_epoch, accuracy_train_epoch, recall_train_epoch, precision_train_epoch, f1_train_epoch, loss_valid_epoch, accuracy_valid_epoch, recall_valid_epoch, precision_valid_epoch, f1_valid_epoch
        if self.loss_function == 'binary_crossentropy':
            return Hamming_loss,loss_train_epochs, accuracy_train_epochs, recall_train_epochs, precision_train_epochs, loss_valid_epochs, accuracy_valid_epochs, recall_valid_epochs, precision_valid_epochs

    def early_stopping(self, x, y, x_valid, y_valid, optimizer,lr, max_epochs=1000):
        prev_valid_loss = float('inf')

        for _ in range(max_epochs):
            self.forward(x)
            y_pred_valid = self.forward(x_valid)
            valid_loss = self.loss_calculation(y_valid, y_pred_valid)

            self.backward(x, 'batch', optimizer, 32)
            self.update_weights(lr)

            if (prev_valid_loss <= valid_loss):
                break


    def check_gradients(self, x, y, epsilon=1e-7):
        y_pred = self.forward(x)
        self.backward(x, y, 'batch')
        
        for layer in range(len(self.weights)):
            grad_approx_weights = np.zeros_like(self.weights[layer])
            grad_approx_biases = np.zeros_like(self.biases[layer])
            
            for i in range(self.weights[layer].shape[0]):
                for j in range(self.weights[layer].shape[1]):
                    self.weights[layer][i, j] += epsilon
                    loss_plus = self.loss_calculation(y, self.forward(x))
                    self.weights[layer][i, j] -= 2 * epsilon
                    loss_minus = self.loss_calculation(y, self.forward(x))
                    grad_approx_weights[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
                    self.weights[layer][i, j] += epsilon
            
            for i in range(self.biases[layer].shape[1]):
                self.biases[layer][0, i] += epsilon
                loss_plus = self.loss_calculation(y, self.forward(x))
                self.biases[layer][0, i] -= 2 * epsilon
                loss_minus = self.loss_calculation(y, self.forward(x))
                grad_approx_biases[0, i] = (loss_plus - loss_minus) / (2 * epsilon)
                self.biases[layer][0, i] += epsilon
            
            weight_diff = np.linalg.norm(self.backward_deltas[layer]['weights'] - grad_approx_weights) / (np.linalg.norm(self.backward_deltas[layer]['weights']) + np.linalg.norm(grad_approx_weights))
            bias_diff = np.linalg.norm(self.backward_deltas[layer]['biases'] - grad_approx_biases) / (np.linalg.norm(self.backward_deltas[layer]['biases']) + np.linalg.norm(grad_approx_biases))
            
            print(f"Layer {layer+1} weights relative difference: {weight_diff}")
            print(f"Layer {layer+1} biases relative difference: {bias_diff}")

        
    def fit(self, x, y, optimizer, loss_function, lr, epochs, batch_size, x_valid = None, y_valid=None):
        np.random.seed(0)
        self.weights = []
        self.biases = []
        self.loss_function = loss_function

        if self.hidden >= 1:
            self.weights.append(np.random.randn(x.shape[1], self.neurons[0]) * np.sqrt(2/x.shape[1]))
            self.biases.append(np.zeros((1, self.neurons[0])))
            for i in range(1, self.hidden):
                self.weights.append(np.random.randn(self.neurons[i-1], self.neurons[i]) * np.sqrt(2/self.neurons[i-1]))
                self.biases.append(np.zeros((1, self.neurons[i])))

            self.weights.append(np.random.randn(self.neurons[-1], y.shape[1])*np.sqrt(2/self.neurons[-1]))
            self.biases.append(np.zeros((1, y.shape[1])))
        if self.hidden == 0:
            self.weights.append(np.random.randn(x.shape[1], y.shape[1])*np.sqrt(2/x.shape[1]))
            self.biases.append(np.zeros((1, y.shape[1])))

        return self.train(x, y, optimizer, lr, epochs, batch_size, x_valid, y_valid)

    def predict(self, x):
        return self.forward(x)