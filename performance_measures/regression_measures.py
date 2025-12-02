import numpy as np

class RegressionMeasures:
    def __init__(self):
        pass

    def MSE(self,y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def variance(self, y_pred):
        return np.var(y_pred)
    
    def standard_deviation(self, y_pred):
        return np.std(y_pred)
    
    def RMSE(self,y_true, y_pred):
        return np.sqrt(self.MSE(y_true, y_pred))
    
    def R_squared(self, y_true, y_pred):
        ss_res  = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    