import numpy as np
from sklearn.metrics import  mean_squared_error


def rmse(y_true,y_pred):  
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmspe(y_true, y_pred, EPSILON=0):
    # The epsilon parameter move the time series away from zero values of values epsilon
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + EPSILON)))) * 100)

def mape(y_true, y_pred, EPSILON=0):
    # The epsilon parameter move the time series away from zero values of values epsilon
    return np.mean(np.abs((y_true - y_pred)/(y_true + EPSILON)))*100   

def maape(y_true,y_pred,EPSILON=0):
    # The epsilon parameter move the time series away from zero values of values epsilon
    return (np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true +EPSILON))))*100)

def rmsse(y_true, y_pred):
    # Calculate the numerator (RMSE)
    numerator = np.sqrt(np.mean(np.square(y_true - y_pred)))

    # Calculate the denominator (scaled error)
    denominator = np.sqrt(np.mean(np.square(y_true[1:] - y_true[:-1])))

    # Calculate the RMSSE
    rmsse = numerator / denominator

    return rmsse

