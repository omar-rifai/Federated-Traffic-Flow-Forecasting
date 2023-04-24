import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import networkx as nx

def rmse(y_true,y_pred):  
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmspe(y_true, y_pred):
    EPSILON = 1e-10
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + EPSILON)))) * 100)

def maape(y_true,y_pred):
    return (np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true ))))*100)