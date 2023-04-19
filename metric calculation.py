
import pickle
from script.util import LSTMModel, rmspe, maape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from random import randint
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mean_absolute_error, mean_squared_error
from random import randint

# Use this 
PeMS = pd.read_csv("./data/PEMS04/PeMS04flow.csv",index_col=0)
maximum = PeMS.max().max()

def my_metrics_dict(dict, samples = 60):
    #Load the best model and evaluate on the test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss()
    size = dict["size"]
    for s in dict.keys():
        if s == "size":
            continue
        
        best_model = dict[s]["model"]
        best_model.double()
        device = torch.device('cuda:0')
        best_model.to(device)
        test_loader = dict[s]["test"]
        best_model.eval()

        # Evaluate the model on the test set
        test_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                x = torch.Tensor(inputs).unsqueeze(1).to(device)
                y = torch.Tensor(targets).unsqueeze(0).to(device)
                outputs = best_model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                # Save the predictions and actual values for plotting later
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())
                
        test_loss /= len(test_loader)
        # Concatenate the predictions and actuals
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        EPSILON = 1e-10
        metric_dict_multi = {}
        
        for j in range(samples): 
            metric_dict_multi[j] = {}
            for k in range(size): 
                y_pred = predictions[:,k]*maximum
                y_true = actuals[:,k]*maximum
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                mape = mean_absolute_percentage_error(y_true,y_pred+EPSILON)
                maape =  np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true + EPSILON))))*100
                number_of_zero = len(y_true[y_true ==0])
                metric_dict_multi[j][j+k]={"mae":mae,  "rmse":rmse,"maape":maape,"zero": y_true.min(),"mape":mape }
    return metric_dict_multi

def global_metric_dict(filelist):
    metric_dict = {}
    for file in filelist:   
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open('./experiment/'+file, 'rb') as f:
            ndict = pickle.load(f)
        metric = my_metrics_dict(ndict)
        metric_dict[file[:-4]]=metric
        with open('./experiment/metric{}.pkl'.format(file[:-4]), 'wb') as f:
            pickle.dump(metric_dict, f)
            
filelist2 = ['clusterS3.pkl','clusterS6.pkl','clusterS9.pkl']
global_metric_dict(filelist2)



