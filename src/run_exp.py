import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import copy

import pickle
from utils_graph import create_graph, subgraph_dijkstra

flow_file= "./data/PEMS04/pems04.npz"
csv_file = "./data/PEMS04/distance.csv"

data = np.load(flow_file)
graph_df = pd.read_csv(csv_file)
TS = data['data']
flow = TS[:,:,0]

# flow dict 100 time series is the sensor number and the value the traffic flow times serie
flow_dict={k:flow[:,k] for k in range(307)}

# list of the first 10 connected sensor, each sensor traffic flow is contained in PeMS 
PeMS = pd.DataFrame(flow_dict)

# time serie of sensor k
#creation of the datetime index
start_date = "2018-01-01 00:00:00"
end_date = "2018-02-28 23:55:00"
interval = "5min"
index = pd.date_range(start=start_date, end=end_date, freq=interval)
PeMS = PeMS.set_index(index)


import matplotlib.pyplot as plt
import networkx as nx

plt.figure(figsize=(13, 5))

graph_distance = create_graph(graph_df)

subgraph_distance = subgraph_dijkstra(graph_distance, node=0, n_neighbors=100)



# Change PeMS only to the subgraph
PeMS = PeMS[list(subgraph_distance .nodes)]

#Sort time series by mean traffic flow
mean_flow = PeMS.mean().sort_values()
#Index of sensor sort by mean traffic flow
mean_flow_index = mean_flow.index


column_order = list(mean_flow_index)
PeMS =PeMS.reindex(columns=column_order)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the sliding window size and stride
window_size = 7
stride = 1
layers = 6



# Time Series Simple exponential smoothing
alpha = 0.2
for i in range(100):
    y = PeMS[PeMS.columns[i]]
    smoothed_values = [y[0]]
    for j in range(1, len(y)):
        smoothed_value = alpha * y[j] + (1 - alpha) * smoothed_values[-1]
        smoothed_values.append(smoothed_value)
    PeMS[PeMS.columns[i]] = smoothed_values

# Normalizing Data
maximum = PeMS.max().max()
PeMS = PeMS/maximum



import torch.cuda
def train_model(model,train_loader, val_loader):

  # Train your model and evaluate on the validation set
    num_epochs = 100
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_losses.append(loss.item())
        val_loss = 0.0
        
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float())
            val_loss += loss.item()            
        val_loss /= len(val_loader)
        valid_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss*maximum:.4f}, Validation Loss: {val_loss*maximum:.4f}')
    best_model =  copy.deepcopy(model)
    best_model.load_state_dict(torch.load('best_model.pth'))
    return best_model

#initialize the experiment datasets as pickle object
for i in [1,5,10,15]:
   experiment_dataset(i,PeMS)
# iterate on cluster size i

for i in [5,10,15]:

# load the experiment datasets from pickle object 
    with open('./experiment/clusterS{}.pkl'.format(i), 'rb') as f:
        my_dict = pickle.load(f)
        # iterate on number of cluster 100-i+1
        for j in range(100-i+1):
            train = my_dict[j]["train"]
            val = my_dict[j]["val"]
            model = my_dict[j]["model"]
            model = train_model(model,train, val)
            my_dict[j]["model"]=copy.deepcopy(model)
    with open('./experiment/clusterS{}.pkl'.format(i), 'wb') as f:
        pickle.dump(my_dict, f)

# # Exp using graph

#initialize the experiment datasets as pickle object
for i in [1,5,10,15]:
   experiment_dataset_subgraph(i,PeMS)
# iterate on cluster size i
for i in [1,5,10,15]:
# load the experiment datasets from pickle object 
    with open('./experiment/clusterGsize{}.pkl'.format(i), 'rb') as f:
        my_dict = pickle.load(f)
        # iterate on number of cluster 100-i+1
        for j in range(100-i+1):
            train = my_dict[j]["train"]
            val = my_dict[j]["val"]
            model = my_dict[j]["model"]
            model = train_model(model,train, val)
            my_dict[j]["model"]=copy.deepcopy(model)
    with open('/experiment/clusterGsize{}.pkl'.format(i), 'wb') as f:
        pickle.dump(my_dict, f)


