import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.utils_graph import create_graph, subgraph_dijkstra 
from src.utils_data import load_PeMS04_flow_data, TimeSeriesDataset, my_data_loader, createLoaders, preprocess_PeMS_data, plot_prediction
from src.models import LSTMModel, testmodel 
from src.fedutil import local_dataset, fed_training_plan
from src.metrics import calculate_metrics
import networkx as nx

import sys
import json

# Set the random seed
seed = 42
torch.manual_seed(seed)

# Get the path to the configuration file from the command-line arguments
if len(sys.argv) != 2:
    print("Usage: python3 experiment.py CONFIG_FILE_PATH")
    sys.exit(1)
config_file_path = sys.argv[1]

# Load the configuration file using the provided path
with open(config_file_path) as f:
    config = json.load(f)

init_node = config['init_node']
n_neighbours = config['n_neighbours']
smooth = config['smooth']
center_and_reduce = config['center_and_reduce']
normalize = config['normalize']
sort_by_mean = config['sort_by_mean']
goodnodes = config['goodnodes']  #[118,168,261]
number_of_nodes =  config['number_of_nodes']  #[118,168,261]
model_input = config['model_input']
model_output = config['model_output']
communication_rounds = config['communication_rounds']
epochs = config['epochs']

#Load traffic flow dataframe and graph dataframe from PEMS
PeMS, distance = load_PeMS04_flow_data()


PeMS, adjmat, meanstd_dict = preprocess_PeMS_data(PeMS, distance, init_node, n_neighbours, smooth, center_and_reduce, normalize, sort_by_mean)
PeMS, adjmat, meanstd_dict = preprocess_PeMS_data(PeMS,distance,0,99,True,True,False,False)

G = create_graph(distance)
subgraph = subgraph_dijkstra(G,init_node,n_neighbours)
PeMS = PeMS[list(subgraph.nodes)]

for i in goodnodes:
    print("Nodes {} with mean traffic flow : {}".format(i,meanstd_dict[i]['mean']))
    print("Nodes {} with standard deviation : {}".format(i,meanstd_dict[i]['std']))

# plot graph
import networkx as nx
import matplotlib.pyplot as plt


pos = nx.spring_layout(G)

nx.draw(G, pos=pos, node_color='b', node_size=20)
nx.draw_networkx_nodes(G, pos=pos, nodelist=[118,168,261], node_color='r', node_size=20)
plt.title("Nodes Graph with nodes of interest in red")
plt.show()

# Plot time series 
plt.figure(figsize = (40,9))
plt.plot(PeMS[118])
plt.plot(PeMS[168])
plt.plot(PeMS[261])
plt.title('Our Sensor Traffic Flow')
plt.show()

# Federated Learning Experiment
datadict = local_dataset(PeMS[[118,168,261]],3,len(PeMS[[118,168,261]]))
main_model = LSTMModel(1,32,1)
fed_training_plan(main_model, datadict, communication_rounds, epochs)

# Training Local
#train_losses = {}
# val_losses = {}
# for j in range(len(datadict)):
#     data_dict = datadict[j]
#     new_model, train_losses[j], val_losses[j] = train_model(main_model, data_dict['train'], data_dict['val'], model_path ='./dummy{}.pth'.format(j),num_epochs=200, remove = False)

# plt.plot(val_losses[0],label='validation')
# plt.plot(train_losses[0],label= 'train')
# plt.legend()
# plt.show()

y_true, y_pred = testmodel(new_model, data_dict['test'], model, meanstd_dict,sensor_order_list=[118])

y_true, y_pred = testmodel(new_model,data_dict['test'],'local0.pth', meanstd_dict,sensor_order_list=[118])

plot_prediction(y_true,y_pred)

calculate_metrics(y_true,y_pred,1)