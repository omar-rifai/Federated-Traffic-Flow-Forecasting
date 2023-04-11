# %%
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
import networkx as nx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
def create_graph(df):
    """
    create_graph(df)
    Create a graph from the PeMS distance.csv (using the from to data) 
    extracted dataframe were each sensor is a node where the distance between 
    each nodes correspond to the edges cost   
    """
    # Create a new NetworkX graph object
    G = nx.Graph()

    # Iterate over each row in the DataFrame and add nodes and edges to the graph
    for i, row in df.iterrows():
        # Add the "from" node to the graph if it doesn't already exist
        if not G.has_node(row["from"]):
            G.add_node(row["from"])
        # Add the "to" node to the graph if it doesn't already exist
        if not G.has_node(row["to"]):
            G.add_node(row["to"])
        # Add the edge between the "from" and "to" nodes with the cost as the edge weight
        G.add_edge(row["from"], row["to"], weight=row["cost"])
    return G 
    
def subgraph_dijkstra(G,node, n_neighbors):
    """ 
    subgraph_dijkstra(G , node , n_neighbors)
    G : NetworkX graph, node : node number to see neighbors, n_neighbors : Number of neighbors for subgraph
    Construct a subgraph of G using the n-nearest neighbors from the selected nodes of the graph. 
    """ 
    # Compute the shortest path from node 0 to all other nodes
    distances = nx.single_source_dijkstra_path_length(G, node)

    # Sort the nodes based on their distance from node 0
    nearest_nodes = sorted(distances, key=distances.get)[:n_neighbors+1]

    # Construct the subgraph by including only the selected nodes and their edges
    subgraph = G.subgraph(nearest_nodes)
    return subgraph

# %%

# Define the sliding window size and stride
# Define a PyTorch dataset to generate input/target pairs for the LSTM model
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=7, stride=1):
        self.data = data
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        inputs = self.data[idx:idx+self.window_size]
        target = self.data[idx+self.window_size]
        return inputs, target

# Define your LSTM model here with 6 LSTM layers and 1 fully connected layer
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size,output_size, num_layers=6):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# %%
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def ExpSmooth(PeMS,alpha=0.2):
    # Apply exponential smoothing to the time serie
    for i in range(len(PeMS.columns)):
        y = PeMS[PeMS.columns[i]]
        model = ExponentialSmoothing(y).fit(smoothing_level=alpha)
        smooth = model.fittedvalues
        PeMS[PeMS.columns[i]] = smooth
    return PeMS

# %%
def my_data_loader(data, window_size = 7, stride = 1):
    dataset = TimeSeriesDataset(data.values, window_size, stride)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    loader = [(inputs.to(device), targets.to(device)) for inputs, targets in loader]
    return loader
    

# %%
def experiment_dataset(cluster_size, df, train_len=0):
    if train_len == 0:
        train_len = len(df)

    cluster_dict={"size":cluster_size}
    for i in range(len(PeMS.columns)+1-cluster_size):
        model = LSTMModel(input_size=cluster_size, hidden_size=32, num_layers=layers, output_size=cluster_size)
        
        train_data= df[df.columns[i:i+cluster_size]][:int(train_len*0.7)]
        val_data =  df[df.columns[i:i+cluster_size]][:int(train_len*0.7): int(train_len*0.85)]
        test_data = df[df.columns[i:i+cluster_size]][int(train_len*0.85):]
        
        train_loader = my_data_loader(train_data)
        val_loader = my_data_loader(val_data)
        test_loader = my_data_loader(test_data) 
            
        cluster_dict[i]={"model":model,"train":train_loader,"val":val_loader,"test":test_loader}
    with open('./experiment/clusterS{}l{}.pkl'.format(cluster_size,train_len), 'wb') as f:
        pickle.dump(cluster_dict, f)

# %%
def experiment_dataset_subgraph(cluster_size, G , df , train_len=0):
    if train_len == 0:
        train_len = len(df)
    cluster_dict={"size":cluster_size}
    for i in PeMS.columns:
        # Compute the shortest path from node 0 to all other nodes
        distances = nx.single_source_dijkstra_path_length(G, i)

        # Sort the nodes based on their distance from node 0
        nearest_nodes = sorted(distances, key=distances.get)[:cluster_size]

        # Construct the subgraph by including only the selected nodes and their edges
        subgraph = subgraph_dijkstra(G,i, cluster_size-1)
        nodes = list(subgraph.nodes)
        model = LSTMModel(input_size=cluster_size, hidden_size=32, num_layers=layers, output_size=cluster_size)
        
        train_data= df[nodes][:int(train_len*0.7)]
        val_data =  df[nodes][:int(train_len*0.7): int(train_len*0.85)]
        test_data = df[nodes][int(train_len*0.85):]

        train_loader = my_data_loader(train_data)
        val_loader = my_data_loader(val_data)
        test_loader = my_data_loader(test_data) 
        
        cluster_dict[i]={"model":model,"train":train_loader,"val":val_loader,"test":test_loader,"nodes": nodes}
    
    with open('./experiment/clusterGsize{}l{}.pkl'.format(cluster_size,train_len), 'wb') as f:
        pickle.dump(cluster_dict, f)

# %%
import os
def train_LSTM(model,train_loader,val_loader, num_epochs = 200):
    # Train your model and evaluate on the validation set
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    best_model =  copy.deepcopy(model)
    best_model.load_state_dict(torch.load('best_model.pth'))
    os.remove("./best_model.pth")
    return best_model

# %%



