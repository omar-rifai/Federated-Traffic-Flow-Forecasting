# %%
from script.util import train_LSTM, LSTMModel
from script.util import my_data_loader
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
import copy

def init_model(n_nodes,main_model):
    return {k:copy.deepcopy(main_model) for k in range(n_nodes)}
    
def send_model(main_model, model_dict, number_of_nodes): #send model to nodes
    with torch.no_grad():
        for i in range(number_of_nodes):
            state_dict= copy.deepcopy(main_model.state_dict())
            model_dict[i].load_state_dict(state_dict)
    return model_dict

def fedavg(main_model, model_dict, number_of_nodes): 
    state_dict = model_dict[0].state_dict()
    for name, param in model_dict[0].named_parameters():
        for i in range(1, number_of_nodes):
            state_dict[name]=  state_dict[name] + model_dict[i].state_dict()[name]
        state_dict[name] = state_dict[name]/number_of_nodes
    new_model = copy.deepcopy(main_model)
    new_model.load_state_dict(state_dict)
    return new_model

# %%
#define local dataset for each nodes
def local_dataset(df, nodes,train_len=0 ):
    if nodes == 0 :
        nodes = len(df.columns)
    if train_len == 0:
        train_len= len(df)
    # Define the sliding window size and stride
    window_size = 7
    stride = 1
    data_dict={}
    for i in range(nodes): 
        # Create datasets and data loaders for training, validation, and test sets
    
        train_data= pd.DataFrame(df.iloc[:,i][:int(train_len*0.7)])
        val_data =  pd.DataFrame(df.iloc[:,i][int(train_len*0.7): int(train_len*0.85)])
        test_data = pd.DataFrame(df.iloc[:,i][int(train_len*0.85):])

        
        data_dict[i]={'train':my_data_loader(train_data),'val':my_data_loader(val_data),'test':my_data_loader(test_data),'test_data':test_data}
    return data_dict

# %%
import torch.cuda
def train_local_model(model_number,local_model,train_loader, val_loader,num_epochs = 200):
  # Train your model and evaluate on the validation set
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(device)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = local_model(inputs.float())
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_losses.append(loss.item())
        val_loss = 0.0
        
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = local_model(inputs.float())
            loss = criterion(outputs, targets.float())
            val_loss += loss.item()            
        val_loss /= len(val_loader)
        valid_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(local_model.state_dict(), 'best_model_'+str(model_number)+'.pth')
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    best_model =  copy.deepcopy(local_model)
    best_model.load_state_dict(torch.load('best_model_'+str(model_number)+'.pth'))
    return best_model

# %%
def rmspe(y_true, y_pred):
    EPSILON = 1e-10
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + EPSILON))))) * 100

# %%
# n_nodes = 'number of nodes'
# main_model = 'Model central'
# model_dict = init_nodes_dict(n_nodes, main_model) dictionnary of local models
def fed_training_plan(data_dict,rounds=3,nodes=3,epoch=200):
    nodes = len(data_dict)
    main_model = LSTMModel(input_size=1, hidden_size=32, num_layers=6, output_size=1)
    model_dict = init_model(nodes,main_model)
    for round in range(1,rounds+1):
        print('INIT ROUND {} :'.format(round))
        model_dict = send_model(main_model, model_dict, 3)
        for i in range(nodes):
            print('Training node {} for round {}'.format(i, round))
            model_dict[i] = train_local_model(i,model_dict[i],data_dict[i]['train'], data_dict[i]['val'],epoch)
        print('FedAVG for round {}:'.format(round))
        main_model = fedavg(main_model, model_dict, nodes)
        print('Done')
        torch.save(main_model.state_dict(), './model_round_{}.pth'.format(round))
    print("FedAvg All Rounds Complete !")

# %%



