
import matplotlib.pyplot as plt

from src.utils_graph import create_graph, subgraph_dijkstra 
from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data, plot_prediction
from src.models import LSTMModel, train_model, testmodel
from src.fedutil import local_dataset, fed_training_plan
from src.metrics import calculate_metrics 

import sys
import json

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


G = create_graph(distance)
subgraph = subgraph_dijkstra(G,init_node,n_neighbours)
PeMS = PeMS[list(subgraph.nodes)]

for i in goodnodes:
    print("Nodes {} with mean traffic flow : {}".format(i,meanstd_dict[i]['mean']))
    print("Nodes {} with standard deviation : {}".format(i,meanstd_dict[i]['std']))

# Federated Learning Experiment
datadict = local_dataset(PeMS,number_of_nodes)
main_model = LSTMModel(input_size= model_input, hidden_size=32, output_size= model_output, num_layers=6)

fed_training_plan(datadict, rounds=communication_rounds, epoch=epochs)

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
for j in range(number_of_nodes):
    print(f'Node {j} for {communication_rounds}')
    y_true, y_pred = testmodel(main_model, data_dict[j]['test'],f"./model_round_{communication_rounds}.pth", meanstd_dict,sensor_order_list=list(goodnodes[j]))

    plot_prediction(y_true,y_pred)

    calculate_metrics(y_true,y_pred,1)
    y_true, y_pred = testmodel(main,data_dict['test'],f"./model_round_0.pth", meanstd_dict,sensor_order_list=list(goodnodes[j]))
    plot_prediction(y_true,y_pred)

    calculate_metrics(y_true,y_pred,1)