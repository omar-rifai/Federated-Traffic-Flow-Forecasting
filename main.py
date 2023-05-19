
import torch

from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data
from src.models import LSTMModel
from src.utils_training import train_model
from src.utils_fed import local_dataset, fed_training_plan

import src.config
import sys



seed = 42
torch.manual_seed(seed)


# Get the path to the configuration file from the command-line arguments
if len(sys.argv) != 2:
    print("Usage: python3 experiment.py CONFIG_FILE_PATH")
    sys.exit(1)
config_file_path = sys.argv[1]

params = src.config.Params(config_file_path)

LSTM_input_size=1, 
LSTM_hidden_size=32,
LSTM_num_layers=6,
LSTM_output_size=1


#Load traffic flow dataframe and graph dataframe from PEMS
df_PeMS, distance = load_PeMS04_flow_data()
df_PeMS, adjmat = preprocess_PeMS_data(df_PeMS, distance, params.init_node, params.n_neighbours,
                                    params.smooth, params.center_and_reduce,
                                    params.normalize, params.sort_by_mean)

datadict = local_dataset(df_PeMS,
                        params.number_of_nodes,
                        len(df_PeMS),
                        window_size=params.window_size,
                        stride=params.stride,
                        target_size=params.target_size)

if params.num_epochs_local_no_federation:
    # Local Training 
    train_losses = {}
    val_losses = {}

    for j in range(params.number_of_nodes):
        local_model = LSTMModel(input_size= LSTM_input_size,
                                hidden_size= LSTM_hidden_size,
                                num_layers= LSTM_num_layers,
                                output_size= LSTM_output_size)
    
        data_dict = datadict[j]
        local_model, train_losses[j], val_losses[j] = train_model(local_model, data_dict['train'], data_dict['val'], 
                                                                  model_path ='./local{}.pth'.format(j),
                                                                  num_epochs=params.num_epochs_local_no_federation, 
                                                                  remove = False, learning_rate=params.learning_rate)

# Federated Learning Experiment
if params.num_epochs_local_federation:
    main_model = LSTMModel(input_size= LSTM_input_size,
                            hidden_size= LSTM_hidden_size,
                            num_layers= LSTM_num_layers,
                            output_size= LSTM_output_size)
    
    fed_training_plan(main_model, datadict, params.communication_rounds, params.num_epochs_local_federation)





