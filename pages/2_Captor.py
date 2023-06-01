###############################################################################
# Libraries
###############################################################################
import copy
import os
import glob

import json
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def filtering_path_file(file_dict, filter_list):
    """
    Returns a new dictionary that contains only the files that are in the filter list.

    Parameters
    ----------
    file_dict : dict
        The dictionary that maps some keys to lists of files.
    filter_list : list
        The list of files that are used as a filter.

    Return
    ------
    filtered_file_dict : dict
        The new dictionary that contains only the keys and files from file_dict that are also in filter_list.
    """
    filtered_file_dict = {}
    for key in file_dict.keys():
        for file in file_dict[key]:
            if file in filter_list:
                if key in filtered_file_dict:
                    filtered_file_dict[key].append(file)
                else:
                    filtered_file_dict[key] = [file]
    return filtered_file_dict


key_config_json = \
[
    "number_of_nodes",
    "window_size",
    "prediction_horizon",
    "model"
]

user_selection = \
{
    "nb_captors": {},
    "windows_size": {},
    "predictions_horizon": {},
    "models": {}
}

#######################################################################
# Loading Data
#######################################################################
experiments = "experiments"
if files := glob.glob(f"./{experiments}/**/config.json", recursive=True):
    for file in files:
        with open(file) as f:
            config_json = json.load(f)
        for key_config, selection in zip(key_config_json, user_selection):
            if config_json[key_config] in user_selection[selection]:
                user_selection[selection][config_json[key_config]].append(file)
            else:
                user_selection[selection][config_json[key_config]] = [file]
                
    nb_captor = st.selectbox('Choose the number of captor', user_selection["nb_captors"].keys())

    windows_size_filtered = filtering_path_file(user_selection["windows_size"], user_selection["nb_captors"][nb_captor])
    window_size = st.selectbox('Choose the windows size', windows_size_filtered.keys())
        
    horizon_filtered = filtering_path_file(user_selection["predictions_horizon"], windows_size_filtered[window_size])
    horizon_size = st.selectbox('Choose the prediction horizon', horizon_filtered.keys())
    
    models_filtered = filtering_path_file(user_selection["models"], horizon_filtered[horizon_size])
    model = st.selectbox('Choose the model', models_filtered.keys())

    if(len(models_filtered[model]) > 1):
        st.write("TODO : WARNING ! More than one results correspond to your research pick only one (see below)")

    path_results = ("\\".join(models_filtered[model][0].split("\\")[:-1]))

    with open(f"{path_results}/test.json") as f:
        results = json.load(f)
    
    def dataframe_results(dict_results, node):
        results = []
        captors = []
        for key in dict_results[node].keys():
            results.append(dict_results[node][key])
            captors.append(config_json["nodes_to_filter"][int(node)])

        df = pd.DataFrame(results)
        df.insert(0, "Captor", captors, True)
        df = df.set_index("Captor")
        return df

    nodes = results.keys()
    mapping_captor_and_node = {}
    for node in results.keys():
        mapping_captor_and_node[config_json["nodes_to_filter"][int(node)]] = node
        
    captor = st.selectbox('Choose the captor', mapping_captor_and_node.keys())

    local_node = results[mapping_captor_and_node[captor]]["local_only"]
    local_node = pd.DataFrame(local_node, columns=results[mapping_captor_and_node[captor]]["local_only"].keys(), index=["Captor alone"])
    
    federated_node = results[mapping_captor_and_node[captor]]["Federated"]
    federated_node = pd.DataFrame(federated_node, columns=results[mapping_captor_and_node[captor]]["Federated"].keys(), index=["Captor in Federation"])
        
    st.dataframe(pd.concat((local_node, federated_node ), axis=0), use_container_width=True)
    
    
    from os import makedirs
    import torch
    import importlib
    import contextlib

    from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data, local_dataset, plot_prediction
    from src.utils_training import train_model, testmodel
    from src.utils_fed import fed_training_plan
    from src.metrics import calculate_metrics, metrics_table, Percentage_of_Superior_Predictions
    import src.config
    import sys

    import json 
    params = src.config.Params('experiments/exp2/config.json')
    module_name = 'src.models'
    class_name = params.model
    module = importlib.import_module(module_name)
    model = getattr(module, class_name)

    input_size = 1
    hidden_size = 32
    num_layers = 6
    output_size = 1
    
    @st.cache_data
    def load_PeMS():
        df_PeMS, distance = load_PeMS04_flow_data()
        df_PeMS, adjmat, meanstd_dict = preprocess_PeMS_data(df_PeMS, distance, params.init_node, params.n_neighbours,
                                                        params.smooth, params.center_and_reduce,
                                                        params.normalize, params.sort_by_mean)
        return df_PeMS, adjmat, meanstd_dict
    
    @st.cache_data
    def create_data_dict(df_peMS):
        return local_dataset(
            df=df_PeMS,
            nodes=config_json["nodes_to_filter"],
            window_size=config_json["window_size"],
            stride=config_json["stride"],
            prediction_horizon=config_json["prediction_horizon"],
        )
    
    df_PeMS, adjmat, meanstd_dict = load_PeMS()
    datadict = create_data_dict(df_PeMS)
    

    y_true, y_pred, y_true_fed, y_pred_fed = {},{},{},{}

    @st.cache_data
    def wrap_testmodel(_test_loader, path=None, meanstd_dict =None, sensor_order_list =[], maximum= None):
        # Create model object here
        best_model = model(1,32,1)
        # Test model on data here
        y_true, y_pred = testmodel(best_model, _test_loader, path, meanstd_dict, sensor_order_list, maximum)
        return y_true, y_pred


    for node in range(len(config_json["nodes_to_filter"])):
        y_true[node], y_pred[node] = wrap_testmodel(datadict[node]['test'], f'{config_json["save_model_path"]}local{node}.pth', meanstd_dict = meanstd_dict, sensor_order_list=[params.nodes_to_filter[node]])  
        y_true_fed[node], y_pred_fed[node] = wrap_testmodel(datadict[node]['test'], f'{config_json["save_model_path"]}bestmodel_node{node}.pth', meanstd_dict = meanstd_dict, sensor_order_list=[params.nodes_to_filter[node]])

    def plot_comparison(y_true, y_pred, y_pred_fed, node):
        from src.metrics import rmse
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        test_set = datadict[int(node)]['test_data'] * meanstd_dict[config_json["nodes_to_filter"][int(node)]]['std'] + meanstd_dict[config_json["nodes_to_filter"][int(node)]]['mean']
        y_true, y_pred, y_pred_fed = y_true[int(node)], y_pred[int(node)], y_pred_fed[int(node)] 

        index= test_set.index

        def plot_slider(i):
            plt.figure(figsize=(20, 9))
            # Plot first subplot
            plt.subplot(2, 1, 1)
            plt.axvspan(index[i], index[i+ params.window_size -1], alpha=0.1, color='gray')
            plt.plot(index[i:i+params.window_size], test_set[i:i+params.window_size], label='Window')
            plt.plot(index[i+params.window_size-1:i+params.window_size + params.prediction_horizon], test_set[i+params.window_size -1 :i+params.window_size + params.prediction_horizon], label='y_true')
            plt.scatter(index[i+params.window_size:i+ params.window_size + params.prediction_horizon], y_pred_fed[i, :], color='blue', label='Federated prediction')
            plt.plot(index[i+params.window_size:i +params.window_size + params.prediction_horizon], y_pred_fed[i, :], color='blue', linestyle='-', linewidth=1)
            
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.xlabel('Time')
            plt.ylabel('Traffic Flow')
            plt.title("Federated Prediction for the {}".format(index[i].strftime('%Y-%m-%d')), fontsize=18, fontweight='bold')
            plt.legend(fontsize='large')

            
            # Plot second subplot
            plt.subplot(2, 1, 2)
            plt.axvspan(index[i], index[i+params.window_size -1], alpha=0.1, color='gray')
            plt.plot(index[i:i+ params.window_size], test_set[i:i+params.window_size], label='Window')
            plt.plot(index[i+params.window_size-1 :i+ params.window_size +params.prediction_horizon], test_set[i+params.window_size-1:i+params.window_size +params.prediction_horizon], label='y_true')
            plt.scatter(index[i+params.window_size:i+ params.window_size +params.prediction_horizon], y_pred[i, :], color='black', label='Local prediction')
            plt.plot(index[i+ params.window_size :i+ params.window_size +params.prediction_horizon], y_pred[i, :], color='black', linestyle='-', linewidth=1)
            
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            plt.xlabel('Time')
            plt.ylabel('Traffic Flow')
            plt.title("Local Prediction for the {}".format(index[i].strftime('%Y-%m-%d')), fontsize=18, fontweight='bold')
            plt.legend(fontsize='large')
            # plt.text(index[i+84], max(y_pred[i,:]+50), f'RMSE: {rmse(y_true_fed[i, :].flatten(), y_pred[i, :].flatten()):.2f}', fontsize='large', fontweight='bold')

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.5) 
            plt.show()
            st.pyplot(plt)
        #slider = widgets.IntSlider(min=0, max=len(y_true)-params.window_size, value=0, description='Index')

        # def update_slider_description(change):
            
        #     index_value = index[change.new + params.window_size].strftime('%H:%M')
        #     slider.description = f'Index: {index_value}'

        # slider.observe(update_slider_description, 'value')
        slider = st.slider('Select time?', 0, 130, 25)
        plot_slider(i=slider)
        # display(interactive_plot)

    plot_comparison(y_true, y_pred, y_pred_fed, mapping_captor_and_node[captor])
