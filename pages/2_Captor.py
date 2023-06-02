###############################################################################
# Libraries
###############################################################################
import glob
import json
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import importlib
import numpy as np
import src.config
import json
from os import path

@st.cache_data
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

@st.cache_data
def load_numpy(path):
    return np.load(path)


def plot_comparison(y_pred, y_pred_fed, node):
    from src.metrics import rmse
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    test_set = load_numpy(f"{params.save_model_path}/test_data_{node}.npy")
    index = load_numpy(f"{params.save_model_path}/index_{node}.npy")
    index = pd.to_datetime(index, format='%Y-%m-%dT%H:%M:%S.%f')
    
    @st.cache_data
    def plot_slider(i):
        plt.figure(figsize=(30, 20))
        
        # FEDERATED
        # Plot first subplot
        plt.subplot(2, 1, 1)
        plt.axvspan(index[i], index[i + params.window_size - 1], alpha=0.1, color='gray')
        plt.plot(index[i:i + params.window_size], test_set[i:i + params.window_size], label='Window')
        plt.plot(index[i + params.window_size-1:i + params.window_size + params.prediction_horizon], test_set[i + params.window_size -1 :i + params.window_size + params.prediction_horizon], label='y_true', color="violet")
        plt.scatter(index[i + params.window_size :i + params.window_size + params.prediction_horizon], test_set[i + params.window_size :i + params.window_size + params.prediction_horizon], color="violet")
        plt.scatter(index[i + params.window_size: i + params.window_size + params.prediction_horizon], y_pred_fed[i, :], color='green', label='Federated prediction')
        plt.plot(index[i + params.window_size: i + params.window_size + params.prediction_horizon], y_pred_fed[i, :], color='green', linestyle='-', linewidth=1)
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.xlabel('Time')
        plt.ylabel('Traffic Flow')
        plt.title("Federated Prediction for the {}".format(index[i].strftime('%Y-%m-%d')), fontsize=18, fontweight='bold')
        plt.legend(fontsize='large')

        # LOCAL
        # Plot second subplot
        plt.subplot(2, 1, 2)
        plt.axvspan(index[i], index[i+params.window_size -1], alpha=0.1, color='gray')
        plt.plot(index[i:i+ params.window_size], test_set[i:i+params.window_size], label='Window')
        plt.plot(index[i+params.window_size-1 :i+ params.window_size +params.prediction_horizon], test_set[i+params.window_size-1:i+params.window_size +params.prediction_horizon], label='y_true', color="violet")
        plt.scatter(index[i + params.window_size :i + params.window_size + params.prediction_horizon], test_set[i + params.window_size :i + params.window_size + params.prediction_horizon], color="violet")
        plt.scatter(index[i+params.window_size:i+ params.window_size +params.prediction_horizon], y_pred[i, :], color='red', label='Local prediction')
        plt.plot(index[i+ params.window_size :i+ params.window_size +params.prediction_horizon], y_pred[i, :], color='red', linestyle='-', linewidth=1)
        
        
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
    
    slider = st.slider('Select time?', 0, len(index)-params.prediction_horizon-params.window_size, params.prediction_horizon)
    plot_slider(i=slider)



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
        select_exp = st.selectbox("Choose", models_filtered[model])
        select_exp = models_filtered[model].index(select_exp)
        path_results = ("\\".join(models_filtered[model][select_exp].split("\\")[:-1]))
        
    else:
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
    
    local_node = []
    if "local_only" in results[mapping_captor_and_node[captor]].keys():
        local_node = results[mapping_captor_and_node[captor]]["local_only"]
        local_node = pd.DataFrame(local_node, columns=results[mapping_captor_and_node[captor]]["local_only"].keys(), index=["Captor alone"])

    federated_node = []
    if "Federated" in results[mapping_captor_and_node[captor]].keys():
        federated_node = results[mapping_captor_and_node[captor]]["Federated"]
        federated_node = pd.DataFrame(federated_node, columns=results[mapping_captor_and_node[captor]]["Federated"].keys(), index=["Captor in Federation"])

    st.dataframe(pd.concat((local_node, federated_node ), axis=0), use_container_width=True)


    params = src.config.Params(f'{path_results}/config.json')
    if (path.exists(f'{params.save_model_path}y_true_local_{mapping_captor_and_node[captor]}.npy') and
        path.exists(f"{path_results}/y_pred_fed_{mapping_captor_and_node[captor]}.npy")):
        
        y_true = load_numpy(f"{path_results}/y_true_local_{mapping_captor_and_node[captor]}.npy")
        y_pred = load_numpy(f"{path_results}/y_pred_local_{mapping_captor_and_node[captor]}.npy")
        y_pred_fed = load_numpy(f"{path_results}/y_pred_fed_{mapping_captor_and_node[captor]}.npy")
        
        plot_comparison(y_pred, y_pred_fed, mapping_captor_and_node[captor])
