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