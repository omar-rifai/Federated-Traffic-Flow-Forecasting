###############################################################################
# Libraries
###############################################################################
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


# def compute_percentage_change(df, metric):
#     df_final = pd.DataFrame()
#     df_index = df.index
#     for index_one in df_index:
#         for index_two in df_index:
#             if(index_one != index_two):
#                 temp = \
#                 pd.DataFrame(
#                     (((final_results_groupby_captor.loc[index_one].loc[metric] - final_results_groupby_captor.loc[index_two].loc[metric]) / (final_results_groupby_captor.loc[index_two].loc[metrics_ratio]).abs()) * 100)).T
#                 temp.index = [f"(({index_one} - {index_two}) / {index_two}) * 100"]
#                 temp.index.name = f"{metric}"
#                 temp = temp[["mean","std"]]
#                 df_final = pd.concat([df_final, temp],axis=0) 
#     return df_final

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
    st.write(path_results)