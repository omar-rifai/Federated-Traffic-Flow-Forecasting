import streamlit as st
import json
import numpy as np
import folium
import glob


# A dictionary to map the options to their aliases
OPTION_ALIASES = {
    "time_serie_percentage_length": "Choose the time series length used to train the model",
    "number_of_nodes": "Choose the number of sensors",
    "window_size": "Choose the windows size",
    "prediction_horizon": "Choose how far you want to see in the future",
    "model": "Choose the model"
}


@st.cache_resource
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


@st.cache_resource
def load_numpy(path):
    return np.load(path)


@st.cache_resource
def map_path_experiments_to_params(path_files, params_config_use_for_select):
    """
    Map all path experiments with parameters of their config.json

    Parameters:
    -----------
        path_files :
            The path to the config.json of all experiments
        params_config_use_for_select :
            The parameters use for the selection

    Returns:
        The mapping between path to the experiments and parameters of the experiements
        exemple :
        mapping = {
            "nb_node" : {
                3: [path_1, path_3],
                4: [path_4]
            },
            windows_size: {
                1: [path_1],
                5: [path_3],
                8: [path_4]
            }
        }
    """
    mapping_path_with_param = {}
    for param in params_config_use_for_select:
        mapping_path_with_param[param] = {}
        for file in path_files:
            with open(file) as f:
                config = json.load(f)
            if config[param] in mapping_path_with_param[param].keys():
                mapping_path_with_param[param][config[param]].append(file)
            else:
                mapping_path_with_param[param][config[param]] = [file]
    return mapping_path_with_param


def format_windows_prediction_size(value):
    return f"{int((value * 5 ) / 60)}h (t+{(value)})"


#  A function to convert the option to its alias
def format_option(option):
    return OPTION_ALIASES.get(option, option)


def format_radio(value):
    from config import Params
    params = Params(f'{value}')
    return f"{params.model} | Sensor(s): ({params.number_of_nodes}) {params.nodes_to_filter} \
    | prediction {format_windows_prediction_size(params.prediction_horizon)} \
    | the length of the time series used {params.time_serie_percentage_length * 100}%"


def selection_of_experiment():
    """
    Create the visual for choosing an experiment

    Parameters:
    -----------


    Returns:
        return the path to the experiment that the user choose
    """

    experiments = "./experiments/"  # PATH where your experiments are saved
    if path_files := glob.glob(f"./{experiments}**/config.json", recursive=True):

        options = list(OPTION_ALIASES.keys())
        map_path_experiments_params = map_path_experiments_to_params(path_files, options)

        selected_options = st.multiselect(
            "Choose the options you want to select",
            options,
            format_func=format_option,
            default=["number_of_nodes", "prediction_horizon"]
        )
        if (len(selected_options) > 0):
            test = {}
            first_option = selected_options[0]
            test[first_option] = {}
            test[first_option]["select"] = st.selectbox(
                format_option(first_option),
                map_path_experiments_params[first_option].keys()
            )
            test[first_option]["path_file"] = map_path_experiments_params[first_option][test[first_option]["select"]]
            for i in range(1, len(selected_options)):
                option_filtered = filtering_path_file(map_path_experiments_params[selected_options[i]], map_path_experiments_params[selected_options[i - 1]][test[selected_options[i - 1]]["select"]])
                test[selected_options[i]] = {}
                test[selected_options[i]]["select"] = st.selectbox(
                    format_option(selected_options[i]),
                    option_filtered.keys()
                )
                test[selected_options[i]]["path_file"] = option_filtered[test[selected_options[i]]["select"]]
            select_exp = st.radio("Choose", list(test[selected_options[len(selected_options) - 1]]["path_file"]), format_func=format_radio)
            return ("\\".join(select_exp.split("\\")[:-1]))
        return None


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def create_circle_precision_predict(marker_location, value_percent, map_folium, color):
    """
    Draw a circle at the position of the marker.

    Parameters:
    ----------
        marker_location (Marker Folium)

        value_percent (float)

        map_folium (Map Folium)

        color :
            Hex code HTML
    """
    lat, long = marker_location
    folium.Circle(location=[lat + 0.0020, long + 0.0018], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
    folium.Circle(location=[lat + 0.0020, long + 0.0018], color=color, radius=100 * value_percent, fill=True, opacity=0, fill_opacity=1, fill_color=color).add_to(map_folium)
    folium.map.Marker([lat + 0.0022, long + 0.0014], icon=folium.features.DivIcon(html=f"<div style='font-weight:bold; font-size: 15pt; color: black'>{int(value_percent * 100)}%</div>")).add_to(map_folium)
    # folium.Circle(location=[lat,long], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
    # folium.Circle(location=[lat,long], color=color, radius=100*value_percent, fill=True, opacity=0, fill_opacity=1, fill_color=color).add_to(map_folium)


def get_color_fed_vs_local(fed_value, local_value, superior=True):
    red = "#fe7597"
    green = "#75ff5b"
    if (superior):
        return (green, red) if ((fed_value) >= (local_value)) else (red, green)
    return (green, red) if ((fed_value) < (local_value)) else (red, green)
