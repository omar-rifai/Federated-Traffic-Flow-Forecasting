###############################################################################
# Libraries
###############################################################################
from os import path


import glob
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


from utils_streamlit_app import load_numpy, map_path_experiments_to_params, filtering_path_file
from utils_streamlit_app import get_color_fed_vs_local, format_option, format_windows_prediction_size
from utils_streamlit_app import format_radio, style_dataframe
from config import Params

st.set_page_config(layout="wide")


def compute_absolute_error(path):
    y_true = load_numpy(f"{path}/y_true_local_{mapping_sensor_with_nodes_model_1[sensor_select]}.npy")
    y_pred_fed = load_numpy(f"{path}/y_pred_fed_{mapping_sensor_with_nodes_model_2[sensor_select]}.npy")
    return (np.abs(y_pred_fed.flatten() - y_true.flatten()))


def remove_outliers(data, threshold=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]


def plot_slider(experiment_path):
    ae_model_1 = compute_absolute_error(experiment_path[0])
    ae_model_2 = compute_absolute_error(experiment_path[1])

    def plot_box(title, ae, max_y_value, color):
        fig = go.Figure()
        box = go.Box(y=ae, marker_color=color, boxmean='sd', name=title, boxpoints=False)
        fig.add_trace(box)
        fig.update_layout(
            title={
                'text': f"{title}",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title=f"sensor_select {sensor_select}",
            yaxis_title="Trafic flow (absolute error)",
            yaxis=dict(range=[0, max_y_value]),
            font=dict(
                size=28,
                color="#FF7f7f"
            ),
            height=900, width=350
        )
        fig.update_traces(jitter=0)
        return fig

    ae_model_1 = remove_outliers(ae_model_1)
    ae_model_2 = remove_outliers(ae_model_2)

    max_y_value = max(max(ae_model_1), max(ae_model_2))

    model_1_color, model_2_color = get_color_fed_vs_local(np.mean(ae_model_1), np.mean(ae_model_2), superior=False)

    title_model_1 = paths_experiment_selected[0].split("\\")[1]
    fig_model_1 = plot_box(f"{title_model_1}", ae_model_1, max_y_value, model_1_color)

    title_model_2 = paths_experiment_selected[1].split("\\")[1]
    fig_model_2 = plot_box(f"{title_model_2}", ae_model_2, max_y_value, model_2_color)

    with st.spinner('Plotting...'):
        st.subheader(f"Comparison between two models on sensor {sensor_select} on the federated version (Aboslute Error)")
        _, c2_fed_fig, c3_local_fig, _ = st.columns((1, 1, 1, 1))
        with c2_fed_fig:
            st.plotly_chart(fig_model_1, use_container_width=False)
        with c3_local_fig:
            st.plotly_chart(fig_model_2, use_container_width=False)


OPTION_ALIASES = {
    "time_serie_percentage_length": "Choose the time series length used to train the model",
    "number_of_nodes": "Choose the number of sensors",
    "window_size": "Choose the windows size",
    "prediction_horizon": "Choose how far you want to see in the future",
    "model": "Choose the model"
}


def compare_config(path_file_1, path_file_2):
    with open(f"{path_file_1}/config.json") as f:
        config_1 = json.load(f)
    with open(f"{path_file_2}/config.json") as f:
        config_2 = json.load(f)

    return config_1["nodes_to_filter"] == config_2["nodes_to_filter"]


def selection_of_experiment():  # sourcery skip: assign-if-exp, extract-method
    experiments = "./experiments/"  # PATH where all the experiments are saved
    if path_files := glob.glob(f"./{experiments}**/config.json", recursive=True):

        options = list(OPTION_ALIASES.keys())
        map_path_experiments_params = map_path_experiments_to_params(path_files, options)

        selectbox_options = {}
        selectbox_options["number_of_nodes"] = {
            "select": st.selectbox(
                "Choose the number of sensors",
                map_path_experiments_params["number_of_nodes"].keys()
            )
        }

        options.remove("number_of_nodes")
        selected_options = st.multiselect(
            "Choose the options you want to use to filter the experiments",
            options,
            format_func=format_option,
            default=["prediction_horizon", "model"]
        )

        previous_path_file = map_path_experiments_params["number_of_nodes"][selectbox_options["number_of_nodes"]["select"]]
        for option in selected_options:
            option_filtered = filtering_path_file(map_path_experiments_params[option], previous_path_file)
            selectbox_options[option] = {
                "select": st.selectbox(
                    format_option(option),
                    option_filtered.keys(),
                    format_func=format_windows_prediction_size
                    if option
                    in ["window_size", "prediction_horizon"]
                    else str,
                )
            }
            previous_path_file = option_filtered[selectbox_options[option]["select"]]

        if len(previous_path_file) > 1:
            col1_model_1, col2_model_2 = st.columns(2)

            with col1_model_1:
                model_1 = st.radio(
                    "Choose the first model",
                    list(previous_path_file), key="model_1", format_func=format_radio)
            with col2_model_2:
                model_2 = st.radio(
                    "Choose the second model",
                    list(previous_path_file), key="model_2", format_func=format_radio)
            return [("\\".join(model_1.split("\\")[:-1])), ("\\".join(model_2.split("\\")[:-1]))]
        else:
            st.header(":red[Nothing match with your filter]")
            for option in selected_options:
                st.markdown(f"""* {option}""")
    return None

#######################################################################
# Main
#######################################################################


st.header("Comparison Models")
paths_experiment_selected = selection_of_experiment()
if (paths_experiment_selected is not None):
    path_model_1 = paths_experiment_selected[0]
    path_model_2 = paths_experiment_selected[1]

    with open(f"{path_model_1}/test.json") as f:
        results_1 = json.load(f)
    with open(f"{path_model_1}/config.json") as f:
        config_1 = json.load(f)

    with open(f"{path_model_2}/test.json") as f:
        results_2 = json.load(f)
    with open(f"{path_model_2}/config.json") as f:
        config_2 = json.load(f)

    mapping_sensor_with_nodes_model_1 = {}
    for node in results_1.keys():
        mapping_sensor_with_nodes_model_1[config_1["nodes_to_filter"][int(node)]] = node

    mapping_sensor_with_nodes_model_2 = {}
    for node in results_2.keys():
        mapping_sensor_with_nodes_model_2[config_2["nodes_to_filter"][int(node)]] = node

    sensor_select = st.selectbox('Choose the sensor_select', mapping_sensor_with_nodes_model_1.keys())

    metrics = list(results_1["0"]["local_only"].keys())
    multiselect_metrics = st.multiselect('Choose your metric(s)', metrics, ["RMSE", "MAE", "SMAPE", "Superior Pred %"])

    federated_node_model_1 = []
    if "Federated" in results_1["0"].keys():
        federated_node_model_1.append(results_1[mapping_sensor_with_nodes_model_1[sensor_select]]["Federated"])
        federated_node_model_1 = pd.DataFrame(federated_node_model_1, columns=multiselect_metrics, index=["sensor in Federation"])

    federated_node_model_2 = []
    if "Federated" in results_1["0"].keys():
        federated_node_model_2.append(results_2[mapping_sensor_with_nodes_model_2[sensor_select]]["Federated"])
        federated_node_model_2 = pd.DataFrame(federated_node_model_2, columns=multiselect_metrics, index=["sensor in Federation"])

    _, c2_title_df, _ = st.columns((2, 1, 2))

    with c2_title_df:
        st.header("Sensor in Federation")

    c1_model_1, c2_model_2 = st.columns(2)
    with c1_model_1:
        model_1_name = paths_experiment_selected[0].split("\\")[1]
        st.subheader(f"{model_1_name}")
        st.table(federated_node_model_1.style.set_table_styles(style_dataframe(federated_node_model_1)).format("{:.2f}"))
    with c2_model_2:
        model_2_name = paths_experiment_selected[1].split("\\")[1]
        st.subheader(f"{model_2_name}")
        st.table(federated_node_model_2.style.set_table_styles(style_dataframe(federated_node_model_2)).format("{:.2f}"))

    params_model_1 = Params(f'{path_model_1}/config.json')
    params_model_2 = Params(f'{path_model_2}/config.json')
    if (path.exists(f'{path_model_1}/y_pred_fed_0.npy') and
        path.exists(f"{path_model_2}/y_pred_fed_0.npy")):
        plot_slider([path_model_1, path_model_2])
