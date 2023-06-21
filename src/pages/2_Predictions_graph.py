###############################################################################
# Libraries
###############################################################################
from os import path


import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from annotated_text import annotated_text


from metrics import rmse
from config import Params
from utils_streamlit_app import load_numpy, selection_of_experiment
from utils_streamlit_app import get_color_fed_vs_local, style_dataframe

st.set_page_config(layout="wide")


def plot_prediction_graph(experiment_path):
    test_set = load_numpy(f"{experiment_path}/test_data_{mapping_sensor_with_nodes[sensor_select]}.npy")

    y_true = load_numpy(f"{experiment_path}/y_true_local_{mapping_sensor_with_nodes[sensor_select]}.npy")
    y_pred = load_numpy(f"{experiment_path}/y_pred_local_{mapping_sensor_with_nodes[sensor_select]}.npy")
    y_pred_fed = load_numpy(f"{experiment_path}/y_pred_fed_{mapping_sensor_with_nodes[sensor_select]}.npy")

    index = load_numpy(f"{experiment_path}/index_{mapping_sensor_with_nodes[sensor_select]}.npy")
    index = pd.to_datetime(index, format='%Y-%m-%dT%H:%M:%S.%f')

    slider = st.slider('Select the step (a step equal 5min)?', 0, len(index) - params.prediction_horizon - params.window_size - 1, 0)

    def plot_graph(color, label, title, y_pred, rmse_value, i):
        df = pd.DataFrame({'Time': index[i:i + params.window_size + params.prediction_horizon], 'Traffic Flow': test_set[i:i + params.window_size + params.prediction_horizon].flatten()})
        df['Window'] = df['Traffic Flow'].where((df['Time'] >= index[i]) & (df['Time'] < index[i + params.window_size]))
        df['y_true'] = df['Traffic Flow'].where((df['Time'] >= index[i + params.window_size - 1]))
        df[f'y_pred_{label}'] = np.concatenate([np.repeat(np.nan, params.window_size).reshape(-1, 1), y_pred[i, :]])
        df[f"y_pred_{label}_link_window"] = np.concatenate([np.repeat(np.nan, params.window_size).reshape(-1, 1), y_pred[i]])
        df[f"y_pred_{label}_link_window"].at[params.window_size - 1] = df['Window'].iloc[params.window_size - 1]

        std_true = np.std(df['y_true'].loc[params.window_size:])
        confidence_interval = 1.96 * std_true

        fig = px.line(
            df, x='Time',
            y=["Window"],
            color_discrete_sequence=['black']
        )
        fig.add_scatter(
            x=df['Time'],
            y=df['y_true'],
            mode='lines',
            marker=dict(color='blue'),
            name='y_true'
        )
        fig.add_scatter(
            x=df['Time'],
            y=df[f'y_pred_{label}'],
            mode='markers+lines',
            marker=dict(color=color),
            name=f'{label} RMSE : {rmse_value:.2f}'
        )
        fig.add_scatter(
            x=df['Time'],
            y=df[f"y_pred_{label}_link_window"],
            mode='lines',
            marker=dict(color=color),
            showlegend=False
        )
        fig.add_bar(
            x=df['Time'],
            y=(np.abs(df[f'y_pred_{label}'] - df['y_true'])),
            name='Absolute Error',
            marker=dict(color="#FFAB55")
        )
        fig.add_vrect(
            x0=index[i],
            x1=index[i + params.window_size - 1],
            fillcolor='gray',
            opacity=0.2,
            line_width=0
        )
        if render_confidence_interval:
            fig.add_scatter(
                x=np.concatenate([df['Time'], df['Time'][::-1]]),
                y=np.concatenate([df['y_true'] - confidence_interval, df['y_true'][::-1] + confidence_interval]),
                fill='toself',
                fillcolor='rgb(50,100,100)',
                line=dict(color='#000000'),
                opacity=0.3,
                hoverinfo='skip',
                showlegend=True,
                name="Confidence Interval"
            )
        fig.update_xaxes(
            title='Time',
            tickformat='%H:%M',
            dtick=3600000
        )
        fig.update_yaxes(
            title="Traffic Flow",
            range=[min(min(y_true.flatten()), min(y_pred.flatten())), max(max(y_true.flatten()), max(y_pred.flatten()))],
            dtick=50
        )
        fig.update_layout(
            title=f"| {title} | {index[slider+params.window_size].strftime(f'Day: %Y-%m-%d | Time prediction: {int(params.prediction_horizon*5/60)}h (%Hh%Mmin')} to {index[slider + params.window_size + params.prediction_horizon].strftime('%Hh%Mmin) |')} ",
            title_font=dict(size=28),
            legend=dict(title='Legends', font=dict(size=16))
        )
        return fig

    rmse_local = rmse(y_true[slider, :].flatten(), y_pred[slider, :].flatten())
    rmse_fed = rmse(y_true[slider, :].flatten(), y_pred_fed[slider, :].flatten())

    color_fed, color_local = get_color_fed_vs_local(rmse_fed, rmse_local, superior=False)

    render_confidence_interval = st.radio("Render confidence interval", [1, 0], format_func=(lambda x: "Yes" if x == 1 else "No"))

    # FEDERATED
    fed_fig = plot_graph(color_fed, 'Federated', "Federated Prediction", y_pred_fed, rmse_fed, slider)

    # LOCAL
    local_fig = plot_graph(color_local, 'Local', "Local Prediction", y_pred, rmse_local, slider)

    annotated_text(
        "A lower RMSE value indicates a better prediction. The ",
        ("green", "", "#75ff5b"), " prediction",
        " is better than the ",
        ("red", "", "#fe7597"), " one because it has a lower RMSE value")

    with st.spinner('Plotting...'):
        st.plotly_chart(fed_fig, use_container_width=True)
        st.plotly_chart(local_fig, use_container_width=True)


#######################################################################
# Main
#######################################################################
st.header("Predictions Graph")

path_experiment_selected = selection_of_experiment()
if (path_experiment_selected is not None):
    with open(f"{path_experiment_selected}/test.json") as f:
        results = json.load(f)
    with open(f"{path_experiment_selected}/config.json") as f:
        config = json.load(f)

    mapping_sensor_with_nodes = {}
    for node in results.keys():
        mapping_sensor_with_nodes[config["nodes_to_filter"][int(node)]] = node

    sensor_select = st.selectbox('Choose the sensor', mapping_sensor_with_nodes.keys())

    metrics = list(results[mapping_sensor_with_nodes[sensor_select]]["local_only"].keys())
    multiselect_metrics = st.multiselect('Choose your metric(s)', metrics, ["RMSE", "MAE", "SMAPE", "Superior Pred %"])

    local_node = []
    if "local_only" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        local_node = results[mapping_sensor_with_nodes[sensor_select]]["local_only"]
        local_node = pd.DataFrame(local_node, columns=multiselect_metrics, index=["sensor alone"])

    federated_node = []
    if "Federated" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        federated_node = results[mapping_sensor_with_nodes[sensor_select]]["Federated"]
        federated_node = pd.DataFrame(federated_node, columns=multiselect_metrics, index=["sensor in Federation"])

    st.subheader("sensor in Federation vs sensor alone")
    fed_local_node = pd.concat((federated_node, local_node), axis=0)
    st.table(fed_local_node.style.set_table_styles(style_dataframe(fed_local_node)).format("{:.2f}"))

    params = Params(f'{path_experiment_selected}/config.json')
    if (path.exists(f'{path_experiment_selected}/y_true_local_{mapping_sensor_with_nodes[sensor_select]}.npy') and
        path.exists(f"{path_experiment_selected}/y_pred_fed_{mapping_sensor_with_nodes[sensor_select]}.npy")):
        plot_prediction_graph(path_experiment_selected)
    else:
        st.write("ERROR")
