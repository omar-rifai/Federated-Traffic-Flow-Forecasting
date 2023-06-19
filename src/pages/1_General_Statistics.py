###############################################################################
# Libraries
###############################################################################
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils_streamlit_app import selection_of_experiment

st.set_page_config(layout="wide")

#######################################################################
# Main
#######################################################################
st.header("General Statistics")

path_experiment_selected = selection_of_experiment()
if (path_experiment_selected is not None):
    with open(f"{path_experiment_selected}/test.json") as f:
        results = json.load(f)
    with open(f"{path_experiment_selected}/config.json") as f:
        config = json.load(f)

    nodes = results.keys()
    mapping_sensor_and_node = {}
    for node in nodes:
        mapping_sensor_and_node[config["nodes_to_filter"][int(node)]] = node

    df_local_node = []
    df_federated_node = []
    for sensor in mapping_sensor_and_node.keys():
        if "Federated" in results[mapping_sensor_and_node[sensor]].keys():
            federated_node = results[mapping_sensor_and_node[sensor]]["Federated"]
            df_federated_node.append(federated_node)
        if "local_only" in results[mapping_sensor_and_node[sensor]].keys():
            local_node = results[mapping_sensor_and_node[sensor]]["local_only"]
            df_local_node.append(local_node)

    metrics = list(results['0']["local_only"].keys())
    multiselect_metrics = st.multiselect('Choose your metric(s)', metrics, ["RMSE", "MAE", "SMAPE", "Superior Pred %"])

    if df_federated_node != []:
        st.subheader("How much the Federated version is performant on average taking in account all the sensors for the calculation of the statistics")
        df_federated_node = pd.DataFrame(df_federated_node, columns=multiselect_metrics)
        global_stats_fed_ver = df_federated_node.describe().T
        global_stats_fed_ver.rename(columns={'count': 'Nb sensors'}, inplace=True)
        global_stats_fed_ver = global_stats_fed_ver.applymap(lambda x: '{:.2f}'.format(x))
        st.table(global_stats_fed_ver.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}]))

        c1_boxplot_fed, c2_bobxplot_fed, c3_boxplot_fed = st.columns((1, 2, 1))
        fig, ax = plt.subplots()
        bar_plot_results = df_federated_node
        bar_plot_results.boxplot(column=["RMSE", "MAE"], ylabel="values", xlabel="sensor", ax=ax)
        plt.yticks(np.arange(0, max(bar_plot_results["RMSE"].max(), bar_plot_results["MAE"].max()), 10))
        with c2_bobxplot_fed:
            st.pyplot(fig, use_container_width=True)

    if df_local_node != []:
        st.subheader("How much the Local version is performant on average taking in account all the sensors for the calculation of the statistics")
        df_local_node = pd.DataFrame(df_local_node, columns=multiselect_metrics)
        global_stats_local_ver = df_local_node.describe().T
        global_stats_local_ver = global_stats_local_ver.apply(lambda k: k.round(0))
        global_stats_local_ver.rename(columns={'count': 'Nb sensors'}, inplace=True)
        global_stats_local_ver = global_stats_local_ver.applymap(lambda x: '{:.2f}'.format(x))
        st.table(global_stats_local_ver.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}]))

        c1_boxplot_local, c2_bobxplot_local, c3_boxplot_local = st.columns((1, 2, 1))
        fig, ax = plt.subplots()
        bar_plot_results = df_local_node
        bar_plot_results.boxplot(column=["RMSE", "MAE"], ylabel="values", xlabel="sensor", ax=ax)
        plt.yticks(np.arange(0, max(bar_plot_results["RMSE"].max(), bar_plot_results["MAE"].max()), 10))
        with c2_bobxplot_local:
            st.pyplot(fig, use_container_width=True)
