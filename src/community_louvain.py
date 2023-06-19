# Create and generate multiple community using louvain algorithm for cluster in federated learning. 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import community.community_louvain as community
import json

# Set random seed
np.random.seed(42)

# File paths
flow_file = "./data/PEMS04/PEMS04.npz"
csv_file = "./data/PEMS04/distance.csv"

# Load data
data = np.load(flow_file)
df = pd.read_csv(csv_file)

# Create graph
plt.figure(figsize=(13, 5))
G = nx.Graph()

# Iterate over each row in the DataFrame and add nodes and edges to the graph
for i, row in df.iterrows():
    # Add the "from" node to the graph if it doesn't already exist
    if not G.has_node(row["from"]):
        G.add_node(row["from"])
    # Add the "to" node to the graph if it doesn't already exist
    if not G.has_node(row["to"]):
        G.add_node(row["to"])
    # Add the edge between the "from" and "to" nodes with the cost as the edge weight
    G.add_edge(row["from"], row["to"], weight=row["cost"])

# Set the node positions using the spring layout algorithm
pos = nx.spring_layout(G, seed=42)

# Apply the Louvain algorithm to detect communities
partition = community.best_partition(G)

# Create subgraph and community dictionary for each community
subgraphs = {}
communities = {}
for node, community_id in partition.items():
    if community_id not in subgraphs:
        subgraphs[community_id] = nx.Graph()
    subgraphs[community_id].add_node(node)
    if community_id not in communities:
        communities[community_id] = [node]
    else:
        communities[community_id].append(node)

# Add edges to subgraphs for each community
for edge in G.edges():
    node1, node2 = edge
    community1 = partition[node1]
    community2 = partition[node2]
    if community1 == community2:
        subgraphs[community1].add_edge(node1, node2)
    else:
        for community_id in subgraphs:
            if node1 in subgraphs[community_id] and node2 in subgraphs[community_id]:
                subgraphs[community_id].add_edge(node1, node2)

# Print subgraph and community dictionary for each community
for community_id, subgraph in subgraphs.items():
    print("Community", community_id, ":", communities[community_id])
    print("Subgraph:", subgraph.edges())

# Output a dictionary with each community's nodes and edges
community_dict = {community_id: subgraphs[community_id].edges() for community_id in subgraphs}
print("Community dictionary:", community_dict)

# JSON configuration template
jsondict = {
    "time_serie_percentage_length": 0.25,
    "batch_size": 8000,
    "init_node": 0,
    "n_neighbours": 306,
    "smooth": True,
    "center_and_reduce": True,
    "normalize": False,
    "sort_by_mean": False,
    "nodes_to_filter": [],
    "number_of_nodes": 0,
    "window_size": 168,
    "prediction_horizon": 24,
    "stride": 1,
    "communication_rounds": 200,
    "num_epochs_local_no_federation": 200,
    "num_epochs_local_federation": 10,
    "epoch_local_retrain_after_federation": 200,
    "learning_rate": 0.001,
    "model": "LSTMModel",
    "save_model_path": "LSTMp50h48r200e10n261d5"
}

# Helper function to convert to integer
f = lambda x: int(x)

# Generate JSON files for each community
for i in range(27):
    jsondict["nodes_to_filter"] = [f(x) for x in communities[i]]
    jsondict["number_of_nodes"] = len(jsondict["nodes_to_filter"])
    jsondict["save_model_path"] = f"/community{i}/"
    with open(f"./experiments/community{i}.json", 'w') as file:
        json.dump(jsondict, file, indent=4, separators=(',', ': '))
