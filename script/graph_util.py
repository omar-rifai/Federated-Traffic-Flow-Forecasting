# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx

# %%
def create_graph(df):
    """
    create_graph(df)
    Create a graph from the PeMS distance.csv (using the from to data) 
    extracted dataframe were each sensor is a node where the distance between 
    each nodes correspond to the edges cost   
    """
    # Create a new NetworkX graph object
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
    return G 

# %%
def subgraph_dijkstra(G,node, n_neighbors):
    """ 
    subgraph_dijkstra(G , node , n_neighbors)
    G : NetworkX graph, node : node number to see neighbors, n_neighbors : Number of neighbors for subgraph
    Construct a subgraph of G using the n-nearest neighbors from the selected nodes of the graph. 
    """ 
    # Compute the shortest path from node 0 to all other nodes
    distances = nx.single_source_dijkstra_path_length(G, node)

    # Sort the nodes based on their distance from node 0
    nearest_nodes = sorted(distances, key=distances.get)[:n_neighbors+1]

    # Construct the subgraph by including only the selected nodes and their edges
    subgraph = G.subgraph(nearest_nodes)
    return subgraph
    

# %%



