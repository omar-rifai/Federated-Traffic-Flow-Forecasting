def create_graph(graph_df):

    import networkx as nx
    """
    Create a graph from the PeMS distance.csv file (using the from/to adjacency information) 
    
    Parameters
    ----------
    graph_df: pd.DataFrame
        Dataframe with (to/from) points are sensors and cost corresponds to the distance between nodes 

    Return
    -------
    nx.Graph 
    """


    G = nx.Graph()

    # Iterate over each row in the DataFrame and add nodes and edges to the graph
    for i, row in graph_df.iterrows():
        # Add the "from" node to the graph if it doesn't already exist
        if not G.has_node(row["from"]):
            G.add_node(row["from"])
        # Add the "to" node to the graph if it doesn't already exist
        if not G.has_node(row["to"]):
            G.add_node(row["to"])
        # Add the edge between the "from" and "to" nodes with the cost as the edge weight
        G.add_edge(row["from"], row["to"], weight=row["cost"])
    return G 
    

def subgraph_dijkstra(G, node, n_neighbors):

    import networkx as nx

    """ 
    Computes a subgraph by selecting the n nearest neighbors

    G : NetworkX graph,
    node : int
        node number to start from
    n_neighbors : int
        Number of neighbors for subgraph

    Returns
    -------
    nx.Graph
        subgraph with n nearest neighbors
    
    """ 
    # Compute the shortest path from node 0 to all other nodes
    distances = nx.single_source_dijkstra_path_length(G, node)

    # Sort the nodes based on their distance from node 0
    nearest_nodes = sorted(distances, key=distances.get)[:n_neighbors+1]

    # Construct the subgraph by including only the selected nodes and their edges
    subgraph = G.subgraph(nearest_nodes)
    return subgraph


def calculate_laplacian_with_self_loop(matrix):
    
    import torch
    
    """ 
    Computes Laplacian matrix normalization.

    matrix : any,

    Returns
    -------
    matrix_laplacian
        matrix compute with Laplacian matrix normalization.
    
    """ 
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)