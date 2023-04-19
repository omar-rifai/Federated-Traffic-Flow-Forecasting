
import src.models as models 
import utils_graph as gu
from pathlib import Path
import torch 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    
def expSmooth_data(df_PeMS ,alpha=0.2):
    
    """
    Exponential smoothing using the Holt Winters method
    
    Parameters:
    -----------
    df_PeMs : pd.DataFrame 
        data to smooth
    alpha : float
        exponential smoothing param

    Returns
    -------
    pd.Dataframe
        Dataframe with the input smoothed
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # Apply exponential smoothing to the time serie
    for i in range(len(df_PeMS.columns)):
        y = df_PeMS[df_PeMS.columns[i]]
        model = ExponentialSmoothing(y).fit(smoothing_level=alpha)
        smooth = model.fittedvalues
        df_PeMS[df_PeMS.columns[i]] = smooth
    return df_PeMS


def normalize_data(df_PeMS):
    """
    Normalize the data diving by the maximum
    
    Parameters:
    -----------
    df_PeMs : pd.DataFrame 
        data to smooth

    Returns
    -------
    pd.Dataframe
        Dataframe with the input normalized
    """
    maximum = df_PeMS.max().max()
    df_PeMS = df_PeMS /  maximum

    return df_PeMS

def createExperimentsData(cluster_size, df_PeMS, layers = 6, perc_train = 0.7, perc_val = 0.15, subgraph = False, overwrite = False):
    import pickle 

    """
    Generates pickled (.pkl) dictionary files with the train/val/test data and an associated model

    Parameters
    ----------
    cluster_size : int
        Size of the node clusters

    df_PeMs : pd.Dataframe
        dataframe with all the PeMS data 

    layers: int
        number of layers for the NN model

    perc_train : float
        percentage of the data to be used for training

    perc_test : float
        percentage of the data to be used for testing

    """
    
    train_len = len(df_PeMS)

    if subgraph:
        dirpath = './experiment/cluster'
        subgraph = gu.subgraph_dijkstra(G,i, cluster_size-1)
        nodes_range = range(df_PeMS.columns)
        columns = list(subgraph.nodes)
    else:
        dirpath = './experiments/clusterSubGraph'
        nodes_range = range(len(df_PeMS.columns)+1-cluster_size)
        columns = df_PeMS.columns[i:i+cluster_size]

    filename = Path(dirpath) / f"S{cluster_size}l{train_len}"
    
    if (filename.isfile()):

        
    
        cluster_dict={"size":cluster_size}

        for i in nodes_range:
            model = models.LSTMModel(input_size=cluster_size, hidden_size=32, num_layers=layers, output_size=cluster_size)
            train_loader, val_loader, test_loader = createLoaders(df_PeMS, columns,  perc_train, perc_val)
            cluster_dict[i]={"model":model,"train":train_loader,"val":val_loader,"test":test_loader}

        with open(filename, 'wb') as f:
            pickle.dump(cluster_dict, f)

    return model, train_loader, val_loader, test_loader


    with open('./experiment/clusterS{}.pkl'.format(i), 'rb') as f:
        my_dict = pickle.load(f)
        # iterate on number of cluster 100-i+1
        for j in range(100-i+1):
            train = my_dict[j]["train"]
            val = my_dict[j]["val"]
            model = my_dict[j]["model"]
            model = train_model(model,train, val)
            my_dict[j]["model"]=copy.deepcopy(model)
    with open('./experiment/clusterS{}.pkl'.format(i), 'wb') as f:
        pickle.dump(my_dict, f)
    print('Experiment" + {} +" COMPLETED !'.format(i))




def createLoaders(df_PeMS, columns, perc_train = 0.7, perc_val = 0.15):
    """
    Returns torch.DataLoader for train validation and test data
    """
    from torch.utils.data import  DataLoader

    train_len = len(df_PeMS)

    train_data= df_PeMS[columns][:int(train_len * perc_train)]
    val_data =  df_PeMS[columns][:int(train_len * perc_train): int(train_len * (perc_train + perc_val))]
    test_data = df_PeMS[columns][int(train_len * (perc_train + perc_val)):]
    
    train_loader = DataLoader(train_data)
    val_loader = DataLoader(val_data)
    test_loader = DataLoader(test_data)

    return train_loader, val_loader, test_loader 



def load_PeMS04_data(input_path: Path = "./data/PEMS04/"):
    import pandas as pd
    import numpy as np
    """
    Function to load data from 'npz' and 'csv' files associated with PeMS

    Parameters
    ----------
    input_path: Path
        Path to the input directory

    Returns
    -------
    df_PeMS : pd.Dataframe
        With the flow between two sensors
    df_disntace:
        Dataframe with the distance metrics between sensors
    """


    flow_file = input_path / 'pems04.npz'
    csv_file  = input_path / 'distance.csv'

    # the flow data is stored in 'data' third dimension
    df_flow = np.load(flow_file)['data'][:,:,0]
    df_distance = pd.read_csv(csv_file)
    
    dict_flow = { k : df_flow[:,k] for k in range(len(df_flow))}

    df_PeMS = pd.DataFrame(dict_flow)


    start_date = "2018-01-01 00:00:00"
    end_date = "2018-02-28 23:55:00"
    interval = "5min"
    index = pd.date_range(start=start_date, end=end_date, freq=interval)
    df_PeMS = df_PeMS.set_index(index)

    return df_PeMS, df_distance



def preprocess_PeMS_data(df_PeMS, df_distance, init_node : int = 0, n_neighbors : int = 100):
    from utils_graph import create_graph, subgraph_dijkstra
    """
    Filter to n nearest neightbors from 'init_node', sort by mean traffic flow, and normalize and smooth data

    Parameters
    ----------
    init_node : int
        Index of the node we want to start with

    n_neighbors: int
        Number of nearest neighbors to consider
    """

    # Filter nodes to retain only n nearest neighbors
    graph_init = create_graph(df_distance)
    graph_nearest = subgraph_dijkstra(graph_init, init_node, n_neighbors)
    df_PeMS = df_PeMS[list(graph_nearest.nodes)]

    #Sort data hby mean traffic flow

    df_sorted= df_PeMS.mean().sort_values()
    index_mean_flow = df_sorted.index
    column_order = list(index_mean_flow)
    df_PeMS = df_PeMS.reindex(columns = column_order)

    df_PeMS = expSmooth_data(df_PeMS)
    df_PeMS = normalize_data(df_PeMS)

    return df_PeMS
