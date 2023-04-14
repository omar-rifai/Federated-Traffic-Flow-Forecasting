from torch.utils.data import Dataset, DataLoader
import src.models as models 
import utils_graph as gu

import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset model with input/target pairs for the LSTM model
    Defines the sliding window size and stride
    """
    
    def __init__(self, data, window_size=7, stride=1):
        self.data = data
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        inputs = self.data[idx : idx+self.window_size]
        target = self.data[idx+self.window_size]
        return inputs, target


def DataLoaderTimeSeries(data, window_size = 7, stride = 1):
    """
    Returns
    -------
    DataLoader
    """
    dataset = TimeSeriesDataset(data.values, window_size, stride)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    loader = [(inputs.to(device), targets.to(device)) for inputs, targets in loader]
    return loader
    

def expSmooth(df_PeMS ,alpha=0.2):
    
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

    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # Apply exponential smoothing to the time serie
    for i in range(len(df_PeMS.columns)):
        y = df_PeMS[df_PeMS.columns[i]]
        model = ExponentialSmoothing(y).fit(smoothing_level=alpha)
        smooth = model.fittedvalues
        df_PeMS[df_PeMS.columns[i]] = smooth
    return df_PeMS


def createExperimentsData(cluster_size, df_PeMS, layers = 6, perc_train = 0.7, perc_val = 0.15):
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
        filename = './experiment/cluster'
        subgraph = gu.subgraph_dijkstra(G,i, cluster_size-1)
        nodes_range = range(df_PeMS.columns)
        columns = list(subgraph.nodes)
    else:
        filename = './experiments/clusterSubGraph'
        nodes_range = range(len(df_PeMS.columns)+1-cluster_size)
        columns = df_PeMS.columns[i:i+cluster_size]

    cluster_dict={"size":cluster_size}
    for i in nodes_range:

        model = models.LSTMModel(input_size=cluster_size, hidden_size=32, num_layers=layers, output_size=cluster_size)

        train_loader, val_loader, test_loader = createLoaders(df_PeMS, columns,  perc_train, perc_val)

        cluster_dict[i]={"model":model,"train":train_loader,"val":val_loader,"test":test_loader}

    with open(filename + 'S{}l{}.pkl'.format(cluster_size , train_len), 'wb') as f:
        pickle.dump(cluster_dict, f)



def createLoaders(df_PeMS, columns, perc_train = 0.7, perc_val = 0.15):
    """
    Returns torch.DataLoader for train validation and test data
    """

    train_len = len(df_PeMS)

    train_data= df_PeMS[columns][:int(train_len * perc_train)]
    val_data =  df_PeMS[columns][:int(train_len * perc_train): int(train_len * (perc_train + perc_val))]
    test_data = df_PeMS[columns][int(train_len * (perc_train + perc_val)):]
    
    train_loader = DataLoaderTimeSeries(train_data)
    val_loader = DataLoaderTimeSeries(val_data)
    test_loader = DataLoaderTimeSeries(test_data)

    return train_loader, val_loader, test_loader 

