import torch
from torch.utils import DataLoader

from pathlib import Path

from utils_data import load_PeMS04_data, preprocess_PeMS_data, createExperimentsData
from models import TimeSeriesDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the sliding window size and stride
window_size = 7
stride = 1
layers = 6
lengths_cluster = [1,5,10,15]
experiments_path = Path('./experiment/')

df_PeMS, df_distance  = load_PeMS04_data()
df_PeMS = preprocess_PeMS_data(df_PeMS, df_distance)

dataset = TimeSeriesDataset(df_PeMS.values, window_size, stride)

loader = DataLoader(dataset, batch_size=32, shuffle=False)
loader = [(inputs.to(device), targets.to(device)) for inputs, targets in loader]


#initialize the experiment datasets as pickle object




for cluster_size in lengths_cluster:
    
    
    createExperimentsData(cluster_size, df_PeMS, layers = 6, perc_train = 0.7, perc_val = 0.15, overwrite = False)
   
   loadExperimentsData(filepath)

def 
# load the experiment datasets from pickle object 

# # Exp using graph

#initialize the experiment datasets as pickle object
for i in [1,5,10,15]:
   experiment_dataset_subgraph(i,PeMS)
# iterate on cluster size i
for i in [1,5,10,15]:
# load the experiment datasets from pickle object 
    with open('./experiment/clusterGsize{}.pkl'.format(i), 'rb') as f:
        my_dict = pickle.load(f)
        # iterate on number of cluster 100-i+1
        for j in range(100-i+1):
            train = my_dict[j]["train"]
            val = my_dict[j]["val"]
            model = my_dict[j]["model"]
            model = train_model(model,train, val)
            my_dict[j]["model"]=copy.deepcopy(model)
    with open('/experiment/clusterGsize{}.pkl'.format(i), 'wb') as f:
        pickle.dump(my_dict, f)


