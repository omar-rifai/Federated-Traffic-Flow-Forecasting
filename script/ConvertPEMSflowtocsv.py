# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

flow_file= "./../data/PEMS04/PEMS04.npz"
csv_file = "./../data/PEMS04/distance.csv"
data = np.load(flow_file)
df = pd.read_csv(csv_file)
TS = data['data']
flow = TS[:,:,0]
# flow dict 100 time series is the sensor number and the value the traffic flow times serie
flow_dict={k:flow[:,k] for k in range(307)}
# list of the first 10 connected sensor, each sensor traffic flow is contained in PeMS 
PeMS = pd.DataFrame(flow_dict)
# time serie of sensor k
#creation of the datetime index
start_date = "2018-01-01 00:00:00"
end_date = "2018-02-28 23:55:00"
interval = "5min"
index = pd.date_range(start=start_date, end=end_date, freq=interval)
PeMS = PeMS.set_index(index)
PeMS.to_csv(".\..\data\PEMS04\PeMS04flow.csv")


