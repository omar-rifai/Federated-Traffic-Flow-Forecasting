
import matplotlib.pyplot as plt

from utils_graph import create_graph, subgraph_dijkstra 
from utils_data import load_PeMS04_flow_data, preprocess_PeMS_data, plot_prediction
from models import LSTMModel, testmodel
from fedutil import local_dataset, fed_training_plan


from metrics import calculate_metrics 

#Load traffic flow dataframe and graph dataframe from PEMS
PeMS, distance = load_PeMS04_flow_data()


PeMS[118].max()

PeMS2, adjmat, meanstd_dict = preprocess_PeMS_data(PeMS,distance,0,99,True,True,False,False)

meanp= PeMS[118].mean()
stdp= PeMS[118].std()

G = create_graph(distance)
subgraph = subgraph_dijkstra(G,0,99)
PeMS = PeMS[list(subgraph.nodes)]

goodnodes = [118,168,261]
for i in goodnodes:
    print("Nodes {} with mean traffic flow : {}".format(i,meanstd_dict[i]['mean']))
    print("Nodes {} with standard deviation : {}".format(i,meanstd_dict[i]['std']))

import matplotlib.pyplot as plt



plt.figure(figsize = (40,9))
plt.plot(PeMS[118])
plt.plot(PeMS[168])
plt.plot(PeMS[261])
plt.title('Our Sensor Traffic Flow')
plt.show()

# Federated Learning Experiment
datadict = local_dataset(PeMS,3)
main_model = LSTMModel(input_size=1,hidden_size=32,output_size=1, num_layers=6)

PeMS[118].min()

fed_training_plan(datadict, rounds=50, epoch=50)

# Training Local
from models import LSTMModel, train_model
train_losses = {}
val_losses = {}
for j in range(1):
    data_dict = datadict[j]
    new_model, train_losses[j], val_losses[j] = train_model(main_model, data_dict['train'], data_dict['val'], model_path ='./dummy{}.pth'.format(j),num_epochs=200, remove = False)

plt.plot( val_losses[0],label='validation')
plt.plot(train_losses[0],label= 'train')
plt.legend()
plt.show()

y_true, y_pred = testmodel(new_model,data_dict['test'], None, meanstd_dict,sensor_order_list=[118])

y_true, y_pred = testmodel(new_model,data_dict['test'],'local0.pth', meanstd_dict,sensor_order_list=[118])

plot_prediction(y_true,y_pred)

calculate_metrics(y_true,y_pred,1)