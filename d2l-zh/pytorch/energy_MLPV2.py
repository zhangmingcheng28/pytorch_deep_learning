# -*- coding: utf-8 -*-
"""
@Time    : 5/13/2022 2:18 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import cv2
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import time
import math
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ownMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ownMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64), #1, # 1st para is input size after flatten, output is the number of neurons in hidden layers after input layer
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),  # 2
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),  # 3
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),  # 4
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),  # 5
            nn.ReLU(inplace=True),
            nn.Linear(8, output_dim) # 6
        )

    def forward(self, x: torch.Tensor):
        x = self.classifier(x)
        return x


def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]


def get_dataset(x, y):
    return TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())


def get_dataloader(x: np.array, y: np.array, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    dataset = get_dataset(x, y)  # convert the numpy to torch standard TensorDataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)  # load and prepare the DataSet


def multivariate_data(dataset, target, start_index=0, end_index=None, history_size=20, target_size=-1, step=1, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def df_to_xyArray(inputDF, columnPos, max_colVal, min_colVla):  # convert the df to time series input array (X) and corrosponding output array (Y)
    columnIdx_name_ref = {}
    # only choose part of the dataframe
    features = inputDF[
        ['mass', 'roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate', 'v_n', 'v_e', 'v_d', 'accel_n', 'accel_e', 'accel_d', 'wind_n', 'wind_e', 'energy']]  # only exclude "flight"
    featuresArray = features.values  # extract the values in DF, in terms of 2D array, same size with DF
    # normalise
    featuresArray[:, columnPos] = (featuresArray[:, columnPos] - min_colVla) / (max_colVal - min_colVla)  # normalise every flight's required column's data
    x_input = featuresArray[:, :-1].astype(float)  # set the input of the flight, excluding the last column of power
    supposed_y_output = featuresArray[:, -1].astype(float)
    for columnIdex in range(0, featuresArray.shape[1]):
        columnIdx_name_ref[columnIdex] = features.columns[columnIdex]
    x, y = multivariate_data(x_input, supposed_y_output, single_step=True)
    # Transpose the x, so that features is the row, time-sequence/step is column
    x = np.transpose(x, (0, 2, 1))  # 0th dimension stay at the 1st place, the 1st and 2nd dimension chance place.
    return x, y, columnIdx_name_ref


def load_xyDict_to_dataloaderDict(inputDict):
    dataLoaderDict = {}
    batchSize = 32
    for flightIdx_key, xyArray in inputDict.items():
        dataLoaderDict[flightIdx_key] = get_dataloader(xyArray[0], xyArray[1], batchSize)
    return dataLoaderDict

# ======================
#     Data preparing
# ======================
# load data
df_prepared = pd.read_csv(r"F:\githubClone\TCN\m100_for_model.csv")
totalFlightNum = []
for flightidx in df_prepared.flight.unique():  # Note: the flight number is NOT continous, e.g. there is no flight no.9
    totalFlightNum.append(flightidx)

# ======================
#     Spilt flight data (fixed_altitude) into 80:20, training, test
# ======================
np.random.seed(42)  # ensure spilt the same every time it runs during debugging
train_flightIdx, test_flightIdx = np.split(np.array(totalFlightNum), [int(.8 * len(np.array(totalFlightNum)))])
# grab all rows belongs to the same flight.
train_list = [df_prepared.loc[df_prepared['flight'] == singleFlight, :] for singleFlight in totalFlightNum]
# all element in the list are DF, with same column names, just pick one to identify the index of the column that need to be normalised
ColumnToNormalise = column_index(train_list[0], ['mass', 'roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate', 'v_n', 'v_e', 'v_d', 'accel_n', 'accel_e', 'accel_d', 'wind_n', 'wind_e'])

dataset_train = np.concatenate([df_prepared.loc[df_prepared['flight'] == singleFlight, ['mass', 'roll', 'pitch', 'yaw', 'roll_rate', 'pitch_rate', 'yaw_rate', 'v_n', 'v_e', 'v_d', 'accel_n', 'accel_e', 'accel_d', 'wind_n', 'wind_e', 'energy']] for singleFlight in train_flightIdx])
data_min = dataset_train[:, ColumnToNormalise].min(axis=0)
data_max = dataset_train[:, ColumnToNormalise].max(axis=0)

#build a dict to store the data for each flight for both training and validation DF, and normalize it.
train_data_dict = {}
test_data_dict = {}
for eachFlight in train_list:
    if eachFlight.empty:
        continue
    if eachFlight['flight'].iloc[0] in train_flightIdx:
        x, y, idx_name_ref = df_to_xyArray(eachFlight, ColumnToNormalise, data_max, data_min)  # x:(dim1,dim2,dim3) total of 1320 data points, 20 data points form a timeseries, 19 features in total. y: (dim1), input of 1320 set of 20 data points, leads to a power value.
        train_data_dict[eachFlight['flight'].iloc[0]] = (x, y)
    elif eachFlight['flight'].iloc[0] in test_flightIdx:
        x, y, idx_name_ref = df_to_xyArray(eachFlight, ColumnToNormalise, data_max, data_min)
        test_data_dict[eachFlight['flight'].iloc[0]] = (x, y)
# load the data dictionary to dataloader and store as dictionary for every flight
dataLoader_training_Dict = load_xyDict_to_dataloaderDict(train_data_dict)
dataLoader_test_Dict = load_xyDict_to_dataloaderDict(test_data_dict)

# ======================
#     Configure and load MLP model
# ======================
model = ownMLP(15, 1)  # input is 15 channel, output is 1.
model.to(device)

# ======================
#     Training loop
# ======================
total_epochs = 50
criterion = nn.MSELoss()  # NOTE: MESLoss() is used for regression, while CrossEntroyLoss() is used for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
torch.manual_seed(42)


def train(ep):
    model.train()
    train_loss = 0
    np.random.seed(42)
    np.random.shuffle(train_flightIdx)  # now, the train_flightIdx is a numpy array that has been shuffled randomly
    for idx, flight in enumerate(train_flightIdx):  # train based on flight index
        # train(loop through) the portion of data in the data_loader that matches the flight index
        for batch_idx, (data_in, target) in enumerate(dataLoader_training_Dict[flight]):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            pred_out = model(data_in)  # data_in, is (N,C,L) format,N is batch size, C is number of features, L is the length of sequence (time-steps)
            # calculate loss
            loss = criterion(pred_out.squeeze(), target)
            # weight assignment
            loss.backward()
            # update model weights
            optimizer.step()
            train_loss += loss.item()
    return train_loss

# ======================
#     Evaluation loop
# ======================
def evaluate(X_data, name):
    model.eval()
    total_loss = 0.0
    Idx_list = None
    if name == "Validation":
        #Idx_list = validate_flightIdx
        Idx_list = None
        np.random.shuffle(Idx_list)
    elif name == "Test":
        Idx_list = test_flightIdx

    # record in single evaluation epoch, the "data_in", "target" and "pred_out" for each  flight
    evaRecord_singleFlight = {}
    with torch.no_grad():
        for idx, flight in enumerate(Idx_list):
            for batch_idx, (data_in, target) in enumerate(X_data[flight]):
                # compute the model output
                pred_out = model(data_in)  # data_in, is (N,C,L) format,N is batch size, C is number of features, L is the length of sequence (time-steps)
                # calculate loss
                loss = criterion(pred_out.squeeze(), target)
                total_loss += loss.item()
                # record the (flight, batch_idx) together with data_in, target, and pred_out
                evaRecord_singleFlight[(flight, batch_idx)] = [data_in, target, pred_out]
        return total_loss, evaRecord_singleFlight


training_test_result_with_model = {}
for epIdx in range(1, total_epochs+1):
    begin = time.time()  #
    train_loss = train(epIdx)
    print(epIdx, train_loss)
    tloss, evaRecord_singleFlight = evaluate(dataLoader_test_Dict, name='Test')
    training_test_result_with_model[epIdx] = [tloss, evaRecord_singleFlight]
    # torch.save(model.state_dict(), r'F:\githubClone\TCN\TCN\result_from_ownDataV2\epoch_' + str(epIdx)+'_model')
    # with open(r'F:\githubClone\TCN\TCN\result_from_ownDataV2\epoch_' + str(epIdx) + '_loss_and_evaluationResult.pickle', 'wb') as handle:
    #     pickle.dump(training_test_result_with_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # epoch_time_used = time.time()-begin
    print("epoch {} used {} second to finish".format(epIdx, epoch_time_used))
print("end of all epochs!")