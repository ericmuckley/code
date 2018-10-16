# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:56:18 2018
@author: eric muckley
"""
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
import pandas as pd
import csv
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from timeit import default_timer as timer
from itertools import groupby
import matplotlib.pyplot as plt
#plt.rcParams.update({'figure.autolayout': True})
label_size = 18 #make size of axis tick labels larger
plt.rcParams['xtick.labelsize'] = label_size 
plt.rcParams['ytick.labelsize'] = label_size


#%% define functions

# seq_to_matrix function takes a 2D LabVIEW pressure sequence (times, pressures)
# and converts to a 2D matrix used as inputs for ANN testing with var_num number
# of variables (columns) and number of points designated by points_per_minute

def seq_to_mat(seq0, points_per_minute=10, var_num=4):
    total_seq_points = int(np.sum(seq0[:,0]))*int(points_per_minute)
    #build expanded pressures array
    pressure = np.full(points_per_minute*seq0[0,0],[seq0[0,1]])
    for i in range(1, len(seq0)):
        pressure = np.append(pressure, np.full(points_per_minute*seq0[i,0],[seq0[i,1]]))
    #find indices of setpoint changes
    ei = np.array([i+1 for i in range(len(pressure)-1) if pressure[i] != pressure[i+1]])
    #find values of each setpoint
    setpoint_vals = [k for k,g in groupby(pressure)]
    #find changes in each setpoint
    setpoint_diff_vals = np.insert(np.ediff1d(setpoint_vals),0,0)
    #find length of each setpoint and add initial and final setpoint lengths
    step_len = np.append(np.insert(np.ediff1d(ei),0,ei[0]), len(pressure) - ei[-1])
    #create array of setpoint differences
    setpoint_diff = np.array([])
    rel_time= np.array([])
    for i in range(len(step_len)):
        setpoint_diff = np.append(setpoint_diff, np.repeat(setpoint_diff_vals[i], step_len[i]))
        rel_time= np.append(rel_time, np.arange(step_len[i]))
    rel_time = rel_time / points_per_minute
    #create matrix and populate it
    expanded_seq = np.zeros((total_seq_points, int(var_num)))
    expanded_seq[:,0] = np.arange(total_seq_points)/points_per_minute
    expanded_seq[:,1] = rel_time
    expanded_seq[:,2] = pressure
    expanded_seq[:,3] = setpoint_diff
    return expanded_seq

########################################################################
# find setpoint changes in the array of setpoints, which allows use
# of "setpoint difference" and "relative time" inputs for NN model

def find_setpoint_changes(setpoint0):
    #find indices of setpoint changes
    ei = np.array([i for i in range(len(setpoint0)-1) if setpoint0[i] != setpoint0[i+1]])
    #find values of each setpoint
    setpoint_vals = [k for k,g in groupby(raw_data['setpoint'])]
    #find changes in each setpoint
    setpoint_diff_vals = np.insert(np.ediff1d(setpoint_vals),0,0)
    #find length of each setpoint and add initial and final setpoint lengths
    step_len = np.append(np.insert(np.ediff1d(ei),0,ei[0]), len(setpoint0) - ei[-1])

    #create array of setpoint differences
    setpoint_diff = np.array([])
    rel_time= np.array([])
    for i in range(len(step_len)):
        setpoint_diff = np.append(setpoint_diff, np.repeat(setpoint_diff_vals[i], step_len[i]))
        rel_time= np.append(rel_time, np.arange(step_len[i]))
    return rel_time, setpoint_diff #returns arrays of relative times and setpoint changes

#########################################################################
# look_back function adds "lag" numer of "lookback" points to 
# simulate the input training matrix of an RNN
def look_back(data, lag=1):
    var_num = len(data[0])
    #make zero array with correct size
    data_w_lag = np.zeros((len(data)-lag, var_num+var_num*lag))
    #populate zero array with appropriate lab matrices
    for L in range(lag+1):
        data_i = data[lag-L:len(data)-L,:]
        data_w_lag[:,L*var_num:var_num+L*var_num:] = data_i
    return data_w_lag


#%% import data
raw_import = pd.read_table('mxene_random_steps.txt', sep='\t')

raw_data = raw_import.copy()
#add relative times and setpoint changes as input variables
raw_data['rel_time'], raw_data['setpoint_diff'] = find_setpoint_changes(raw_import['setpoint'])

#fix offset in setpoint column
setpoint_fix = np.append(np.array(raw_import['setpoint'])[1:],
                         np.array(raw_import['setpoint'])[-1])
raw_data['setpoint'] = setpoint_fix



#%% simulate expected signal using pressure sequence data

tau = 18

sim_sig = 12 - 0.65*(raw_data['setpoint'] - 
           raw_data['setpoint_diff']*np.exp(-(raw_data['rel_time'])/tau))

sim_sig_df = raw_data.copy()
sim_sig_df['sim_sig'] = sim_sig

raw_data['sim_signal'] = sim_sig




#%% add simulated signal and reorder dataframe
raw_data = raw_data[['time', 'setpoint', 'rel_time', 'setpoint_diff', 'sim_signal']]
#raw_data = raw_data.drop(['time'], axis=1)




#%% set how many points to "lag" to simulate RNN input    
lag = 2

#%% format data for use as training data
time = np.array(raw_import['time'])[lag:]
sig_raw = np.array(raw_import['delta_f'])

sig_offset_num = 1
#offset signal so response is a couple points after pressure change
sig_offset = np.append(sig_raw[sig_offset_num:], sig_raw[-sig_offset_num:])

sig = sig_offset[lag:]
pressure = np.array(raw_import['setpoint'])[lag:]

#add look_back columns using amount of "lag"
raw_data_lagged = look_back(raw_data.values, lag)
#prepare target signal to re-add to lagged matrix
re_add_sig = np.reshape(sig, (-1, 1))
#add target signal to end of input training matrix
measured_data = np.concatenate((raw_data_lagged, re_add_sig), axis=1)
input_mat = measured_data
#set which column in dataset is the target column
tar_col = len(measured_data[0])-1

#%% sizes of train and test sets
percent_to_train = 80

train_size = int(len(input_mat) * percent_to_train/100) 
test_size= len(input_mat) - train_size

#%% normalization and splitting into train/test sets
scaler = MinMaxScaler(feature_range=(0, 1))
# prefix "_s" = scaled 
training_mat_s = scaler.fit_transform(input_mat)
train_inp_s =  training_mat_s[:train_size, :-1]
train_tar_s = training_mat_s[:train_size, tar_col].reshape((train_size, 1))
test_inp_s = training_mat_s[train_size:, :-1]
test_tar_s = training_mat_s[train_size:, tar_col]

#%% create the model
train_start_time = timer()
model = Sequential([#Dropout(.05, input_shape=(train_inp_s.shape[1:])),
                    Dense(12, activation='relu', input_shape=(train_inp_s.shape[1:])),
                    #Dropout(0.1, seed=1),
                    Dense(12, activation='relu'),
                    Dropout(.2, seed=1),
                  Dense(1)])
model.compile(loss='mean_squared_error', optimizer='adam')

#%% train model

epochs = 1500

history= model.fit(train_inp_s, train_tar_s,
                   validation_split=0.15,
                   epochs=epochs,
                   batch_size=500,
                   verbose=1)
print('training time = %.1f sec (%.2f min)' %(timer()-train_start_time,
                                 (timer()-train_start_time)/60))

#%% plot loss during training if running locally

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.semilogy(loss, c='k') # Plot training loss
plt.semilogy(val_loss, c='r')
plt.xlabel('Epoch', fontsize=label_size)
plt.ylabel('Loss', fontsize=label_size)
plt.legend(['training', 'validation'], fontsize=label_size)
plt.show()

#%% make predictions
trained_model_s = model.predict(train_inp_s), 
future_predictions_s = model.predict(test_inp_s)

#%% get sparse matrices which have same shape as scaled 
# training/testing data so we can unscale ANN results
# so we can unscale using the same scaler as before
trained_model_mat_s = np.zeros((train_size,len(input_mat[0])))
future_predictions_mat_s = np.zeros((test_size,len(input_mat[0])))
test_tar_mat_s = np.copy(future_predictions_mat_s)

# insert our data into its corresponding column of each sparse matrix
trained_model_mat_s[:,tar_col] = np.ndarray.flatten(np.array(trained_model_s))
future_predictions_mat_s[:,tar_col] = np.ndarray.flatten(future_predictions_s)


#%%# unscale our data
trained_model = scaler.inverse_transform(trained_model_mat_s)[:,tar_col]
future_pred = scaler.inverse_transform(future_predictions_mat_s)[:,tar_col]
train_tar = sig[:train_size]

plt.plot(time[train_size:],future_pred, c='r', label='prediction')
plt.scatter(time, sig, c='k', s=3, alpha=0.1, label='measured')
plt.plot(time[:train_size], trained_model, c='b', alpha=0.5, label='model')
plt.xlabel('Time (min)', fontsize=label_size)
plt.ylabel('Amplitude', fontsize=label_size)
plt.legend()
plt.show()

#%% make error calculations
tot_model = np.append(trained_model, future_pred)
error_raw = np.subtract(sig, tot_model)
percent_error = 100*np.abs(error_raw)/np.max(np.abs(sig))

fig, ax1 = plt.subplots()
ax1.plot(time, pressure, linewidth=0.5, c='b')
ax1.set_xlabel('Time (min)', fontsize=label_size)
ax1.set_ylabel('Pressure', color='b', fontsize=label_size)

ax2 = ax1.twinx()
ax2.plot(time, percent_error, linewidth=0.5, c='r')
ax2.set_ylabel('% error', color='r', fontsize=label_size)
ax2.tick_params('y', colors='r')
plt.show()

print('avg. percent error = %.2f' %(np.mean(percent_error)))



#%% plot simulated signal and real signal

plt.plot(time, sim_sig[lag:], label='simulated')
plt.plot(time, sig, label='actual')
plt.legend(fontsize=label_size)
plt.show()





#%% save and export model data
# populate model and prediction arrays so they are all correct length
save_model = np.zeros(len(time))
save_model[:train_size] = trained_model
save_pred = np.zeros(len(time))
save_pred[train_size:] = future_pred

model_output_headers = ['time', 'setpoint', 'signal',
                        'model', 'prediction', 'percent_error']
model_output = np.array([time, pressure, sig, save_model,
                         save_pred, percent_error]).T

with open('temp_model_output.csv','w') as modeloutputfile:
    writer = csv.writer(modeloutputfile, lineterminator='\n')
    writer.writerow(model_output_headers) #write headers
    for row in model_output:
        writer.writerow(row)

#%% save loss data and export
loss_headers = ['epoch', 'loss', 'validation_loss']
loss_save = np.array([np.arange(epochs)+1, loss, val_loss]).T

with open('temp_save_loss.csv','w') as lossfile:
    writer = csv.writer(lossfile, lineterminator='\n')
    writer.writerow(loss_headers) #write headers
    for row in loss_save:
        writer.writerow(row)