# -*- coding: utf-8 -*-

import csv, glob, os, numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
labelsize = 18 #make size of axis tick labels larger
plt.rcParams['xtick.labelsize'] = labelsize 
plt.rcParams['ytick.labelsize'] = labelsize

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer

#%%
def sark_time_series(folder, col='Rs'):
    '''Reads in xlxs files produced from SARK-110 impedance
    analyzer and builds a Pandas dataframe out of them, with frequency
    as first column. This function reads every file inside the 
    'folder' variable.
    '''
    folder.sort(key=os.path.getmtime)
    #get frequencies
    freq = pd.read_csv(folder[0], skiprows=1)['Freq']
    #set up matrix to populate
    series = np.zeros((len(freq), len(folder)+1))
    series[:,0] = freq
    for i in range(len(folder)):
        print('spectrum '+format(i+1)+' / '+format(len(folder)))
        data0 = pd.read_csv(folder[i], skiprows=3)
        #populate columns of matrix
        series[:,i+1] = np.array(data0[col])
    return series



#%% calculate percent error from residuals
    
def percent_error(measured, fit):
    '''
    Calculates percent error between measured and fitted signals.
    Percent differences is calculated from the ratio of residual to
    entire range of measured values.
    '''
    measured_range = np.abs(np.amax(measured) - np.amin(measured))
    residual = np.subtract(np.abs(measured), np.abs(fit))
    percent_error = 100*np.abs(np.divide(residual, measured_range))
    
    return percent_error
    

#%% find all impedance files in the designated folder and sort by time/date
folder = glob.glob('C:\\Users\\a6q\\2018-05-01pedotpss_labeled/*')
folder.sort(key=os.path.getmtime)
print('found ' + format(len(folder)) + ' impedance files') 

#create matrix of spectra for examining or plotting in origin
#qcm_series = sark_time_series(folder, col='Rs')

#start end end indices for removing edge points from data files 
index1, index2 = 5, -1
skip_nth = 20

#%% set size of each data file
data_example = pd.read_csv(folder[0], skiprows=1)
data_example = data_example.iloc[index1:index2:skip_nth,:]
freq = data_example.values[:, 0]#*1e3 #frequencies in kHz

#build data matrix
'''
pressure_list = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 52.5, 55, 57.5, 60,
                 62.5, 65, 67.5, 70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5, 
                 90, 92.5, 95]
'''
#get pressure from filenames
pressure_list = [file.split('\\')[-1].split('.')[0] for file in folder]

#%% organize spectra into matrix for ANN training

#if data is already loaded, do not load again to save time
try:
  spectra_df
except NameError:
    xs_spectra = np.zeros((len(freq), len(folder)+1))
    rs_spectra = np.zeros((len(freq), len(folder)+1))
    rs_spectra[:,0] = freq
    xs_spectra[:,0] = freq
    
    freq_col = np.array([]); pressure_col = np.array([])
    rs_col = np.array([]); xs_col = np.array([])
    
    
    #build data matrix
    for i in range(len(folder)):
        print('spectrum '+format(i+1)+' / '+format(len(folder)))
        data0 = pd.read_csv(folder[i], skiprows=1)
        data0 = data0.iloc[index1:index2:skip_nth,:]
    
        rs0 = np.array(data0['Rs'])
        xs0 = np.array(data0['Xs'])
        
        
        #remove outlier points
        
        
        
        
        rs_spectra[:,i+1] = rs0
        xs_spectra[:,i+1] = xs0
    
        freq_col = np.append(freq_col, freq)
        rs_col = np.append(rs_col, rs0)
        xs_col = np.append(xs_col, xs0)
        pressure_col = np.append(pressure_col,
                                 np.full(len(xs0), pressure_list[i]))
    
    #construct dataframe with results
    spectra_df = pd.DataFrame(np.array([freq_col, pressure_col,
                                        rs_col, xs_col]).T, 
                          columns=['freq', 'pressure', 'rs','xs'],
                          dtype=np.float64)
else:
  pass





#%% normalization and splitting into train/test sets
#number of features to predict
predict_features = 2 
#ratio of train to test samples
tr = 2

# prefix "_s" = scaled
scaler_inp = StandardScaler().fit(spectra_df[['freq', 'pressure']].values)
#select freq and pressure and rows where pressure is even
train_inp = spectra_df[['freq', 'pressure']].loc[spectra_df['pressure'] %tr!=0]
train_inp_s = scaler_inp.transform(train_inp.values)
#select freq and pressure and rows where pressure is odd
test_inp = spectra_df[['freq', 'pressure']].loc[spectra_df['pressure'] %tr==0]
test_inp_s = scaler_inp.transform(test_inp.values)

if predict_features == 1: #-------------------------------------------------
    scaler_tar = StandardScaler().fit(spectra_df[['rs']].values)
    #select target column and rows where pressure is even
    train_tar = spectra_df[['rs']].loc[spectra_df['pressure'] %tr!=0]
    #select target column and rows where pressure is odd
    test_tar = spectra_df[['rs']].loc[spectra_df['pressure'] %tr==0]

if predict_features ==2: #--------------------------------------------------
    scaler_tar = StandardScaler().fit(spectra_df[['rs', 'xs']].values)
    #select columns z and phase and rows where pressure is even
    train_tar = spectra_df[['rs', 'xs']].loc[spectra_df['pressure'] %tr!=0]
    #select columns #2 and #3 and rows where column #1 is odd
    test_tar = spectra_df[['rs', 'xs']].loc[spectra_df['pressure'] %tr==0]

train_tar_s = scaler_tar.transform(train_tar.values)


#%% create the model
train_start_time = timer()
model = Sequential([#Dropout(.05, input_shape=(train_inp_s.shape[1:])),
                    Dense(350, activation='relu',
                          input_shape=(train_inp_s.shape[1:])),
                          Dropout(.2, seed=1),
                    Dense(250, activation='relu'),
                    Dropout(.2, seed=1),
                    Dense(150, activation='relu'),
                    Dropout(.2, seed=1),
                    Dense(50, activation='relu'),
                    Dropout(.2, seed=1),
                    Dense(16, activation='relu'),
                    Dropout(.2, seed=1),
                    Dense(8, activation='relu'),
                    Dropout(.2, seed=1),
                  Dense(predict_features)])
model.compile(loss='mean_squared_error', optimizer='adam')

#%% train model
'''
epochs = 1000

history= model.fit(train_inp_s, train_tar_s,
                   validation_split=0.15, epochs=epochs,
                   batch_size=500, verbose=2)

print('training time = %.1f sec (%.2f min)' %(timer()-train_start_time,
                                 (timer()-train_start_time)/60))
# plot loss
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.semilogy(loss, c='k') # Plot training loss
plt.semilogy(val_loss, c='r')
plt.xlabel('Epoch', fontsize=labelsize)
plt.ylabel('Loss', fontsize=labelsize)
plt.legend(['training', 'validation'], fontsize=labelsize)
plt.show()
'''


#%% try other regressions


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# designate regression model
#model = DecisionTreeRegressor(max_depth=100)
model = RandomForestRegressor(n_estimators=150, max_depth=150, verbose=2, random_state=0)

#model = GradientBoostingRegressor(n_estimators=1500, max_depth=250, verbose=1) 
  
#model = SVR(kernel='poly', degree=4, verbose=True, C=1, epsilon=1)

#model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=3, random_state=0)

model.fit(train_inp_s, train_tar_s)




#%% get model and predictions

#modeled training data
trained_model_s = model.predict(train_inp_s)
trained_model = scaler_tar.inverse_transform(trained_model_s)

# predictions from testing data
predictions_s = model.predict(test_inp_s)
predictions = scaler_tar.inverse_transform(predictions_s)

if predict_features == 1:
    plt.plot(np.array(train_tar['rs']), label='exp. Rs')
    plt.plot(trained_model, label='modeled Rs')
    plt.legend(); plt.show()
    
    plt.plot(np.array(test_tar['rs']), label='exp. Rs')
    plt.plot(predictions, label='predicted Rs')
    plt.legend(); plt.show()
    
if predict_features == 2:    
    plt.plot(np.array(train_tar['xs']), label='exp. Rs')
    plt.plot(trained_model[:,1], label='modeled Rs')
    plt.legend(); plt.show()
    
    plt.plot(np.array(test_tar['xs']), label='exp. Rs')
    plt.plot(predictions[:,1], label='predicted Rs')
    plt.legend(); plt.show()
    
    plt.plot(np.array(train_tar['rs']), label='exp. Xs')
    plt.plot(trained_model[:,0], label='modeled Xs')
    plt.legend(); plt.show()

    plt.plot(np.array(test_tar['rs']), label='exp. Xs')
    plt.plot(predictions[:,0], label='predicted Xs')
    plt.legend(); plt.show()



#%% reshape predictions into individual spectra

if predict_features == 1:  
    modeled_rs = np.reshape(trained_model, (len(freq),-1), order='F')
    predicted_rs = np.reshape(predictions, (len(freq),-1), order='F')

    actual_rs = np.reshape(np.array(test_tar['rs']), (len(freq),-1), order='F')

    for i in range(len(predicted_rs[0])):
        plt.plot(freq, predicted_rs[:,i], label='prediction'+format(i))
        plt.scatter(freq, actual_rs[:,i], s=3, label='actual'+format(i))
    plt.xlabel('Frequency (MHz)', fontsize=labelsize)
    plt.ylabel('Phase (deg)', fontsize=labelsize)
    #plt.legend()
    plt.show()

    #avg_error = np.average(percent_error(predictions, test_tar['phase']))
    #print('avg percent error = '+format(avg_error))

if predict_features == 2:
    modeled_rs = np.reshape(trained_model[:,0], (len(freq),-1), order='F')
    modeled_xs = np.reshape(trained_model[:,1], (len(freq),-1), order='F')
    
    predicted_rs = np.reshape(predictions[:,0], (len(freq),-1), order='F')
    predicted_xs = np.reshape(predictions[:,1], (len(freq),-1), order='F')
    
    actual_rs = np.reshape(np.array(test_tar['rs']), (len(freq), -1), order='F')
    actual_xs = np.reshape(np.array(test_tar['xs']), (len(freq),-1), order='F')

    for i in range(len(predicted_rs[0])):
        plt.plot(freq, predicted_rs[:,i], label='prediction'+format(i))
        plt.scatter(freq, actual_rs[:,i], s=3, label='actual'+format(i))
    plt.xlabel('Frequency (MHz)', fontsize=labelsize)
    plt.ylabel('Rs (Ohm)', fontsize=labelsize)
    plt.legend(); plt.show()
    
    for i in range(len(predicted_rs[0])):
        plt.plot(freq, predicted_xs[:,i], label='prediction'+format(i))
        plt.scatter(freq, actual_xs[:,i], s=3, label='actual'+format(i))
    plt.xlabel('Frequency (MHz)', fontsize=labelsize)
    plt.ylabel('Xs (S)', fontsize=labelsize)
    #plt.legend()
    plt.show()

    #avg_error = np.average(percent_error(predictions[:,1], test_tar['phase']))
    #print('avg percent error = '+format(avg_error))



