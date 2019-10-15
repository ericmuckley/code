# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:47:55 2018

@author: a6q
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.interpolate as inter

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn import linear_model
#from scipy.stats import spearmanr
#from scipy.stats import pearsonr
from minepy import MINE
#from minepy import pstats, cstats




from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout





def config_plot(xlabel='x', ylabel='y', size=16,
               setlimits=False, limits=[0,1,0,1]):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.rcParams.update({'figure.autolayout': True})
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    #set axis limits
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))



def set_up_ann():
    #set up model
    model = Sequential([#Dropout(.05, input_shape=(train_inp_s.shape[1:])),
                    Dense(400, activation='relu',
                          input_shape=(train_in_df.shape[1:])),
                          Dropout(.2, seed=1),
                    Dense(20, activation='relu'),
                    #Dense(6, activation='relu'),
                  Dense(1)])
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # train model
    history= model.fit(train_in_df, train_out_df,
                       validation_split=0.15, epochs=1600,
                       batch_size=500, verbose=1)
    # plot loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.semilogy(loss, c='k') # Plot training loss
    plt.semilogy(val_loss, c='r')
    config_plot('Epoch', 'Loss')
    plt.legend(['training', 'validation'], fontsize=12)
    plt.show()
    return model



def get_mic(x,y):
    #get maximum information coefficient and pearson r value
    r = np.corrcoef(x, y)[0, 1]
    mine = MINE(alpha=0.4, c=15, est='mic_e')
    mine.compute_score(x, y)
    mic = mine.mic()
    return mic, r

def get_percent_diff(X, y):
    #get percent difference between two arrays
    X, y = np.ravel(X), np.ravel(y)
    diff = np.abs(np.subtract(y, X))
    avg = np.add(y, X)/2
    return 100*np.abs(np.divide(diff,avg))

def get_percent_err(X, y):
    X, y = np.ravel(X), np.ravel(y)
    #get percent difference between two arrays
    diff = np.subtract(y, X)
    return 100*np.abs(np.divide(diff,0.5))#X))

def get_train_test_df(rh_list, training_rh, testing_rh):
    #get dataframe showing which RH values are used for training/testing 
    train_test_df = pd.DataFrame(data=rh_list, columns=['rh'])
    train = np.zeros_like(rh_list)
    test = np.zeros_like(rh_list)
    for i, rh in enumerate(rh_list):
        if rh in training_rh: train[i] = 1
        if rh in testing_rh: test[i] = 1
    train_test_df['test'] = test
    train_test_df['train'] = train
    return train_test_df



#%% import all data 

data_raw = pd.read_table('C:\\Users\\a6q\\exp_data\\lab_on_qcm_ML_data.txt')

df_full = pd.DataFrame(columns=data_raw.columns)
rh_list = np.linspace(2, 96, 189)
df_full['rh'] = rh_list



#%% smooth and densify training data using spline

for col in data_raw.columns[1:]:
    #normalize data column
    data0 = np.array(data_raw[col])
    data0 -= np.amin(data0)
    data0 = data0 / np.amax(data0)
    
    #fit to spline
    fit_spline = inter.UnivariateSpline(data_raw['rh'], data0, s=1e-3, k=2)
    spline = fit_spline(df_full['rh'])
    df_full[col] = spline
    
    
    #plot spline fits
    #plt.scatter(data_raw['rh'], data0)
    #plt.scatter(df_full['rh'], spline, s=2)
    #config_plot('RH (%)', col)
    #plt.show()


#%% USER INPUTS

#drop columns to omit from model (if they are assumed unmeasured, etc.) 
df = df_full.drop(['dd9','df9','deta9','dmu9', 
                    #'dmu1',
                    #'df1','df5', 'dd1','dd5', 'dmu5','deta1', 'deta5',
                    'logZ100mHz','logZ10Hz','logfreq45phase',
                    'logIDC',
                    'deltamaxint',
                    'deltamaxwl'
                     ], axis=1)
    
#set which variables to use as targets
target_vars = ['dmu1']



#divide RH list into train and test sets
rh_list = np.unique(df['rh'])
training_rh = np.array([])

#set which RH conditions to use for training
train_by_rh = False
if train_by_rh: #select training by selection of RH values:
    for rh in rh_list:
        if rh<11 or 22<rh<31 or 43<rh<52 or 64<rh<74 or 85<rh<100:
            training_rh = np.append(training_rh, rh)
else: #select training by selection of df rows
    for i in range(len(df)):
        if i % 4 == 0:
            training_rh = np.append(training_rh, df['rh'].iloc[i])

#select regression model
#model = RandomForestRegressor(n_estimators=150, max_depth=150, verbose=1, random_state=0)
#model = AdaBoostRegressor(n_estimators=200, learning_rate=0.1)
#model = DecisionTreeRegressor(max_depth=250)
#model = GradientBoostingRegressor(n_estimators=150, max_depth=150, verbose=1)
#model = BaggingRegressor(base_estimator=SVR(kernel='poly', degree=2,verbose=True, C=1e-3,epsilon=1e-9),
#                         n_estimators=10, max_features=1)
#model = SVR(kernel='poly', degree=3, verbose=True, C=1e3, epsilon=1e-6)
#model = linear_model.Ridge(alpha=1e-2)
model = linear_model.Lasso(alpha=1e-4)
#model = linear_model.LinearRegression()


#%% divide data into input / target matrices

#RH values to use as targets
testing_rh =  np.setdiff1d(rh_list, training_rh)
train_test_df = get_train_test_df(rh_list, training_rh, testing_rh)
input_vars = np.setdiff1d(list(df), target_vars)
#divide dataframes based on input/target RH values
df_train0 = df.loc[df['rh'].isin(training_rh)]
df_test0 = df.loc[df['rh'].isin(testing_rh)]
#divide dataframes further based on input/target variable names
train_in_df = df_train0[input_vars]
train_out_df = df_train0[target_vars]
test_in_df = df_test0[input_vars]
test_out_df = df_test0[target_vars]



#%% for using ANN as model

#ann(train_in_df)
ann = False
if ann:
    model = set_up_ann()
else: 
    model.fit(train_in_df.values, train_out_df.values)



#%% plot measurement and predictions
    
modeled_data = model.predict(train_in_df.values)  

plt.scatter(train_out_df.values, modeled_data, s=3, c='k')
config_plot('Actual', 'Model')
plt.title('Model', fontsize=18)
plt.show()

predicted_data = model.predict(test_in_df.values)

plt.scatter(test_out_df.values, predicted_data, s=3, c='k')
config_plot('Actual', 'Predicted')
plt.title('Prediction', fontsize=18)
plt.show()

#plot all predictions and measured values in 1 series
plt.scatter(np.arange(len(predicted_data)),
         test_out_df.values, c='k', label='measured')
plt.scatter(np.arange(len(predicted_data)),
         predicted_data, c='r', label='predicted')
config_plot('Point #', 'Value')
plt.legend()
plt.show()


#percent difference between actual and predicted
percent_error = get_percent_err(test_out_df, predicted_data)
avg_error = np.mean(percent_error)
print('AVG. ERROR = %.6f' %(avg_error))

plt.scatter(testing_rh, percent_error)
config_plot('RH (%)', 'Percent error')
plt.title('Percent error', fontsize=18)
plt.show()

coef_df = pd.DataFrame(data=np.reshape(list(test_in_df), (-1,)),
                       columns=['feature'])

if ann == False:
    coef_df['coef'] = np.abs(np.ravel(model.coef_))
    #coef_df['coef'] = np.abs(np.ravel(model.feature_importances_))
    
    np.arange(len(coef_df['coef']))   
    #plot model coefficients
    plt.bar(np.arange(len(coef_df['coef'])),
            coef_df['coef'], align='center')
    plt.xticks(np.arange(len(coef_df['coef'])),
               list(test_in_df), rotation='vertical')
    plt.title('Linear model coefficients')
    plt.show()


#%% create dataframe for exporting model results
results = pd.DataFrame(data=test_out_df.values, columns=['actual'])
results['predicted'] = predicted_data
results['percent_error'] = percent_error
results['rh'] = testing_rh








#%% correlation coefficient

show_correlations = False
if show_correlations:
    mic_list = np.empty((0,2))
    
    for var in list(test_in_df):
        #maximum information coefficient
        mic0, r0 = get_mic(train_in_df[var].values,
                                np.reshape(train_out_df.values, (-1,)))
        mic_list = np.vstack((mic_list, np.array([mic0, r0]))) 
    
    #plot MIC
    plt.bar(np.arange(len(mic_list)), np.abs(mic_list[:,0]), align='center')
    plt.xticks(np.arange(len(mic_list)), list(test_in_df), rotation='vertical')
    plt.title('MIC')
    plt.show()
    
    plt.bar(np.arange(len(mic_list)), np.abs(mic_list[:,1]), align='center')
    plt.xticks(np.arange(len(mic_list)), list(test_in_df), rotation='vertical')
    plt.title('Pearson r')
    plt.show()

#%% get correlation matrix
corr_mat = np.empty((len(df_full.columns), len(df_full.columns)))
for i, coli in enumerate(df_full.columns):
    for j, colj in enumerate(df_full.columns):
        corr0 = np.corrcoef(df_full[coli], df_full[colj])[0,1]
        corr_mat[i][j] =  corr0
        