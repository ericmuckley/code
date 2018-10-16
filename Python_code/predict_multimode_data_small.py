import glob, os, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
#import scipy.signal as filt
#import time
#import datetime
#from scipy.signal import medfilt
#from scipy.optimize import curve_fit
#from scipy.signal import savgol_filter
#from scipy.interpolate import splrep
#from scipy.interpolate import splev
#from sklearn.preprocessing import StandardScaler

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

#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout




def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)


def set_up_ann():
    #set up model
    model = Sequential([#Dropout(.05, input_shape=(train_inp_s.shape[1:])),
                    Dense(400, activation='relu',
                          input_shape=(train_in_df.shape[1:])),
                          Dropout(.2, seed=1),
                    #Dense(20, activation='relu'),
                    #Dense(6, activation='relu'),
                  Dense(1)])
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # train model
    epochs = 900
    history= model.fit(train_in_df, train_out_df,
                       validation_split=0.15, epochs=epochs,
                       batch_size=500, verbose=1)
    # plot loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.semilogy(loss, c='k') # Plot training loss
    plt.semilogy(val_loss, c='r')
    label_axes('Epoch', 'Loss')
    plt.legend(['training', 'validation'], fontsize=12)
    plt.show()




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
    return 100*np.abs(np.divide(diff,X))

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






#%% import all pickled data and combine into single dataframe
# reset the indices on each dataframe    
# suffix 'FULL' significes use of interpolated RH values (189 total),
# spanning 2-96% RH range in 0.5% steps

raw_eis0 = pd.read_pickle(
        'exp_data\\pp_eis_ml_small_full.pkl').reset_index(drop=True)

raw_iv0 = pd.read_pickle(
        'exp_data\\pp_iv_ml_small_full.pkl').reset_index(drop=True)

raw_qcm0 = pd.read_pickle(
        'exp_data\\pp_qcm_ml_small_full.pkl').reset_index(drop=True)

raw_optical0 = pd.read_pickle(
        'exp_data\\pp_optical_ml_small_full.pkl').reset_index(drop=True)


df_full = pd.concat([raw_eis0,
                     raw_iv0,
                     raw_qcm0,
                     raw_optical0,
                     ], axis=1)
    
#remove duplicated RH columns
df_full = df_full.loc[:,~df_full.columns.duplicated()]

#drop columns to omit from model (if they are assumed unmeasured, etc.) 
df = df_full.drop(['qcm_df7', 'qcm_df9', 'qcm_dd5',
                   'qcm_dd7', 'qcm_dd9', 'qcm_dmu5', 'qcm_dmu7',
                   'qcm_dmu9', 'qcm_deta5','qcm_deta7','qcm_deta9'
                   ], axis=1)
#df = df_full.copy()

#%%divide RH list into train and test sets
rh_list = np.unique(df['rh'])
training_rh = np.array([])


#%% USER INPUTS

#set which RH conditions to use for training
train_by_rh = False
if train_by_rh: #select training by selection of RH values:
    for rh in rh_list:
        if rh<11 or 22<rh<31 or 43<rh<52 or 64<rh<74 or 85<rh<100:
            training_rh = np.append(training_rh, rh)
else: #select training by selection of df rows
    for i in range(len(df)):
        if i % 10 == 0:
            training_rh = np.append(training_rh, df['rh'].iloc[i])


#set which variables to use as targets
target_vars = ['qcm_df5']

#select regression model
#model = RandomForestRegressor(n_estimators=150, max_depth=150,
#                              verbose=1, random_state=0)
model = AdaBoostRegressor(n_estimators=150, learning_rate=1)
#model = DecisionTreeRegressor(max_depth=150)
#model = GradientBoostingRegressor(n_estimators=150, max_depth=150, verbose=1)
#model = BaggingRegressor(base_estimator=SVR(kernel='poly', degree=2,verbose=True, C=1e-3,epsilon=1e-9),
#                         n_estimators=10, max_features=1)
#model = SVR(kernel='linear', degree=3, verbose=True, C=1e-3, epsilon=1e-6)
#model = linear_model.Ridge(alpha=1e-3)
#model = linear_model.Lasso(alpha=1e-9)
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
    set_up_ann()
else: 
    model.fit(train_in_df.values, train_out_df.values)








#%% plot measurement and predictions
    
modeled_data = model.predict(train_in_df.values)  

plt.scatter(train_out_df.values, modeled_data, s=3, c='k')
label_axes('Actual', 'Model')
plt.title('Model', fontsize=18)
plt.show()

predicted_data = model.predict(test_in_df.values)

plt.scatter(test_out_df.values, predicted_data, s=3, c='k')
label_axes('Actual', 'Predicted')
plt.title('Prediction', fontsize=18)
plt.show()

#plot all predictions and measured values in 1 series
plt.scatter(np.arange(len(predicted_data)),
         test_out_df.values, c='k', label='measured')
plt.scatter(np.arange(len(predicted_data)),
         predicted_data, c='r', label='predicted')
label_axes('Point #', 'Value')
plt.legend()
plt.show()


#percent difference between actual and predicted
percent_error = get_percent_err(test_out_df, predicted_data)
avg_error = np.mean(percent_error)
print('AVG. ERROR = %.6f' %(avg_error))

plt.scatter(testing_rh, percent_error)
label_axes('RH (%)', 'Percent error')
plt.title('Percent error', fontsize=18)
plt.show()

coef_df = pd.DataFrame(data=np.reshape(list(test_in_df), (-1,)),
                       columns=['feature'])

#coef_df['coef'] = np.abs(np.ravel(model.coef_))
coef_df['coef'] = np.abs(np.ravel(model.feature_importances_))

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
results['rh'] = testing_rh
results['percent_error'] = percent_error


#%% correlation coefficient

show_correlations = False

if show_correlations:
    mic_list = np.empty((0,2))
    
    for var in list(test_in_df):
        #maximum information coefficient
        mic0, r0 = get_mic(train_in_df[var].values,
                                np.reshape(train_out_df.values, (-1,)))
        mic_list = np.vstack((mic_list, np.array([mic0, r0])))
    
    x_fake = np.arange(len(mic_list))   
    
    #plot MIC
    plt.bar(x_fake, np.abs(mic_list[:,0]), align='center')
    plt.xticks(x_fake, list(test_in_df), rotation='vertical')
    plt.title('MIC')
    plt.show()
    
    plt.bar(x_fake, np.abs(mic_list[:,1]), align='center')
    plt.xticks(x_fake, list(test_in_df), rotation='vertical')
    plt.title('Pearson r')
    plt.show()



