# coding: utf-8

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#%% define functions

def seq_to_sig(seq0, points_per_minute=2, simulate_signal=True,
              coeffs = {'amp':1, 'tau': 10, 'offset':0, 'drift':0.0}):
    '''
    Creates a matrix out of 2-column sequence data: [times, pessures].
    Outputs: [times, relative times, pressures, relative pressures, 
    simulated signal constructed from exponential fits using 'coeffs'].
    Outputs 2D numpy array and Pandas dataframe populated with 
    'points_per_minute' number of points per minute of the original sequence.
    '''
    #initialize absolute time column
    abs_t, dt = np.linspace(0, np.sum(seq0[:,0]),
                        num = np.multiply(np.sum(seq0[:,0]),
                                          points_per_minute).astype(int),
                                          retstep=True)
                        
    #init dialize relative time (time since last pressure change)
    rel_t = np.zeros((np.sum(seq0[0,0])*points_per_minute).astype(int))
    #initialize initial pressure
    abs_p = np.ones_like(rel_t) * seq0[0,1]
    #relative pressure (pressure difference since last pressure change)
    rel_p = np.zeros_like(rel_t)
    #loop over steps in sequence and populate columns########################
    for i in range(1, len(seq0)):
        #numper of points to append during this step
        point_num = np.multiply(seq0[i,0], points_per_minute).astype(int)
        #append each column with appropriate values
        rel_t = np.append(rel_t, np.linspace(0, dt*point_num, num=point_num))
        abs_p = np.append(abs_p, np.full(point_num, seq0[i,1]))
        rel_p = np.append(rel_p, np.full(point_num, seq0[i,1] - seq0[i-1,1]))
    #put all columns toether
    seq_mat = np.array([abs_t, rel_t, abs_p, rel_p]).T
    
    #construct dataframe with results
    seq_mat_df = pd.DataFrame(seq_mat, 
                          columns=['abs_time', 'rel_time', 'abs_pressure',
                                   'rel_pressure'])
    #simulate exponential response signal using seq_mat_df and coeffs
    if simulate_signal == True:
        #construct signal 
        sim_sig = coeffs['offset'] + (
                coeffs['drift'] * seq_mat_df['abs_time']) - coeffs['amp']*(
                seq_mat_df['abs_pressure'] + np.multiply(
                seq_mat_df['rel_pressure'], np.exp(
                        -seq_mat_df['rel_time']/coeffs['tau'])))
        #loop over each point in signal
        for i in range(points_per_minute*seq0[0,0], len(seq_mat_df)):
            #if theres a step change:
            if seq_mat_df['rel_time'][i] == 0:
                #remove discontinuous jumps which may occur at step change
                sim_sig[i:] = sim_sig[i:] - (sim_sig[i] - sim_sig[i-1])
            else: pass
        #append simulated signal to dataframe
        seq_mat_df['sim_sig'] = sim_sig
    else: pass

    return seq_mat_df






# simulate expected signal using pressure sequence data
def exp_response(seq_df, amp, tau, offset, drift):
    '''
    Creates simulated sensor response using sequence dataframe:
    [abs_time, rel_time, abs_pressure, rel_pressure].
    Fit parameters for the exponential decays are:
    [amp (amplitude of exponential), tau (exponential time constant),
    offset (vertical offset of exponential),
    drift (scaling factor for linear drift)]
    '''
    points_per_minute = 2
    
    #build simulated signal from seq_out matrix
    #construct signal 
    exp_sig = offset + (drift * seq_df['abs_time']) - amp*(
            seq_df['abs_pressure'] + np.multiply(
            seq_df['rel_pressure'], np.exp(-seq_df['rel_time']/tau)))
    #loop over each point in signal
    for i in range(points_per_minute*seq0[0,0], len(seq_df)):
        #if theres a step change:
        if seq_df['rel_time'][i] == 0:
            #remove discontinuous jumps which may occur at step change
            exp_sig[i:] = exp_sig[i:] - (exp_sig[i] - exp_sig[i-1])
        else: pass
    return exp_sig







# calculate percent error from residuals   
def percent_error(measured, fit):
    '''
    Calculates percent error between measured and fitted signals.
    Percent differences is calculated from the ratio of residual to
    entire range of measured values.
    '''
    measured.flatten()
    fit.flatten()
    measured_range = np.abs(np.amax(measured) - np.amin(measured))
    residual = np.subtract(measured, fit)
    percent_error = 100*np.abs(np.divide(residual, measured_range))
    
    return percent_error
    





# create lagged input matrix
def lag_series(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = [], []
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0: names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else: names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg




#%% get sequence matrix



#seq0 = pd.read_table('sample_seq.txt', header=None).values
seq0 = np.array([[60,2], [60,40], [60,20], [60,10], [60,30], [60,15], [60,2]])
#seq0 = np.array([[90,0], [90,1], [90,0], [90,10], [90,0], [90,30],
#                 [90,0], [90,60], [90,0], [90,95], [90,0]])

#create dataframe and simulated signal from sequence
sim_sig_coeffs = {'amp':1, 'tau': 10, 'offset':0.0, 'drift':0.0}
seq_df = seq_to_sig(seq0, points_per_minute=6, coeffs=sim_sig_coeffs)

seq_df['sim_sig'] = seq_df['sim_sig'] + 3*(np.random.rand(len(seq_df))-.5)

#generate lagged features
seq_df_lagged0 = seq_df #lag_series(seq_df, 2, 1)

seq_df_lagged = seq_df_lagged0.copy()

# drop columns we don't want to predict
#seq_df_lagged.drop(seq_df_lagged.columns[[4,5,6,7,8]],
#                   axis=1, inplace=True)
print(list(seq_df_lagged))

seq_lagged_vals = seq_df_lagged.values


#%% normalize features
scaler = MinMaxScaler()
input_s = scaler.fit_transform(seq_lagged_vals)


#%% split into train and test sets

train_num = 1000
train = input_s[:train_num, :]
test = input_s[train_num:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]


train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))



#%% design network

#good network = 
#LSTM(800), Dropout(0.2, seed=1), LSTM(20), Dropout(0.2, seed=1), Denmse(1)

model = Sequential([LSTM(8, return_sequences=True,
                         input_shape=(train_X.shape[1], train_X.shape[2])),
                    Dropout(0.2, seed=1),
                    LSTM(20),
                    Dropout(0.2, seed=1),
                    Dense(1)])
        
model.compile(loss='mae', optimizer='adam')


#%% fit model

time0 = timer()

history = model.fit(train_X, train_y, epochs=1500, batch_size=750,
                    validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)

time1 = timer()


# plot history
plt.semilogy(history.history['loss'], c='k', label='loss')
plt.semilogy(history.history['val_loss'], c='r', label='validation loss')
plt.legend()
plt.show()



#%% make predictions

calc_model = model.predict(train_X).flatten()
prediction = model.predict(test_X).flatten()

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

#calc_model = calc_model.reshape((calc_model.shape[0], calc_model.shape[2]))
train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))

#%%
time = seq_lagged_vals[:, 0]
time_train = time[:train_num]
time_test = time[train_num:]




plt.scatter(time, input_s[:,4], s=3, alpha=.3, c='k', label='exp.')
plt.plot(time_train, calc_model, c='g', label='model')
plt.plot(time_test, prediction, c='r', label='prediction')

plt.xlabel('Time', fontsize=15)
plt.ylabel('Signal', fontsize=15)
plt.legend(fontsize=12)
plt.show()


print('train time = '+format(time1-time0)+' seconds')

percent_error = np.average(percent_error(input_s[train_num:,4], prediction))
print('avg. percent error = '+format(percent_error))

#%%


'''
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

'''





#%%




#fit measured signal and calculate residuals
'''
fit_params, _  = curve_fit(exp_response, seq_df, sig0)
fit = exp_response(seq_df, *fit_params)
percent_err = percent_error(sig0, fit)
'''



'''

#create simulated signal with noise
seq_df['sim_sig_noise']  = seq_df['sim_sig'] + 2*(
        np.random.rand(len(seq_df))-.5)







#%% organize data for training/testing

train_size = 600

Xnames = seq_df[['abs_time', 'rel_time', 'abs_pressure', 'rel_pressure', 'sim_sig']]
X = Xnames.values
y = seq_df['sim_sig_noise']

X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]




#%% plot results

plt.scatter(seq_df['abs_time'], seq_df['sim_sig_noise'],
            alpha=.3,s=3, label='exp.')
plt.plot(seq_df['abs_time'], seq_df['sim_sig'], lw=2, c='g', label='fit')
plt.plot(seq_df['abs_time'], seq_df['abs_pressure'], linewidth=1, c='b',
         label='pressure')
plt.legend(); plt.show()


timer1 = timer()
print('fit time = '+format(int(timer1-timer0)+1)+' sec')


#%% 

calc_coeffs, _  = curve_fit(exp_response, seq_df, seq_df['sim_sig'])

#update coefficent values using calculated fit parameters
for i, key in enumerate(sim_sig_coeffs):
    sim_sig_coeffs[key] = calc_coeffs[i]






'''


