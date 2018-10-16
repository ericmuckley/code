import glob, os, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.signal as filt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from scipy.interpolate import splrep
from scipy.interpolate import splev

from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RandomizedLasso
from sklearn.decomposition import PCA

from sklearn.feature_selection import f_regression

from sklearn.svm import SVR
from sklearn.metrics import r2_score




def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)



def plot_heatmap(Xf, Yf, Zf):
    #plot heatmap using 3 columns: X, Y, and Z data
    
    # create x-y points to be used in heatmap
    xf = np.linspace(Xf.min(),Xf.max(),100)
    yf = np.linspace(Yf.min(),Yf.max(),100)
    # Z is a matrix of x-y values
    zf = griddata((Xf, Yf), Zf, (xf[None,:], yf[:,None]), method='cubic')
    # Create the contour plot
    CSf = plt.contourf(xf, yf, zf, 100, cmap=plt.cm.rainbow, 
                       vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))
    plt.colorbar()








#%% import data
   
#folder with measured data files
folder = glob.glob('C:\\Users\\a6q\\exp_data\\pedotpss_ML_spectra/*')
print('found ' + format(len(folder)) + ' data files')

qcm_files = [file for file in folder if 'spec_mat' in file]
print('found ' + format(len(qcm_files)) + ' QCM data files')

#overtones = np.array([os.path.splitext(file)[0].split(
#                'mat_')[1] for file in qcm_files]).astype(int)
overtones = [0.073, 0.16, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]


#get pressure values
pressures = np.array(list(pd.read_table(folder[-1]))[1:]).astype(int)



#%% plot heat maps of QCM response


#make long arrays to hold data from every file together
long_rh, long_f, long_g = np.array([]), np.array([]), np.array([])
long_overtones = np.array([])

#loop over every file in folder
for j in range(len(qcm_files)):
    
    
    #read file
    file0 = pd.read_table(qcm_files[j])
    
    
    #reshape spec_mat into columns for heatmap plotting
    Xf, Yf, Zf = np.array([]), np.array([]), np.array([])
    
    
    #loop over each pressure
    for i in range(len(pressures)):

        #long arrays to hold daya from every file
        
        long_rh = np.append(long_rh, np.repeat(pressures[i], len(file0)))
        long_f = np.append(long_f, file0.values[:,0])
        long_g = np.append(long_g, file0.values[:,i+1])
        long_overtones = np.append(long_overtones,
                                   np.repeat(overtones[j], len(file0)))
        
   
'''
        #for plotting of heatmaps
        
        #set X values = pressure 
        Xf = np.append(Xf, np.repeat(pressures[i], len(file0)))
        #set Y values = frequency
        Yf = np.append(Yf, file0.values[:,0])
        #set Z values = G
        Zf = np.append(Zf, file0.values[:,i+1])
    
    # create x-y points to be used in heatmap
    xf = np.linspace(Xf.min(),Xf.max(),100)
    yf = np.linspace(Yf.min(),Yf.max(),100)
    # Z is a matrix of x-y values
    zf = griddata((Xf, Yf), Zf, (xf[None,:], yf[:,None]), method='cubic')
    # Create the contour plot
    CSf = plt.contourf(xf, yf, zf, 100, cmap=plt.cm.rainbow, 
                       vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))
    plt.colorbar()
    label_axes('RH (%)', 'F$_0$ (Hz/cm$^2$)')
    plt.show()
'''


#[long_overtones[i] = ]
all_spec = np.column_stack((long_overtones, long_rh, long_f, long_g))



#%% organize data for training / testing models 





#overtone to predict
n0 = 9

#training data
train_in = np.array([row[:-1] for row in all_spec if row[0] != n0])
train_tar = np.array([row[-1] for row in all_spec if row[0] != n0])

#target data
test_in = np.array([row[:-1] for row in all_spec if row[0] == n0])
test_tar = np.array([row[-1] for row in all_spec if row[0] == n0])







# select regression model

model = DecisionTreeRegressor(max_depth=50)
#model = RandomForestRegressor(n_estimators=20, max_depth=20,
#                              verbose=1, random_state=0)
#model = GradientBoostingRegressor(n_estimators=50, max_depth=50, verbose=1)
#model = SVR(kernel='poly', degree=2)#, verbose=True, C=1, epsilon=1)
#model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=3, random_state=0)



#fit model
model.fit(train_in, train_tar)




#%%
#get modeled training data
training_model = model.predict(train_in)    

plt.plot(np.arange(len(train_in))[::120], train_tar[::120], label='targets')
plt.plot(np.arange(len(train_in))[::120], training_model[::120], label='model')
label_axes('Sample', 'Model (G)')
plt.legend(fontsize=14)
plt.show()




#get prediction
prediction00 = model.predict(test_in)

#reshape prediction into 2D array
prediction =  np.reshape(prediction00, (len(file0),-1), order='F')




#plot heatmap of prediction
#set X values = pressure 
Xf = test_in[:,1]
#set Y values = frequency
Yf = test_in[:,2]
#set Z values = G
Zf = prediction00

plot_heatmap(Xf, Yf, Zf)
plt.title('Model prediction', fontsize=18)
label_axes('RH (%)', 'F$_0$ (Hz/cm$^2$)')
plt.show()




print('f_regression = '+format(f_regression(train_in, train_tar)))

#show avg. score of model
print('score = '+format(model.score(test_in, test_tar)))

print('feature importances = '+format(model.feature_importances_))

print('PCA explained variance ratio = '+format(
        PCA().fit(train_in).explained_variance_ratio_))


#%%
#get prediction error wrt RH
err_raw = []
for sample in range(len(test_tar)):
    
    err0 = np.abs(test_tar[sample] - prediction00[sample])*100
    err_raw.append(err0)

    
err = np.reshape(err_raw, (len(file0),-1), order='F')   
err_avg = np.average(err, axis=0) 


#plot heatmap of prediction error
plot_heatmap(test_in[:,1], test_in[:,2], err_raw)
plt.title('Prediction residual (%)', fontsize=18)
label_axes('RH (%)', 'F$_0$ (Hz/cm$^2$)')
plt.show()

plt.plot(pressures, err_avg)
label_axes('RH (%)', 'Avg. residual (%)')
plt.show()


