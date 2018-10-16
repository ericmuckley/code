import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%%
def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    
    
def simulate_isotherm(pressures, isotherm_type):
    '''create a simulated isotherm using array of pressures
    and the designated isotherm type. Pressures may range from
    0 to 1, which is saturation pressure of the analyte.'''
    
    if isotherm_type == 1:
        noise = np.random.random(len(pressures))
        signal = np.log(pressures) - np.min(np.log(pressures))
        total = signal + noise
        return (total-np.min(total)) / np.max(total)

    if isotherm_type == 2:
        noise = np.random.random(len(pressures))*2
        signal = np.log(pressures*100)  + np.power(pressures+.5, 4)
        signal  - np.min(signal)
        total = signal + noise
        return (total-np.min(total)) / np.max(total)
  
    if isotherm_type == 3:
        noise = np.random.random(len(pressures))
        signal = np.power(pressures+1, 4)
        total = signal + noise
        return (total-np.min(total)) / np.max(total)
      
    else:
        return 'Isotherm type must be 1, 2, 3, 4 or 5.'
  
    

def classify_isotherm(derivs):
        #classify istherm type based on its derivatives in 3 regions:
        #low, mid, and high analyte pressures
        
        if derivs[0] > derivs[1] > derivs[2]:
            return('1')
        
        if derivs[0] > derivs[1] and derivs[2] > derivs[1]:
            return('2')
        
        if derivs[2] > derivs[1] > derivs[0]:
            return('3')
        
        else:
            return('0')



def get_derivs(pressures, loadings, window=3):
    #get derivative at different loadings: low, mid, and high using specified
    #window size (number of points) at each regime
    
    delta_loadings_low = loadings[window] - loadings[0]
    delta_p_low = pressures[window] - pressures[0]
    deriv_low = np.abs(delta_loadings_low / delta_p_low)
    
    delta_loadings_high = loadings[-1] - loadings[-window]
    delta_p_high = pressures[-1] - pressures[-window]
    deriv_high = np.abs(delta_loadings_high / delta_p_high)
    
    cent = int(len(pressures)/2)
    delta_loadings_mid = loadings[cent+int(window/2)] - loadings[cent-int(window/2)]
    delta_p_mid = pressures[cent+int(window/2)] - pressures[cent-int(window/2)]
    deriv_mid = np.abs(delta_loadings_mid / delta_p_mid)
    
    return deriv_low, deriv_mid, deriv_high




#%% create isotherms
#possible types of isotherms
isotherm_types = [1,2,3]
#isotherm pressures
pressures = np.linspace(2, 98, 15) / 100

#create some random isotherms and try to classify them
n = 35

df = pd.DataFrame(index=np.arange(0, n),
                  columns=['pressures', 'loading', 'type',
                           'deriv_low', 'deriv_mid', 'deriv_high'])
isotherm_mat = pressures

for i in range(n):
  
    #randomly select an isotherm type
    type0 = random.choice(isotherm_types)
  
    #simulate raw isotherm data
    loadings = {'raw': simulate_isotherm(pressures, type0)}
    #normalize loadings
    loadings['norm'] = loadings['raw'] / np.max(loadings['raw'])
    
    #get derivative at different loadings: low, mid, and high using specified
    #window size (number of points) at each regime
    deriv_low, deriv_mid, deriv_high = get_derivs(pressures, loadings['norm'],
                                                  window=3)
    
    #classfy the isotherm based on some basic rules
    #prediction = classify_isotherm((deriv_low, deriv_mid, deriv_high))

    #populate dictionary for saving isotherms
    df.iloc[i] = [pressures, loadings['norm'], type0,
                  deriv_low, deriv_mid, deriv_high]

    #plt.plot(pressures, loadings['norm'])
    
    isotherm_mat = np.column_stack((isotherm_mat, loadings['norm']))
    


#%% classify
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in
    Parameters
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns: xx, yy : ndarray
    """
    x_min, x_max = x.min() - 2, x.max() + 2
    y_min, y_max = y.min() - 2, y.max() + 2
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    Parameters
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



X_tot = np.column_stack((np.divide(df['deriv_low'], df['deriv_mid']),
        np.divide(df['deriv_high'], df['deriv_mid'])))
    
y_tot = np.array(df['type']).astype(int)


for i in range(10, len(df)+5, 5):

    X = X_tot[:i,:]
    y = y_tot[:i]

    print(np.shape((y)))

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1e5  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.SVC(kernel='rbf', gamma=0.1, C=C),
              DecisionTreeClassifier(max_depth=4))
    
    models = (clf.fit(X, y) for clf in models)
    
    # title for the plots
    titles = ('SVM, linear kernel',
              'SVM, RBF kernel',
             'Decision Tree')
    
    # Setup grid for plotting.
    fig, sub = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    
    fig = plt.gcf()
    fig.set_size_inches(10,3.5)
    
    for clf, title, ax, in zip(models, titles, sub.flatten()):
        
        score = round(clf.score(X,y), 3)
        
        plot_contours(ax, clf, xx, yy, alpha=0.2, cmap=plt.cm.RdYlBu)
        ax.scatter(X0, X1, c=y, s=20, edgecolors='k',
                   cmap=ListedColormap(['r', 'y', 'b']))
        
        ax.set_xlim(-1, 10)
        ax.set_ylim(-0.5, 5)
        ax.set_xlabel('$dA/dP_{L/M}$', fontsize=18)
        ax.set_ylabel('$dA/dP_{H/M}$', fontsize=18)
        ax.set_title(title, fontsize=18)
        label_axes('$dA/dP_{L/M}$', '$dA/dP_{H/M}$')
        
        ax.text(-.02, 4.5, 'n: '+format(i)+',', fontsize=14)
        ax.text(3.2, 4.5, 'score: '+format(score), fontsize=14)
    
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.2)
    save_pic_filename = 'image_dump\fig_'+format(str(i).zfill(2))+'.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=150)
    
    plt.show()
