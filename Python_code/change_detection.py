import numpy as np
import matplotlib.pyplot as plt


def single_exp(x, a, tau, y0):
    return y0 + a*np.exp(-(x)/tau)

x  = np.arange(-20, 160, 1).astype(float)+1
noise = np.random.random(len(x))/.8
#y_exp = np.append( np.full(15, 0), single_exp(x[15:], 2, 1, -.05))
y_exp = np.append(np.full(90, 0), 1+np.full(90, 0))
y = noise + y_exp

  
pad = 10

change = np.zeros_like(x).astype(float) 
norm_diff = np.zeros_like(x).astype(float)
norm_std =  np.zeros_like(x).astype(float)
mean = np.zeros_like(x).astype(float)

for i in range(2*pad, len(x)):
    #get most recent array and previous array, each of length 'pad',
    #with one unused buffer point in between
    y2, y1 = y[i-pad:i], y[i-2*pad:i-pad]
    
    #get normalized diff between present window and previous window
    norm_diff[i] = np.abs((
            np.mean(y2)-np.mean(y1))/((np.mean(y2)+np.mean(y1))/2))
    
    #get noramlized std of window
    norm_std[i] = np.abs(np.std(y2))#/np.mean(y2))

    mean[i] = np.mean(y2)

    #is change significant?
    if norm_diff[i] / norm_std[i] > 2: change[i-3] = 1

all_data = np.column_stack((x, y, mean, norm_std, y+norm_std, y-norm_std, norm_diff))

plt.scatter(x, y, alpha=.5, label='data')
plt.plot(x, norm_diff, label='diff')
plt.plot(x, norm_std, label='std')
plt.plot(x, norm_diff/norm_std/10, label='diff/std')
plt.plot(x, change, linestyle=':', c='k', label='change?')
plt.legend()
plt.show()

